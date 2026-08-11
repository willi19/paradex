"""Robot-side gated execution client (paradex-backed).

REPL loop:
  - 'q' + Enter  → query the policy with the current obs, print the plan
  - 'e' + Enter  → execute the most-recent plan on the robot (cartesian, 16 steps)
  - 'x' + Enter  → exit

The TODOs from the dryrun version are filled in using paradex controllers:
  - xArm6 cartesian:    paradex.io.robot_controller.get_arm('xarm').move(homo, is_servo=True)
                        → set_servo_cartesian_aa internally (m→mm, rotvec rad)
  - Inspire register:   paradex.io.robot_controller.get_hand('inspire', ip=True).move(raw)
  - FK link6:           paradex.vla.client.fk_link6 (same urdf path as training)
  - Front camera:       FLIR stream brought up via paradex (run_script
                        stream_client.py on every capture PC + DataCollector
                        + CameraFrameSource). NOT a RealSense — the prior
                        TODO comment was a hint, not a requirement.

This shares all protocol math (POS_SCALE, ROT_SCALE, INSPIRE_LIMITS_RAD,
hand register↔qpos, integrate_chunk, ARM_TO_HAND_OFFSET, safety check)
with src/stream/vla/viewer.py via paradex.vla.client.

Usage:
  PYTHONPATH=. python vla_test/robot_server/robot_client.py \\
      --server tcp://localhost:5600 \\
      --task_text "throw the pepsi can into the white basket" \\
      --urdf  vla_test/robot_server/xarm_inspire/xarm_inspire.urdf \\
      --log_dir logs/deploy/$(date +%Y%m%d_%H%M%S)

NOTE: xArm SDK allows ONE client at a time. Do NOT run viewer.py at the
same time as this script — pick one mode.
"""

import argparse
import json
import signal
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import yourdfpy

# Make this script robust when launched from outside the repo or without
# PYTHONPATH=.; otherwise Python can import an older installed paradex package.
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from paradex.io.camera_system.remote_camera_controller import remote_camera_controller
from paradex.io.capture_pc.data_sender import DataCollector
from paradex.io.capture_pc.ssh import run_script
from paradex.io.robot_controller import get_arm, get_hand
from paradex.utils.system import get_camera_list, get_pc_ip, get_pc_list
from paradex.vla.client import (
    ACTION_CHUNK_LENGTH,
    POS_SCALE,
    CameraFrameSource,
    PolicyClient,
    check_plan_safe,
    execute_plan,
    fk_link6,
    integrate_chunk,
    prepare_frame_for_vla,
    read_arm_qpos_rad,
    read_hand_vla_rad,
    vla_rad_to_hand_raw_register,
)
from scipy.spatial.transform import Rotation as R


# ------------------------------------------------------------------
# Frame handling: policy target frame vs the frame the xArm SDK commands in
# ------------------------------------------------------------------
# read_state feeds the policy `state.eef_*` = FK on link6. The policy's action
# chunk comes back in the SAME frame (verified on hardware: every plan's step 0
# sits 0.2-2.7 mm from the state it was conditioned on).
#
# xarm_controller.move() hands the pose to set_servo_cartesian_aa, which reads
# it in the arm's *TCP* frame, not link6. With a TCP configured (this rig:
# 138.4 mm z), sending a link6 target unchanged parks link6 ~135 mm short along
# tool z. The next query then sees that displaced state, returns a target near
# it, and the error repeats — a positive feedback loop that walks the arm ~135
# mm per execution. That is exactly what run 20260810_215510 did:
#   query start x = 526.8 -> 384.7 -> 253.6 -> 124.4 mm, then ErrorCode 23
#   (Joints Angle Exceed Limit), while each 16-step plan only spanned ~13 mm.
# Replaying the log against the measured TCP reproduces each post-execution
# state to 0.5-1.3 mm, versus 135 mm if the command were taken as link6.
#
# Fix: always convert the link6 target into the SDK's TCP frame with the LIVE
# measured transform (src/stream/vla/viewer.py does the same). Nothing here
# depends on the TCP being set to any particular value — only on measuring it.
ACTION_FRAME_TOL_M = 0.05   # |step0 - state| above this contradicts --action-frame


def _homo(pos_m, rotvec_rad):
    T = np.eye(4)
    T[:3, :3] = R.from_rotvec(rotvec_rad).as_matrix()
    T[:3, 3] = pos_m
    return T


def measure_link6_to_tcp(arm, urdf):
    """Live link6 -> SDK-command frame, MEASURED (base->TCP from the SDK vs
    base->link6 from URDF FK). This is the transform the controller actually
    applies, which is what matters — not what someone believes is configured."""
    q = read_arm_qpos_rad(arm)
    pos6, rot6 = fk_link6(urdf, q)
    code, aa = arm.arm.get_position_aa(is_radian=True)
    if code != 0:
        raise RuntimeError(f"get_position_aa failed with code {code}")
    T_link6 = _homo(pos6, rot6)
    T_tcp = _homo(np.asarray(aa[:3], dtype=np.float64) / 1000.0,
                  np.asarray(aa[3:], dtype=np.float64))
    return np.linalg.inv(T_link6) @ T_tcp


def resolve_command_transform(arm, urdf, action_frame):
    """Returns the `link6_to_sdk_tcp` to pass to execute_plan.

    action_frame='link6' : policy targets are link6 poses -> post-multiply by
                           the measured link6->TCP so the controller lands link6
                           where the policy asked.
    action_frame='tcp'   : policy targets are already SDK TCP poses -> send
                           unchanged (None).
    """
    print("[robot_client] measuring the xArm command frame ...")
    try:
        configured = arm.arm.tcp_offset
        print(f"  arm.tcp_offset (config) = {np.round(np.asarray(configured, float), 3).tolist()}"
              "  [x,y,z mm, roll,pitch,yaw deg]")
    except Exception as e:
        print(f"  arm.tcp_offset unavailable: {e}")

    T_live = measure_link6_to_tcp(arm, urdf)
    print(f"  measured link6->TCP = {np.round(T_live[:3, 3] * 1000, 2).tolist()} mm, "
          f"{np.degrees(R.from_matrix(T_live[:3, :3]).magnitude()):.2f} deg")

    if action_frame == "tcp":
        print("  --action-frame tcp — sending policy targets unchanged.")
        return None
    print("  --action-frame link6 — every target will be post-multiplied by the "
          "measured link6->TCP before it reaches set_servo_cartesian_aa.")
    return T_live


def check_action_frame(targets, state_pos, action_frame, tcp_norm_m, action_mode):
    """Cross-check --action-frame against what the policy actually returned.

    Only meaningful for ABSOLUTE actions: an absolute link6 target starts at the
    conditioning state, an absolute TCP target starts ~|link6->TCP| away from it.
    Under delta integration step 0 is state + delta*POS_SCALE by construction, so
    this comparison can never fail and proves nothing — say that instead of
    printing a green light that means nothing."""
    d = float(np.linalg.norm(np.asarray(targets[0]["target_pos_m"]) - state_pos))
    if action_mode == "delta":
        return True, (f"|step0-state| = {d * 1000:.1f} mm — not a frame test under "
                      "delta integration (targets are built from the state, so they "
                      "are link6-framed by construction)")
    if action_frame == "link6":
        ok = d <= ACTION_FRAME_TOL_M
        why = (f"|step0-state| = {d * 1000:.1f} mm "
               f"(<= {ACTION_FRAME_TOL_M * 1000:.0f} mm expected for link6-framed actions)")
        if not ok:
            why += (f"; {tcp_norm_m * 1000:.0f} mm would mean the actions are TCP-framed "
                    "— re-run with --action-frame tcp")
    else:
        ok = d >= 0.5 * tcp_norm_m
        why = (f"|step0-state| = {d * 1000:.1f} mm "
               f"(>= {0.5 * tcp_norm_m * 1000:.0f} mm expected for TCP-framed actions)")
        if not ok:
            why += "; this looks link6-framed — re-run with --action-frame link6"
    return ok, why


# Bound on a legitimate delta, taken from the checkpoint's own action statistics
# rather than from any client-side heuristic. `world_delta_action` is stored
# already divided by POS_SCALE / ROT_SCALE and clipped to +/-1, so unit space is
# the trained range: per-axis q99 = 0.14 and per-axis min/max = [-0.63, 0.60],
# i.e. a 3-axis delta can legitimately reach a norm of ~1.1. Anything with a
# component beyond +/-1 is outside what the column can even contain and means
# the reply is not unit-space deltas at all — a protocol break, not a fast move.
#
# An earlier version of this guard rejected at norm 0.18, which sat INSIDE the
# training distribution: the 20260811_001225 run peaked at 0.1621, within 10% of
# aborting the loop on a perfectly ordinary chunk. Per-step magnitude is already
# bounded elsewhere (check_plan_safe caps 0.10 m/step, interpolate_targets splits
# anything over 20 mm), so this check only has to catch the protocol break.
DELTA_UNIT_CLIP = 1.0        # the training column's own clip bound
DELTA_WARN_COMPONENT = 0.63  # largest per-axis value seen in the training set


def check_action_scale(chunk, action_mode):
    """Reject action chunks that are not in the trained unit space."""
    if action_mode != "delta":
        return True, "n/a for absolute actions"
    arm = np.asarray(chunk)[:ACTION_CHUNK_LENGTH, 0:6]
    peak_comp = float(np.abs(arm).max())
    peak_norm = float(np.linalg.norm(arm[:, 0:3], axis=1).max())
    msg = (f"peak |component| = {peak_comp:.3f} (unit space, clipped at "
           f"{DELTA_UNIT_CLIP:.0f} in training); peak step = "
           f"{peak_norm * POS_SCALE * 1000:.1f} mm")
    if peak_comp > DELTA_UNIT_CLIP:
        return False, msg + " — OUTSIDE unit space, the reply is not delta actions"
    if peak_comp > DELTA_WARN_COMPONENT:
        return True, msg + " — beyond the largest value in the training set"
    return True, msg


# ------------------------------------------------------------------
# State / camera / robot I/O (the former TODO blocks)
# ------------------------------------------------------------------

def read_state(robot_handles, urdf):
    """Returns (obs_for_policy, raw_state).

    Schema matches vla_test/robot_server/ping_server.py + the trained policy:
      state.eef_position       (3,)  float64  link6 in link_base (FK), meters
      state.eef_rotation       (3,)  float64  rotvec, link6 in link_base
      state.arm_joint_position (6,)  float64  joint1..6, radians
      state.dexhand_position   (6,)  float64  [thumb_yaw..pinky] radians in the
                                              TRAINING convention (open ->
                                              LIMITS), not the URDF joint angle
    """
    arm_joints = read_arm_qpos_rad(robot_handles["xarm"]).astype(np.float64)
    eef_pos, eef_rotvec = fk_link6(urdf, arm_joints)
    hand_rad = read_hand_vla_rad(robot_handles["hand"]).astype(np.float64)

    obs = {
        "state.eef_position":       eef_pos.astype(np.float64),
        "state.eef_rotation":       eef_rotvec.astype(np.float64),
        "state.arm_joint_position": arm_joints,
        "state.dexhand_position":   hand_rad,
    }
    raw = {
        "arm_joints_rad":    arm_joints.tolist(),
        "hand_rad_qpos":     hand_rad.tolist(),
    }
    return obs, raw


def read_camera(cam_handle, serial: str):
    """Latest BGR frame for `serial` → (480, 640, 3) RGB uint8.

    cam_handle is a CameraFrameSource (background-thread JPEG decoder).
    Blocks (up to 5s) until a frame is available so the very first query
    after startup doesn't fail before the stream warms up.
    """
    deadline = time.time() + 5.0
    while time.time() < deadline:
        bgr = cam_handle.latest(serial)
        if bgr is not None:
            return prepare_frame_for_vla(bgr)
        time.sleep(0.05)
    # available_serials() only lists serials this source actually DECODED, so on
    # its own it cannot distinguish "nothing arrives" from "the wrong camera is
    # being published". Report the raw collector keys too.
    arriving = sorted(cam_handle._dc.get_data().keys())
    raise RuntimeError(
        f"no frame for serial {serial} within 5s.\n"
        f"  serials arriving at the DataCollector: {arriving}\n"
        f"  serials decoded by this source        : {cam_handle.available_serials()}\n"
        f"  If {serial} is missing from the first list, the stream_client on its "
        f"capture PC is not publishing it — most often a stale stream_client from "
        f"an earlier session still owns PUB port 1234 (the new one dies on 'Address "
        f"already in use', and run_script sends its output to /dev/null). Check with:\n"
        f"    ssh -p 77 capture4@172.30.0.2 \"pgrep -af stream_client.py\""
    )


def send_to_robot(robot_handles, target, link6_to_sdk_tcp=None):
    """Cartesian send for ONE integrated target.

    paradex's xarm_controller.move(homo, is_servo=True) takes the 4x4 in
    METERS (link_base frame), does homo2aa internally → set_servo_cartesian_aa
    (m→mm scaling, rotvec radians) at its 100Hz control loop.

    paradex's inspire controller .move(raw) takes 6 register ints (0..1000).
    `integrate_chunk` already filled `target_hand_raw_reg` in register order.
    """
    arm = robot_handles["xarm"]
    hand = robot_handles["hand"]

    from paradex.vla.client import target_to_arm_homo
    arm.move(target_to_arm_homo(target, link6_to_sdk_tcp), is_servo=True)
    hand.move(target["target_hand_raw_reg"])


# ------------------------------------------------------------------
# Temporal ensembling
# ------------------------------------------------------------------
# With --exec-steps N < ACTION_CHUNK_LENGTH consecutive chunks overlap: after
# executing N steps of chunk A, chunk B is built from the pose A was commanded
# to reach, so B's step i predicts the same instant as A's step N+i. Averaging
# those two predictions halves the sampling noise on that instant.
#
# The noise is worth attacking: querying this server six times with a byte-identical
# image and state gave a mean deviation of 0.0175 against action components of
# 0.02-0.06 — roughly half of every command is sampling noise.
#
# Nothing overlaps at --exec-steps 16, so ensembling is a no-op there.

def blend_plans(new_targets, prev_tail, alpha):
    """Weighted average of two chunks' predictions for the same timesteps.

    alpha weights the NEW chunk (fresher observation); 1-alpha the carried-over
    tail. Rotations are averaged as rotations, not as raw rotvec components."""
    if not prev_tail:
        return new_targets
    out = []
    for i, a in enumerate(new_targets):
        if i >= len(prev_tail):
            out.append(a)
            continue
        b = prev_tail[i]
        rot = R.from_rotvec([a["target_rotvec_rad"], b["target_rotvec_rad"]]).mean(
            weights=[alpha, 1.0 - alpha]).as_rotvec()
        hand = (alpha * np.asarray(a["target_hand_rad_qpos"])
                + (1.0 - alpha) * np.asarray(b["target_hand_rad_qpos"]))
        out.append({
            "target_pos_m": (alpha * np.asarray(a["target_pos_m"])
                             + (1.0 - alpha) * np.asarray(b["target_pos_m"])),
            "target_rotvec_rad": rot,
            "target_hand_rad_qpos": hand,
            "target_hand_raw_reg": vla_rad_to_hand_raw_register(hand),
        })
    return out


# ------------------------------------------------------------------
# CLI helpers
# ------------------------------------------------------------------

def query_policy(policy, obs_for_policy, frame_rgb, task_text):
    return policy.query(obs_for_policy, frame_rgb, task_text)


def print_plan(targets, start_pos, start_rotvec):
    print(f"  start eef pos = {start_pos.round(3)}  rotvec = {start_rotvec.round(3)}")
    n = len(targets)
    for i in sorted({0, n // 2, n - 1}):
        t = targets[i]
        print(f"  step[{i:2d}]: pos={t['target_pos_m'].round(3)}  "
              f"rotvec={t['target_rotvec_rad'].round(3)}  "
              f"hand_reg={t['target_hand_raw_reg'].tolist()}")


def pc_owning(serial):
    """Which capture PC has this camera plugged into it (from pc.json)."""
    for pc_name in get_pc_list():
        if serial in get_camera_list(pc_name):
            return pc_name
    raise SystemExit(
        f"camera serial {serial} is not in any pc.json cam_list. "
        f"Configured PCs: {get_pc_list()}"
    )


def assert_daemon_responsive(pc_name, timeout_ms=3000, boot_grace_s=120):
    """Fail fast if the daemon cannot serve commands — but tell the truth about why.

    Two failures look identical from a ZMQ REQ socket and need opposite responses:

      not bound yet  camera_server_daemon.__init__ runs CameraLoader() BEFORE it
                     binds any socket (camera_server_daemon.py:9-25), so a daemon
                     that is still enumerating cameras has NOTHING listening — not
                     even the ping port. Killing it here would just restart the
                     slow part, forever. Wait instead.
      bound, silent  the command thread is stuck inside execute_command, almost
                     always camera_loader.start(), whose per-camera t.join() has
                     no timeout (camera_loader.py:92-101). The process stays
                     alive with the port listening. Only a restart clears this.

    A plain TCP connect separates them, so we wait out the first and abort on
    the second.
    """
    import socket as _socket
    import zmq
    ip = get_pc_ip(pc_name)

    def port_open():
        s = _socket.socket()
        s.settimeout(2)
        try:
            s.connect((ip, 5482))
            return True
        except OSError:
            return False
        finally:
            s.close()

    t0 = time.time()
    warned = False
    while not port_open():
        if time.time() - t0 > boot_grace_s:
            raise SystemExit(
                f"\n{pc_name}'s camera daemon has not bound its command port "
                f"(5482) within {boot_grace_s}s.\n"
                "Nothing is listening at all, so it is still inside "
                "CameraLoader() in camera_server_daemon.__init__ — a camera is "
                "not coming up. Check the cameras are powered and enumerated, "
                "then restart the daemon:\n"
                f"  ssh -p 77 {pc_name}@{ip} \"pkill -f -- "
                "'[s]rc/camera/[s]erver_daemon.py'\"\n"
                f"  ssh -p 77 {pc_name}@{ip} \"cd ~/paradex && nohup bash -i -c "
                "'source ~/anaconda3/etc/profile.d/conda.sh && conda activate "
                "flir_env && python src/camera/server_daemon.py &' </dev/null "
                ">/dev/null 2>&1 &\"\n"
            )
        if not warned:
            print(f"  {pc_name} is still initialising cameras (no socket bound "
                  f"yet); waiting up to {boot_grace_s}s ...")
            warned = True
        time.sleep(3)

    ctx = zmq.Context.instance()
    sock = ctx.socket(zmq.REQ)
    sock.setsockopt(zmq.LINGER, 0)
    sock.setsockopt(zmq.RCVTIMEO, timeout_ms)
    sock.setsockopt(zmq.SNDTIMEO, timeout_ms)
    sock.connect(f"tcp://{ip}:5482")
    t1 = time.time()
    try:
        sock.send_json({"action": "heartbeat", "controller_name": "__preflight__"})
        reply = sock.recv_json()
        print(f"  {pc_name} command port OK ({(time.time() - t1) * 1000:.0f} ms) "
              f"-> {reply.get('msg')}")
    except Exception:
        raise SystemExit(
            f"\n{pc_name}'s camera daemon is listening on 5482 but did not answer "
            f"within {timeout_ms} ms.\n"
            "The port is bound, so startup finished — the command thread itself "
            "is stuck, almost certainly in camera_loader.start() "
            "(paradex/io/camera_system/camera_loader.py:92-101 joins the "
            "per-camera start threads with no timeout). The process will look "
            "healthy: alive, port listening, ping answering from its own thread. "
            "Only a restart clears it:\n"
            f"  ssh -p 77 {pc_name}@{ip} \"pkill -f -- "
            "'[s]rc/camera/[s]erver_daemon.py'\"\n"
            f"  ssh -p 77 {pc_name}@{ip} \"cd ~/paradex && nohup bash -i -c "
            "'source ~/anaconda3/etc/profile.d/conda.sh && conda activate "
            "flir_env && python src/camera/server_daemon.py &' </dev/null "
            ">/dev/null 2>&1 &\"\n"
        )
    finally:
        sock.close()


def kill_stale_stream_clients(pc_list):
    """A stream_client left over from an earlier session is never usable, and
    leaving one running breaks this session in two ways at once:

      1. It still owns PUB port 1234, so the stream_client we launch dies on
         bind. run_script sends remote output to /dev/null, so the traceback is
         never seen — the run just reports "no frame within 5s".
      2. It is silent anyway. stream_client only publishes when
         frame_id > last_frame_ids[cam] (src/capture/camera/stream_client.py:58),
         but frame_id is the FLIR frameID, which restarts from 0 every time
         acquisition restarts (paradex/io/camera_system/camera.py:423). Once our
         rcc.start() restarts acquisition, the old process's high-water mark can
         never be beaten, so it publishes nothing for the rest of its life.

    The `[s]` brackets keep the pattern from matching the ssh command line that
    carries it — otherwise pkill kills its own shell and ssh exits 255.
    """
    pattern = "'[s]rc/capture/camera/[s]tream_client.py'"
    print(f"[robot_client] clearing stale stream_clients on {pc_list} ...")
    for pc_name in pc_list:
        ip = get_pc_ip(pc_name)
        cmd = ["ssh", "-p", "77", f"{pc_name}@{ip}", f"pkill -f -- {pattern}"]
        # pkill exits 1 when nothing matched, which is the normal case.
        rc = subprocess.run(cmd, capture_output=True).returncode
        print(f"  {pc_name}: {'killed a leftover' if rc == 0 else 'nothing to clear'}")
    time.sleep(1.0)   # let the PUB socket release port 1234


def bring_up_cameras(fps, decode_serials, resize_divisor=3):
    """Mirrors src/stream/vla/viewer.py:bring_up_cameras.

    Only the serial we actually decode is published: stream_client filters
    --cameras against the local camera list, so it exits cleanly on the capture
    PCs that do not own it instead of flooding port 1234 with unused streams.

    Everything is scoped to the ONE capture PC that owns the camera. Talking to
    PCs we do not need is not free: remote_camera_controller.send_command sends
    to every PC and then recvs them sequentially, so a PC whose daemon is slow
    or wedged eats the whole timeout budget before the others are read. During
    `start` that budget is 15 s, which is longer than the daemon's own 5 s
    command-socket timeout — so an unrelated PC that goes quiet makes the
    HEALTHY ones time out, drop their controller registration and stop their
    cameras (camera_server_daemon.py:167-170). Nothing re-registers afterwards.
    Restricting the pc_list removes that coupling entirely.
    """
    pcs = sorted({pc_owning(s) for s in decode_serials})
    print(f"[robot_client] cameras {sorted(decode_serials)} live on {pcs} "
          f"— all camera traffic scoped to those PCs")
    kill_stale_stream_clients(pcs)
    serial_arg = ",".join(sorted(decode_serials))
    print(f"[robot_client] launching stream_client "
          f"(cameras={serial_arg}, resize-divisor={resize_divisor}) ...")
    run_script("python src/capture/camera/stream_client.py "
               f"--cameras {serial_arg} --resize-divisor {resize_divisor}",
               pc_list=pcs)
    print("[robot_client] checking daemon command ports ...")
    for pc_name in pcs:
        assert_daemon_responsive(pc_name)
    print("[robot_client] opening remote_camera_controller as OWNER ...")
    rcc = remote_camera_controller("robot_client", pc_list=pcs)
    print("[robot_client] starting DataCollector (port 1234) ...")
    dc = DataCollector(pc_list=pcs)
    dc.start()
    print(f"[robot_client] commanding daemons -> stream mode (fps={fps}) ...")
    rcc.start("stream", False, fps=fps)
    cam_source = CameraFrameSource(dc, decode_serials=decode_serials)
    return rcc, dc, cam_source


def teardown_cameras(rcc, dc, cam_source):
    if cam_source is not None:
        cam_source.stop()
    for fn in (rcc.stop, rcc.end, dc.end):
        try: fn()
        except Exception as e: print(f"[robot_client] teardown err: {e}")


# ------------------------------------------------------------------
# Main loop
# ------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--server", required=True)
    p.add_argument("--task_text", required=True)
    p.add_argument("--urdf",
                   default="vla_test/robot_server/xarm_inspire/xarm_inspire.urdf",
                   help="URDF used for FK. Defaults to the bundled VLA URDF.")
    p.add_argument("--log_dir", required=True)
    p.add_argument("--control_hz", type=float, default=30.0,
                   help="cartesian execute rate (step pacing)")
    p.add_argument("--camera-serial", type=str, default="26053260",
                   help="serial sent as `video.front` (26053260 lives on capturenew)")
    p.add_argument("--camera-fps", type=int, default=30)
    p.add_argument("--resize-divisor", type=int, default=3,
                   help="capture-PC JPEG downsample before publish; 3 gives "
                        "682x512, which prepare_frame_for_vla then resizes to "
                        "640x480. The old default of 8 published 256x192 and "
                        "the upscale to 640x480 threw away real detail.")
    p.add_argument("--no-safety", action="store_true",
                   help="skip the per-step delta safety check (DANGEROUS)")
    p.add_argument("--action-frame", choices=("link6", "tcp"), default="link6",
                   help="frame the policy's action chunk is expressed in. "
                        "'link6' (default, verified on this policy) means the "
                        "targets are link6 poses like state.eef_*, so they get "
                        "converted to the SDK TCP frame before being sent. "
                        "'tcp' sends them unchanged.")
    p.add_argument("--action-mode", choices=("delta", "absolute", "auto"),
                   default="delta",
                   help="how to read the action chunk. Pinned to 'delta': a live "
                        "query to this server returns |chunk[0,0:3]| = 0.0225, "
                        "i.e. normalized deltas (POS_SCALE=0.05 -> ~1 mm/step), "
                        "not workspace coordinates. 'auto' re-guesses per query "
                        "from a 0.20 magnitude threshold and would flip to "
                        "absolute on a single fast chunk — do not use it.")
    p.add_argument("--exec-steps", type=int, default=ACTION_CHUNK_LENGTH,
                   metavar="N",
                   help=f"execute only the first N of the {ACTION_CHUNK_LENGTH} "
                        "chunk steps, then re-query (action chunking). Smaller N "
                        "= fresher observations per metre travelled, but the "
                        "policy RTT (~1.3 s) is paid every cycle either way, so "
                        "N below ~8 mostly buys idle time, not responsiveness.")
    p.add_argument("--ensemble-alpha", type=float, default=None, metavar="A",
                   help="temporal ensembling: average each timestep's prediction "
                        "from the two chunks that cover it, weighting the newer "
                        "one by A (0.5 = equal). Requires --exec-steps < 16 for "
                        "the chunks to overlap at all. Off by default.")
    p.add_argument("--save-frames", action="store_true", default=True,
                   help="write the exact image sent to the policy for every "
                        "query to <log_dir>/frames/. On by default — without it "
                        "there is no way to see what the policy saw when it "
                        "decided to close the hand.")
    p.add_argument("--no-save-frames", dest="save_frames", action="store_false")
    p.add_argument("--pipeline", action="store_true",
                   help="overlap the next policy query with the current chunk's "
                        "motion. Measured on this rig the query is 1.29 s "
                        "(server 1.11 s) against 0.24 s of motion, so serial "
                        "operation leaves the arm still 85%% of the time. "
                        "Cost: the observation the policy sees is one chunk "
                        "stale; deltas are integrated from the pose the running "
                        "chunk ends at, not from the pose that was observed.")
    p.add_argument("--pace-to-query", action="store_true",
                   help="with --pipeline, stretch each chunk to last exactly as "
                        "long as the previous query took, so the arm never "
                        "idles. Overrides --control_hz per cycle.")
    p.add_argument("--loop", action="store_true",
                   help="start looping query→execute immediately instead of "
                        "waiting at the prompt")
    p.add_argument("--max-iters", type=int, default=0, metavar="M",
                   help="stop the loop after M iterations (0 = unlimited)")
    args = p.parse_args()

    if not 1 <= args.exec_steps <= ACTION_CHUNK_LENGTH:
        raise SystemExit(f"--exec-steps must be in 1..{ACTION_CHUNK_LENGTH}, "
                         f"got {args.exec_steps}")

    log_dir = Path(args.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    print(f"Logging to {log_dir}")

    # ── ZMQ + URDF ───────────────────────────────────────────────────────────
    policy = PolicyClient(args.server)
    print(f"Connected to policy server at {args.server}")
    urdf_path = Path(args.urdf).expanduser()
    if not urdf_path.is_absolute():
        urdf_path = REPO_ROOT / urdf_path
    if not urdf_path.is_file():
        raise SystemExit(
            f"URDF not found: {urdf_path}\n"
            f"Use --urdf {REPO_ROOT / 'rsc/robot/xarm_inspire.urdf'} "
            "or omit --urdf to use the bundled default."
        )
    urdf = yourdfpy.URDF.load(str(urdf_path))

    # ── Robot SDKs ───────────────────────────────────────────────────────────
    print("[robot_client] connecting xArm + Inspire ...")
    arm = get_arm("xarm")
    try:
        link6_to_sdk_tcp = resolve_command_transform(arm, urdf, args.action_frame)
        tcp_norm_m = float(np.linalg.norm(measure_link6_to_tcp(arm, urdf)[:3, 3]))
    except BaseException:
        # XArmController runs a non-daemon control thread; without end() the
        # process would hang instead of reporting the TCP failure.
        try: arm.end(set_break=False)
        except Exception as e: print(f"[robot_client] arm.end err: {e}")
        raise
    hand = get_hand("inspire", ip=True)
    robot_handles = {"xarm": arm, "hand": hand}

    # ── Camera stream ────────────────────────────────────────────────────────
    rcc, dc, cam_source = bring_up_cameras(
        args.camera_fps, decode_serials={args.camera_serial},
        resize_divisor=args.resize_divisor)

    log_records = []
    st = {"targets": None, "frame_ok": False, "queries": 0, "execs": 0,
          "t_query": 0.0, "t_move": 0.0, "last_rtt_s": 0.0,
          "t_wall_loop": 0.0, "prev_tail": []}
    frames_dir = log_dir / "frames"
    if args.save_frames:
        frames_dir.mkdir(exist_ok=True)
        print(f"[robot_client] saving policy input frames to {frames_dir}")
    stop_flag = threading.Event()

    def fetch_chunk():
        """Observe + ask the policy. NO integration — this runs on the worker
        thread under --pipeline, while the arm is still moving, so there is no
        pose here worth integrating from. Returns the raw reply or None."""
        t0 = time.time()
        try:
            obs_for_policy, raw_state = read_state(robot_handles, urdf)
            frame = read_camera(cam_source, args.camera_serial)
            chunk_arr, rtt_ms, server_ms = query_policy(
                policy, obs_for_policy, frame, args.task_text)
        except Exception as e:
            print(f"  query failed: {e}")
            return None
        st["last_rtt_s"] = rtt_ms / 1000.0
        st["t_query"] += time.time() - t0
        return {"chunk": chunk_arr, "rtt_ms": rtt_ms, "server_ms": server_ms,
                "obs": obs_for_policy, "raw": raw_state, "frame": frame}

    def build_plan(fetched, base_pos, base_rotvec):
        """Integrate a fetched chunk from a base pose and run the checks.

        Called on the MAIN thread immediately before execution, with the base
        pose freshly measured — so the plan is anchored to where the arm
        actually is, never to a prediction of where it should have got to.
        """
        if fetched is None:
            return False
        chunk_arr = fetched["chunk"]
        obs_for_policy, raw_state = fetched["obs"], fetched["raw"]
        rtt_ms, server_ms = fetched["rtt_ms"], fetched["server_ms"]
        try:
            targets = integrate_chunk(
                chunk_arr,
                cur_pos=base_pos,
                cur_rotvec=base_rotvec,
                action_mode=args.action_mode,
                hand_rad_to_raw=vla_rad_to_hand_raw_register,
            )
        except Exception as e:
            print(f"  chunk integration failed: {e}")
            return False

        raw_targets = targets
        blended = ""
        if args.ensemble_alpha is not None and st["prev_tail"]:
            n_ov = min(len(st["prev_tail"]), len(targets))
            targets = blend_plans(targets, st["prev_tail"], args.ensemble_alpha)
            blended = f"  [ensembled {n_ov} steps @ alpha={args.ensemble_alpha}]"
        # Carry the part of THIS chunk we will not execute; it predicts the same
        # instants as the head of the next one. Store the raw prediction, not the
        # blended one, so averaging never compounds across many chunks.
        st["prev_tail"] = raw_targets[args.exec_steps:]

        st["targets"] = targets
        st["queries"] += 1
        print(f"  [query #{st['queries']}] RTT={rtt_ms:.0f}ms "
              f"server={server_ms:.0f}ms{blended}")
        frame_name = None
        if args.save_frames and fetched.get("frame") is not None:
            import cv2
            frame_name = f"q{st['queries']:04d}.jpg"
            cv2.imwrite(str(frames_dir / frame_name),
                        cv2.cvtColor(fetched["frame"], cv2.COLOR_RGB2BGR),
                        [int(cv2.IMWRITE_JPEG_QUALITY), 85])
        obs_age_mm = np.linalg.norm(
            np.asarray(base_pos) - obs_for_policy["state.eef_position"]) * 1000
        if obs_age_mm > 0.5:
            print(f"  observation is {obs_age_mm:.1f} mm behind the integration "
                  f"base (pipelined: policy saw the pose one chunk ago)")
        print_plan(targets, np.asarray(base_pos), np.asarray(base_rotvec))
        ok, why = check_plan_safe(targets, start_pos=base_pos,
                                  start_rotvec=base_rotvec)
        print(f"  safety: {'OK' if ok else 'REJECT'} — {why}")
        frame_ok, frame_why = check_action_frame(
            targets, base_pos, args.action_frame, tcp_norm_m, args.action_mode)
        print(f"  action frame ({args.action_frame}): "
              f"{'OK' if frame_ok else 'MISMATCH'} — {frame_why}")
        scale_ok, scale_why = check_action_scale(chunk_arr, args.action_mode)
        print(f"  action scale ({args.action_mode}): "
              f"{'OK' if scale_ok else 'REJECT'} — {scale_why}")
        frame_ok = frame_ok and scale_ok

        log_records.append({
            "t_wall":    time.time(),
            "event":     "query",
            "idx":       st["queries"],
            "rtt_ms":    rtt_ms,
            "server_ms": server_ms,
            "state":     {k: v.tolist() for k, v in obs_for_policy.items()},
            "raw":       raw_state,
            "plan":      [{k: (v.tolist() if hasattr(v, "tolist") else v)
                           for k, v in t.items()} for t in targets],
            "safety_ok": bool(ok),
            "safety_why": why,
            "action_frame": args.action_frame,
            "action_frame_ok": bool(frame_ok),
            "action_frame_why": frame_why,
            "action_mode": args.action_mode,
            "action_scale_ok": bool(scale_ok),
            "action_scale_why": scale_why,
            "exec_steps": args.exec_steps,
            "frame_file": frame_name,
            "ensemble_alpha": args.ensemble_alpha,
            "link6_to_sdk_tcp": (None if link6_to_sdk_tcp is None
                                 else np.asarray(link6_to_sdk_tcp).tolist()),
        })
        st["frame_ok"] = frame_ok
        return True

    def do_query(integrate_from=None):
        """Serial path / REPL 'q': fetch and integrate from the live pose."""
        fetched = fetch_chunk()
        if fetched is None:
            return False
        base = integrate_from or (fetched["obs"]["state.eef_position"],
                                  fetched["obs"]["state.eef_rotation"])
        return build_plan(fetched, base[0], base[1])

    def do_execute(control_hz=None, plan_targets=None, plan_frame_ok=None,
                   start_obs=None):
        """Execute the first --exec-steps targets.

        Returns (ok, end_pose) where end_pose is the (pos, rotvec) the plan was
        commanded to finish at — the integration base the NEXT chunk must use
        when --pipeline overlaps the query with this motion.

        plan_targets/plan_frame_ok must be passed under --pipeline: the
        overlapped query rewrites st["targets"] from its worker thread while
        this runs, so reading the shared slot here would race and could execute
        the NEXT chunk instead of this one."""
        targets = plan_targets if plan_targets is not None else st["targets"]
        frame_ok_now = (st["frame_ok"] if plan_frame_ok is None else plan_frame_ok)
        if targets is None:
            print("  no plan yet — run 'q' first")
            return False, None
        # Action chunking: keep only the head of the chunk, then re-query. The
        # tail is discarded, not queued, so the next plan starts from a fresh
        # observation instead of extrapolating.
        plan = targets[:args.exec_steps]

        if not args.no_safety and not frame_ok_now:
            print("  REFUSED: the plan contradicts --action-frame "
                  f"{args.action_frame}. Sending it would offset every command "
                  "by the link6<->TCP difference and walk the arm into a joint "
                  "limit.")
            return False, None
        # Re-read the pose NOW, not at query time: execute_plan and
        # check_plan_safe both need the actual pose the motion starts from.
        # Without it execute_plan interpolates from plan[0], so the
        # current-pose -> plan[0] leg goes out as one unbounded servo command,
        # and check_plan_safe skips its start->step0 jump test entirely.
        if start_obs is not None:
            obs_now = start_obs          # already measured for the integration base
        else:
            try:
                obs_now, _ = read_state(robot_handles, urdf)
            except Exception as e:
                print(f"  REFUSED: could not read the current pose — {e}")
                return False, None
        start_pos = obs_now["state.eef_position"]
        start_rotvec = obs_now["state.eef_rotation"]
        start_hand_rad = obs_now["state.dexhand_position"]
        jump_mm = np.linalg.norm(plan[0]["target_pos_m"] - start_pos) * 1000
        print(f"  start→step0 jump = {jump_mm:.1f} mm "
              f"(interpolated in ≤20 mm steps)")

        if not args.no_safety:
            ok, why = check_plan_safe(plan, start_pos=start_pos,
                                      start_rotvec=start_rotvec)
            if not ok:
                print(f"  REFUSED: safety check failed — {why}")
                print(f"  (re-run with --no-safety to override)")
                return False, None

        st["execs"] += 1
        exec_idx = st["execs"]
        hz = control_hz if control_hz is not None else args.control_hz
        print(f"  [execute #{exec_idx}] sending {len(plan)} of "
              f"{len(targets)} chunk steps at {hz:.1f}Hz "
              f"({len(plan) / hz:.2f}s of motion)")
        t_start = time.time()

        step_records = []
        def _on_step(idx, target):
            step_records.append({
                "t_wall":  time.time(),
                "event":   "execute_step",
                "exec_idx": exec_idx,
                "step":    idx,
                "target":  {k: (v.tolist() if hasattr(v, "tolist") else v)
                            for k, v in target.items()},
            })

        sent, err = 0, None
        try:
            sent = execute_plan(
                arm, hand, plan,
                control_hz=hz,
                on_step=_on_step,
                stop_check=stop_flag.is_set,
                link6_to_sdk_tcp=link6_to_sdk_tcp,
                start_pos=start_pos,
                start_rotvec=start_rotvec,
                start_hand_rad=start_hand_rad,
                hand_rad_to_raw=vla_rad_to_hand_raw_register,
            )
            dt = time.time() - t_start
            st["t_move"] += dt
            print(f"  done — {sent} steps in {dt:.2f}s"
                  + ("  (INTERRUPTED)" if stop_flag.is_set() else ""))
        except Exception as e:
            err = e
            print(f"  execute failed: {e}")
        log_records.extend(step_records)
        end_pose = (np.asarray(plan[-1]["target_pos_m"], dtype=np.float64),
                    np.asarray(plan[-1]["target_rotvec_rad"], dtype=np.float64))
        return err is None, end_pose

    def run_loop(max_iters):
        """query→execute until Ctrl-C, a failed check, or --max-iters.

        SIGINT is trapped rather than raised so an interrupt lands between two
        servo steps (execute_plan polls stop_check before each one) instead of
        unwinding mid-motion; the arm is then snapped to its measured pose so
        it does not coast to the last streamed target.
        """
        stop_flag.clear()
        loop_t0 = time.time()
        prev_handler = signal.signal(signal.SIGINT, lambda *_: stop_flag.set())
        print(f"\n[loop] running — {args.exec_steps}/{ACTION_CHUNK_LENGTH} chunk "
              f"steps per cycle, Ctrl-C to stop"
              + (f", max {max_iters} iters" if max_iters else ""))
        i = 0
        pool = ThreadPoolExecutor(max_workers=1) if args.pipeline else None
        pending = None
        try:
            while not stop_flag.is_set():
                if max_iters and i >= max_iters:
                    print(f"[loop] reached --max-iters {max_iters}")
                    break
                i += 1
                t0 = time.time()
                print(f"\n[loop] iter {i}")

                if not args.pipeline:
                    if not do_query():
                        print("[loop] stopping — query failed")
                        break
                    plan_now, frame_ok_now, start_obs = st["targets"], st["frame_ok"], None
                else:
                    fetched = fetch_chunk() if pending is None else pending.result()
                    if fetched is None:
                        print("[loop] stopping — query failed")
                        break
                    # Measure NOW, integrate NOW: the chunk is anchored to the
                    # pose the arm is actually in as execution begins, so no
                    # predicted pose enters the plan. The observation inside
                    # `fetched` is still one chunk old — that staleness is
                    # inherent to overlapping, the integration error is not.
                    try:
                        start_obs, _ = read_state(robot_handles, urdf)
                    except Exception as e:
                        print(f"[loop] stopping — could not read the pose: {e}")
                        break
                    if not build_plan(fetched,
                                      start_obs["state.eef_position"],
                                      start_obs["state.eef_rotation"]):
                        print("[loop] stopping — chunk integration failed")
                        break
                    plan_now, frame_ok_now = st["targets"], st["frame_ok"]

                if stop_flag.is_set():
                    break

                # Pace the chunk to cover the query it overlaps, so the arm is
                # still moving when the next plan lands.
                hz = None
                if args.pipeline and args.pace_to_query and st["last_rtt_s"] > 0:
                    hz = max(2.0, args.exec_steps / st["last_rtt_s"])

                if args.pipeline:
                    # Submit the next fetch BEFORE blocking on this motion. The
                    # policy is a REQ socket, so exactly one worker may touch it
                    # and the main thread must not query while this is in flight.
                    pending = pool.submit(fetch_chunk)

                ok, _ = do_execute(control_hz=hz, plan_targets=plan_now,
                                   plan_frame_ok=frame_ok_now,
                                   start_obs=start_obs)
                if not ok:
                    print("[loop] stopping — execute refused or failed")
                    break
                print(f"[loop] iter {i} cycle {time.time() - t0:.2f}s")
        finally:
            st["t_wall_loop"] += time.time() - loop_t0
            if pending is not None:
                try: pending.result(timeout=20)   # let the in-flight fetch finish
                except Exception: pass
            if pool is not None:
                pool.shutdown(wait=True)
            signal.signal(signal.SIGINT, prev_handler)
            if stop_flag.is_set():
                try:
                    arm.hold()
                    print("[loop] stopped by Ctrl-C — arm held at its measured pose")
                except Exception as e:
                    print(f"[loop] arm.hold failed: {e}")
            stop_flag.clear()

    print("\nReady. Commands:")
    print("  q + Enter  → query policy with current obs")
    print(f"  e + Enter  → execute the first {args.exec_steps} of "
          f"{ACTION_CHUNK_LENGTH} chunk steps")
    print("  l + Enter  → loop query→execute until Ctrl-C")
    print("  x + Enter  → exit")

    try:
        if args.loop:
            run_loop(args.max_iters)

        while True:
            try:
                cmd = input("\n> ").strip().lower()
            except (EOFError, KeyboardInterrupt):
                print("\nexit")
                break

            if cmd == "x":
                break
            elif cmd == "q":
                do_query()
            elif cmd == "e":
                do_execute()
            elif cmd == "l":
                run_loop(args.max_iters)
            else:
                print("  unknown command. Use q / e / l / x")

    finally:
        # Flush log first so it survives any teardown error.
        out_jsonl = log_dir / "session.jsonl"
        with open(out_jsonl, "w") as f:
            for rec in log_records:
                f.write(json.dumps(rec) + "\n")
        print(f"\nWrote {len(log_records)} records to {out_jsonl}")
        print(f"Queries: {st['queries']}, Executions: {st['execs']} "
              f"({args.exec_steps}/{ACTION_CHUNK_LENGTH} steps each)")
        if st["execs"]:
            wall = max(st["t_wall_loop"], 1e-9)
            print(f"Time in policy queries: {st['t_query']:.1f}s, "
                  f"in motion: {st['t_move']:.1f}s, loop wall {wall:.1f}s "
                  f"({100 * st['t_move'] / wall:.0f}% of wall-clock moving)")

        teardown_cameras(rcc, dc, cam_source)
        try: arm.end(set_break=False)
        except Exception as e: print(f"[robot_client] arm.end err: {e}")
        try: hand.end()
        except Exception as e: print(f"[robot_client] hand.end err: {e}")
        try: policy.close()
        except Exception: pass


if __name__ == "__main__":
    main()
