"""Live VLA viewer: camera streams + robot realtime state + VLA query + execute.

Brings up the paradex camera stream (run_script → stream_client.py on every
capture PC, owns the daemon, subscribes via DataCollector), connects to the
arm + hand, renders the live URDF in viser, and shows the selected camera
serial in a SEPARATE cv2 window (default = 25322639, same serial sent to
the VLA as `video.front`).

GUI folder 'VLA' adds (when --vla-server is set):
  - Task text + camera serial (editable per-query)
  - 'Query Policy' button   → sends one obs to the policy, renders the
                              16-step plan as polyline + per-step frames +
                              a ghost inspire_float hand at the slider step.
  - 'Plan step' slider      → scrub the ghost hand through the plan.
  - 'Execute Plan' button   → cartesian execute the most-recent plan on the
                              real robot (arm + hand). Requires the
                              'Arm motion' confirm checkbox to be on.
                              Aborts if check_plan_safe() rejects the plan.

  python src/stream/vla/viewer.py --arm xarm --hand inspire
  python src/stream/vla/viewer.py --arm xarm --hand inspire \\
      --vla-server tcp://localhost:5600 \\
      --vla-task "throw the pepsi can into the white basket"

Ctrl-C (or 'q' inside the image window) exits.
"""
import argparse
import os
import time
import traceback
from threading import Event, Lock

import cv2
import numpy as np
from scipy.spatial.transform import Rotation as R

from paradex.io.camera_system.remote_camera_controller import remote_camera_controller
from paradex.io.capture_pc.ssh import run_script
from paradex.io.capture_pc.data_sender import DataCollector
from paradex.io.robot_controller import get_arm, get_hand
from paradex.robot.utils import get_robot_urdf_path
from paradex.robot.inspire import limit as INSPIRE_LIMIT
from paradex.visualization.robot import RobotModule
from paradex.visualization.visualizer.viser import ViserViewer
from paradex.vla.client import (
    ARM_TO_HAND_OFFSET,
    CameraFrameSource,
    check_plan_safe,
    execute_plan,
    fk_link6,
    integrate_chunk,
    make_policy_client,
    prepare_frame_for_vla,
    read_arm_qpos_rad,
    read_hand_qpos_rad,
)


# ── LIVE-robot hand qpos remapping (URDF actuated-joint order for ViserViewer) ──
# These mappings are for displaying the LIVE robot's current state. The VLA
# protocol mappings (qpos order [thumb_yaw..pinky]) live in paradex.vla.client.
ALLEGRO_CMD_TO_URDF = np.array(
    [12, 13, 14, 15,  # thumb_base, _proximal, _medial, _distal
     0, 1, 2, 3,      # index_base, _proximal, _medial, _distal
     4, 5, 6, 7,      # middle_base, _proximal, _medial, _distal
     8, 9, 10, 11],   # ring_base, _proximal, _medial, _distal
)
_INSPIRE_RAW_NAMES = ['right_little_1_joint', 'right_ring_1_joint',
                      'right_middle_1_joint', 'right_index_1_joint',
                      'right_thumb_2_joint', 'right_thumb_1_joint']
_INSPIRE_URDF_ORDER = ['right_thumb_1_joint', 'right_thumb_2_joint',
                       'right_index_1_joint', 'right_middle_1_joint',
                       'right_ring_1_joint', 'right_little_1_joint']
INSPIRE_RAW_TO_URDF = np.array(
    [_INSPIRE_RAW_NAMES.index(n) for n in _INSPIRE_URDF_ORDER]
)
INSPIRE_LIMIT_VEC = np.array([INSPIRE_LIMIT[n] for n in _INSPIRE_URDF_ORDER])


def live_hand_qpos_for_urdf(hand_name, hand):
    if hand_name == "allegro":
        return hand.get_data()['qpos'][ALLEGRO_CMD_TO_URDF]
    if hand_name == "inspire":
        if hasattr(hand, "get_qpos"):
            raw = np.asarray(hand.get_qpos(), dtype=float)
        else:
            raw = np.asarray(hand.get_data()['joint_value'], dtype=float)
        ordered = raw[INSPIRE_RAW_TO_URDF]
        return INSPIRE_LIMIT_VEC * (1.0 - ordered / 1000.0)
    raise ValueError(f"Unknown hand: {hand_name}")


def make_ghost_cfg(eef_pos, eef_rotvec, hand_rad_qpos):
    """(12,) cfg for inspire_float at a predicted link6 pose.
    URDF actuated order: [x, y, z, rx, ry, rz, thumb_1, thumb_2, index_1,
    middle_1, ring_1, little_1]. Floating base is intrinsic XYZ Euler."""
    link6_T = np.eye(4)
    link6_T[:3, :3] = R.from_rotvec(eef_rotvec).as_matrix()
    link6_T[:3, 3] = eef_pos
    wrist_T = link6_T @ ARM_TO_HAND_OFFSET
    xyz = wrist_T[:3, 3]
    rxyz = R.from_matrix(wrist_T[:3, :3]).as_euler('XYZ')
    # VLA qpos order = _INSPIRE_URDF_ORDER (verified, identity mapping).
    hand_urdf = np.asarray(hand_rad_qpos, dtype=float)
    return np.concatenate([xyz, rxyz, hand_urdf])


def pose_vec_to_homo(pos_m, rotvec_rad):
    T = np.eye(4)
    T[:3, :3] = R.from_rotvec(rotvec_rad).as_matrix()
    T[:3, 3] = pos_m
    return T


def update_frame_handle(handle, T):
    handle.position = T[:3, 3]
    handle.wxyz = R.from_matrix(T[:3, :3]).as_quat()[[3, 0, 1, 2]]


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--arm", type=str, default=None, choices=[None, "xarm"])
    p.add_argument("--hand", type=str, default=None,
                   choices=[None, "allegro", "inspire"])
    p.add_argument("--hand-no-ip", action="store_true",
                   help="(inspire only) use USB controller instead of TCP/IP")
    p.add_argument("--no-camera", action="store_true",
                   help="skip camera stream — robot state only")
    p.add_argument("--fps", type=int, default=30, help="camera stream fps")
    p.add_argument("--rate", type=float, default=30.0,
                   help="viewer update rate (Hz) — also imshow rate")
    p.add_argument("--port", type=int, default=8080, help="viser port")
    # ── Image viewer (separate cv2 window) ──────────────────────────────────
    p.add_argument("--view-camera", type=str, default="25322639",
                   help="camera serial shown in the cv2 window. "
                        "Use 'all' to merge every stream.")
    p.add_argument("--view-scale", type=float, default=2.0,
                   help="upscale factor for the cv2 viewer")
    p.add_argument("--no-view", action="store_true")
    p.add_argument("--record-view", type=str, default=None,
                   help="save the cv2 camera viewer frames to this mp4 path.")
    p.add_argument("--record-view-fps", type=float, default=None,
                   help="fps for --record-view. Defaults to --rate.")
    # ── VLA query + execute ─────────────────────────────────────────────────
    p.add_argument("--vla-server", type=str, default=None,
                   help="ZMQ policy server (e.g. tcp://localhost:5600). "
                        "Omit to disable the VLA folder.")
    p.add_argument("--vla-protocol", type=str, default="legacy",
                   choices=["legacy", "gr00t"],
                   help="policy wire protocol: legacy flat dict or NVIDIA "
                        "GR00T PolicyServer endpoint RPC.")
    p.add_argument("--gr00t-arm-basis", type=str, default="absolute_eef",
                   choices=["absolute_eef", "delta_eef_scaled", "delta_eef_m",
                            "joint_delta", "joint_absolute"],
                   help="how to interpret GR00T action['arm']: absolute EEF, "
                        "normalized EEF delta using POS_SCALE/ROT_SCALE, or "
                        "raw meter/radian EEF delta, joint delta, or absolute joints.")
    p.add_argument("--gr00t-hand-basis", type=str, default="absolute_raw",
                   choices=["absolute_raw", "delta_raw",
                            "absolute_qpos", "delta_qpos"],
                   help="how to interpret GR00T action['hand']. raw means "
                        "Inspire register order [little..thumb_yaw].")
    p.add_argument("--vla-task", type=str,
                   default="throw the pepsi can into the white basket")
    p.add_argument("--vla-camera", type=str, default=None,
                   help="serial sent as video.front. Defaults to --view-camera.")
    p.add_argument("--vla-cameras", type=str, default=None,
                   help="comma-separated camera serials sent to policy as "
                        "cam0,cam1,...; defaults to --vla-camera.")
    p.add_argument("--exec-control-hz", type=float, default=10.0,
                   help="cartesian execute rate (Hz). Step pacing.")
    p.add_argument("--exec-max-step-m", type=float, default=0.001,
                   help="max Cartesian translation per sent servo substep.")
    p.add_argument("--exec-max-step-rot", type=float, default=0.01,
                   help="max rotation radians per sent servo substep.")
    p.add_argument("--safe-max-step-m", type=float, default=0.20,
                   help="reject policy plans with per-target translation jumps above this.")
    p.add_argument("--safe-max-step-rot", type=float, default=1,
                   help="reject policy plans with per-target rotation jumps above this.")
    p.add_argument("--safe-max-start-m", type=float, default=0.30,
                   help="reject policy plans whose first target is farther than this.")
    p.add_argument("--safe-max-start-rot", type=float, default=1.5,
                   help="reject policy plans whose first target rotates farther than this.")
    p.add_argument("--safe-workspace-min", type=str, default="0.20,-0.55,0.05",
                   help="comma-separated xyz minimum allowed target position in meters.")
    p.add_argument("--safe-workspace-max", type=str, default="0.85,0.35,0.85",
                   help="comma-separated xyz maximum allowed target position in meters.")
    p.add_argument("--loop-delay", type=float, default=0.0,
                   help="seconds to wait between automatic Query/Execute loop cycles.")
    p.add_argument("--vla-action-mode", type=str, default="auto",
                   choices=["auto", "absolute", "delta"],
                   help="interpret policy eef outputs as auto-detected, "
                        "absolute targets, or normalized deltas. In auto, "
                        "the selected policy client chooses the default.")
    p.add_argument("--stop-camera-on-exit", action="store_true",
                   help="stop remote camera daemons on quit. By default the "
                        "viewer only releases ownership so reconnecting keeps "
                        "the stream warm.")
    return p.parse_args()


def bring_up_cameras(fps):
    print("[vla] launching stream_client on capture PCs ...")
    run_script("python src/capture/camera/stream_client.py")
    print("[vla] opening remote_camera_controller as OWNER ...")
    rcc = remote_camera_controller("vla_viewer")
    print("[vla] starting DataCollector (port 1234) ...")
    dc = DataCollector()
    dc.start()
    print(f"[vla] commanding daemons -> stream mode (fps={fps}) ...")
    rcc.start("stream", False, fps=fps)
    time.sleep(0.5)
    if rcc.is_error():
        print("[vla] stream start reported camera errors; restarting cameras once ...")
        rcc.stop()
        time.sleep(0.5)
        rcc.start("stream", False, fps=fps)
    return rcc, dc


def parse_camera_serials(serials_text):
    return [s.strip() for s in serials_text.split(",") if s.strip()]


def parse_vec3(text):
    vals = [float(v.strip()) for v in text.split(",") if v.strip()]
    if len(vals) != 3:
        raise ValueError(f"expected 3 comma-separated values, got: {text}")
    return vals


def teardown_cameras(rcc, dc):
    for fn in (rcc.stop, rcc.end, dc.end):
        try: fn()
        except Exception as e: print(f"[vla] teardown err (ignored): {e}")


def release_camera_owner(rcc, dc):
    for fn in (rcc.release, rcc.end, dc.end):
        try: fn()
        except Exception as e: print(f"[vla] camera release err (ignored): {e}")


def setup_vla_gui(vis, vla_client, fk_rm, arm, hand, cam_source,
                  default_task, default_camera_serial,
                  exec_control_hz, action_mode, loop_delay,
                  policy_camera_serials, exec_max_step_m,
                  exec_max_step_rot, safety_limits):
    """Adds the VLA Query/Execute UI and returns a `state` dict."""

    state = {
        "plan_targets": None,
        "plan_start_pos": None,
        "plan_start_rotvec": None,
        "plan_start_hand_rad": None,
        "link6_to_sdk_tcp": None,
        "frame_handles": [],
        "spline_handle": None,
        "executing": False,
        "abort_execute": False,
        "loop_running": False,
    }
    if action_mode == "auto":
        action_mode = getattr(vla_client, "default_action_mode", "auto")

    # ── Ghost hand robot ────────────────────────────────────────────────────
    inspire_float_urdf = get_robot_urdf_path(arm_name=None, hand_name="inspire")
    print(f"[vla] adding ghost inspire_float for plan preview: {inspire_float_urdf}")
    vis.add_robot("vla_ghost", inspire_float_urdf)
    ghost = vis.robot_dict["vla_ghost"]
    ghost.set_visibility(False)

    # ── GUI ─────────────────────────────────────────────────────────────────
    with vis.server.gui.add_folder("VLA"):
        srv_text  = vis.server.gui.add_text("Server", initial_value=vla_client.server_url)
        srv_text.disabled = True
        task_text = vis.server.gui.add_text("Task", initial_value=default_task)
        cam_text  = vis.server.gui.add_text("Front cam serial",
                                            initial_value=default_camera_serial)
        cams_text = vis.server.gui.add_text(
            "Policy cam serials",
            initial_value=",".join(policy_camera_serials),
        )
        query_btn = vis.server.gui.add_button("Query Policy")
        status    = vis.server.gui.add_markdown("_no plan yet_")
        step_slider = vis.server.gui.add_slider(
            "Plan step", min=0, max=15, step=1, initial_value=0)
        step_slider.disabled = True

        vis.server.gui.add_markdown("---")
        confirm_chk = vis.server.gui.add_checkbox("Arm motion (confirm)", True)
        exec_btn    = vis.server.gui.add_button("Execute Plan")
        query_exec_btn = vis.server.gui.add_button("Query + Execute")
        loop_start_btn = vis.server.gui.add_button("Start Loop")
        loop_stop_btn = vis.server.gui.add_button("Stop Loop", disabled=True)
        abort_btn   = vis.server.gui.add_button("Abort", disabled=True)
        exec_status = vis.server.gui.add_markdown("_idle_")

    def _clear_viz():
        for h in state["frame_handles"]:
            try: h.remove()
            except Exception: pass
        state["frame_handles"] = []
        if state["spline_handle"] is not None:
            try: state["spline_handle"].remove()
            except Exception: pass
            state["spline_handle"] = None

    def _render_plan(targets):
        _clear_viz()
        positions = np.stack([t["target_pos_m"] for t in targets])
        state["spline_handle"] = vis.server.scene.add_spline_catmull_rom(
            "/vla_plan/spline",
            positions=positions, color=(1.0, 0.2, 0.2), line_width=3.0,
        )
        for i, t in enumerate(targets):
            wxyz = R.from_rotvec(t["target_rotvec_rad"]).as_quat()[[3, 0, 1, 2]]
            h = vis.server.scene.add_frame(
                f"/vla_plan/step_{i:02d}",
                position=t["target_pos_m"], wxyz=wxyz,
                show_axes=True, axes_length=0.04, axes_radius=0.003,
            )
            state["frame_handles"].append(h)
        ghost.set_visibility(True)
        _update_ghost_to_step(int(step_slider.value))

    def _update_ghost_to_step(idx):
        if state["plan_targets"] is None:
            return
        idx = max(0, min(idx, len(state["plan_targets"]) - 1))
        t = state["plan_targets"][idx]
        cfg = make_ghost_cfg(t["target_pos_m"], t["target_rotvec_rad"],
                             t["target_hand_rad_qpos"])
        try:
            ghost.update_cfg(cfg)
        except Exception as e:
            print(f"[vla] ghost update_cfg failed (cfg={cfg.shape}): {e}")

    @step_slider.on_update
    def _(_): _update_ghost_to_step(int(step_slider.value))

    def _query_policy_plan() -> bool:
        try:
            arm_joints = read_arm_qpos_rad(arm)
            eef_pos, eef_rotvec = fk_link6(fk_rm, arm_joints)
            hand_rad = read_hand_qpos_rad(hand)

            serials = parse_camera_serials(cams_text.value)
            if not serials:
                serials = [cam_text.value.strip()]
            frames = {}
            missing = []
            for i, serial in enumerate(serials):
                bgr = cam_source.latest(serial)
                if bgr is None:
                    missing.append(serial)
                    continue
                frames[f"cam{i}"] = prepare_frame_for_vla(bgr)
            if missing or not frames:
                avail = cam_source.available_serials()
                status.content = (f"❌ no frame for `{missing}`. "
                                  f"available: `{avail}`")
                return False
            frame_payload = frames if len(frames) > 1 else next(iter(frames.values()))

            obs = {
                "state.eef_position":       eef_pos,
                "state.eef_rotation":       eef_rotvec,
                "state.arm_joint_position": arm_joints,
                "state.dexhand_position":   hand_rad,
            }
            chunk_arr, rtt_ms, server_ms = vla_client.query(
                obs, frame_payload, task_text.value)

            # ── frame-sanity & chunk-magnitude DEBUG ──────────────────────
            # (1) xArm SDK's own base→TCP pose vs URDF FK link6: if these
            #     disagree by ~current-eef-distance, our frame assumption
            #     is wrong (the "link6 vs link_base shift" hypothesis).
            try:
                ret, sdk_pose = arm.arm.get_position_aa(is_radian=True)
                # sdk_pose = [x_mm, y_mm, z_mm, rx, ry, rz] axis-angle.
                sdk_pos_m = np.asarray(sdk_pose[:3], dtype=float) / 1000.0
                sdk_rotvec = np.asarray(sdk_pose[3:], dtype=float)
                d_sdk_urdf_mm = np.linalg.norm(sdk_pos_m - eef_pos) * 1000.0
                link6_T = pose_vec_to_homo(eef_pos, eef_rotvec)
                sdk_T = pose_vec_to_homo(sdk_pos_m, sdk_rotvec)
                state["link6_to_sdk_tcp"] = np.linalg.inv(link6_T) @ sdk_T
                print(f"[vla.dbg] URDF FK link6 pos = {np.round(eef_pos, 3).tolist()}")
                print(f"[vla.dbg] xArm SDK get_pos  = {np.round(sdk_pos_m, 3).tolist()}  "
                      f"|Δ|={d_sdk_urdf_mm:.1f}mm  (>5mm ⇒ frame mismatch)")
            except Exception as e:
                state["link6_to_sdk_tcp"] = None
                print(f"[vla.dbg] arm.get_position_aa failed: {e}")
            # (2) Raw policy output magnitude — absolute eef positions are
            #     workspace-scale (~0.5m), normalized deltas are unitless.
            print(f"[vla.dbg] chunk[ 0,:6] = {np.round(chunk_arr[ 0, :6], 3).tolist()}")
            print(f"[vla.dbg] chunk[15,:6] = {np.round(chunk_arr[15, :6], 3).tolist()}")

            targets = integrate_chunk(
                chunk_arr, eef_pos, eef_rotvec, action_mode=action_mode)
            step0 = targets[0]
            dpos0_mm = np.linalg.norm(step0["target_pos_m"] - eef_pos) * 1000.0
            drot0 = (R.from_rotvec(step0["target_rotvec_rad"]) *
                     R.from_rotvec(eef_rotvec).inv()).as_rotvec()
            drot0_deg = np.linalg.norm(drot0) * 180.0 / np.pi
            hand_chunk = np.asarray(chunk_arr[:, 6:12], dtype=float)
            print(f"[vla.dbg] action_mode used = {action_mode}")
            print(f"[vla.dbg] start eef pos   = {np.round(eef_pos, 4).tolist()} "
                  f"rotvec={np.round(eef_rotvec, 4).tolist()}")
            print(f"[vla.dbg] step0 target   = {np.round(step0['target_pos_m'], 4).tolist()} "
                  f"rotvec={np.round(step0['target_rotvec_rad'], 4).tolist()} "
                  f"|start->0|={dpos0_mm:.1f}mm/{drot0_deg:.1f}deg")
            print(f"[vla.dbg] hand chunk min/max = "
                  f"{np.round(hand_chunk.min(axis=0), 3).tolist()} / "
                  f"{np.round(hand_chunk.max(axis=0), 3).tolist()}")
            state["plan_targets"] = targets
            state["plan_start_pos"] = eef_pos.copy()
            state["plan_start_rotvec"] = eef_rotvec.copy()
            state["plan_start_hand_rad"] = hand_rad.copy()
            _render_plan(targets)
            step_slider.disabled = False
            step_slider.value = 0

            ok, why = check_plan_safe(
                targets,
                start_pos=state["plan_start_pos"],
                start_rotvec=state["plan_start_rotvec"],
                **safety_limits,
            )
            safety_line = (f"\n\n🟢 safety: {why}" if ok
                           else f"\n\n🔴 safety: {why}")
            status.content = (
                f"✅ plan ready — RTT={rtt_ms:.0f}ms server={server_ms:.0f}ms\n\n"
                f"action mode = `{action_mode}`\n\n"
                f"step 0 pos = `{targets[0]['target_pos_m'].round(3).tolist()}`\n\n"
                f"step 15 pos = `{targets[15]['target_pos_m'].round(3).tolist()}`"
                f"{safety_line}"
            )
            return True
        except Exception as e:
            traceback.print_exc()
            status.content = (
                f"❌ query failed: `{e}`\n\n"
                "Policy socket was reset; check the server address/log and retry."
            )
            return False

    @query_btn.on_click
    def _(_) -> None:
        _query_policy_plan()

    @abort_btn.on_click
    def _(_) -> None:
        state["abort_execute"] = True
        state["loop_running"] = False
        exec_status.content = "🟡 abort requested ..."

    def _execute_current_plan_blocking(auto_untick_confirm=True) -> bool:
        if state["plan_targets"] is None:
            exec_status.content = "❌ no plan to execute — run Query first"
            return False
        if not confirm_chk.value:
            exec_status.content = ("❌ tick 'Arm motion (confirm)' first — "
                                   "this moves the real robot")
            return False
        if state["executing"]:
            exec_status.content = "❌ already executing"
            return False
        targets = state["plan_targets"]
        ok, why = check_plan_safe(
            targets,
            start_pos=state["plan_start_pos"],
            start_rotvec=state["plan_start_rotvec"],
            **safety_limits,
        )
        if not ok:
            exec_status.content = f"❌ refused: safety check failed — {why}"
            return False

        state["executing"] = True
        state["abort_execute"] = False
        abort_btn.disabled = False
        exec_btn.disabled = True
        query_exec_btn.disabled = True
        loop_start_btn.disabled = True

        def _on_step(idx, target):
            # Move the slider so user sees progress; ghost also updates.
            try:
                step_slider.value = idx
            except Exception:
                pass
            # Debug: target link6 (= what arm.move was sent) vs measured link6
            # (FK of arm.get_data after the SDK has had ~1/control_hz to settle).
            try:
                m_joints = read_arm_qpos_rad(arm)
                m_pos, m_rotvec = fk_link6(fk_rm, m_joints)
                t_pos = target["target_pos_m"]
                t_rotvec = target["target_rotvec_rad"]
                dpos_mm = np.linalg.norm(t_pos - m_pos) * 1000.0
                drot = (R.from_rotvec(t_rotvec) *
                        R.from_rotvec(m_rotvec).inv()).as_rotvec()
                drot_deg = np.linalg.norm(drot) * 180.0 / np.pi
                print(f"[vla.exec step {idx:2d}] "
                      f"tgt_pos={np.round(t_pos, 3).tolist()} "
                      f"meas_pos={np.round(m_pos, 3).tolist()} "
                      f"|dpos|={dpos_mm:5.1f}mm |drot|={drot_deg:5.1f}deg")
            except Exception as e:
                print(f"[vla.exec step {idx:2d}] debug read failed: {e}")

        completed = False
        try:
            exec_status.content = (
                f"🔵 executing {len(targets)} steps at {exec_control_hz} Hz ...")
            sent = execute_plan(
                arm, hand, targets,
                control_hz=exec_control_hz,
                on_step=_on_step,
                stop_check=lambda: state["abort_execute"],
                link6_to_sdk_tcp=state["link6_to_sdk_tcp"],
                start_pos=state["plan_start_pos"],
                start_rotvec=state["plan_start_rotvec"],
                start_hand_rad=state["plan_start_hand_rad"],
                max_pos_step_m=exec_max_step_m,
                max_rot_step_rad=exec_max_step_rot,
            )
            if state["abort_execute"]:
                exec_status.content = f"🟡 aborted after {sent}/{len(targets)} steps"
            else:
                exec_status.content = f"✅ done — {sent} steps sent"
                completed = True
            if auto_untick_confirm:
                confirm_chk.value = False
        except Exception as e:
            traceback.print_exc()
            exec_status.content = f"❌ execute failed: `{e}`"
        finally:
            state["executing"] = False
            state["abort_execute"] = False
            abort_btn.disabled = True
            exec_btn.disabled = False
            query_exec_btn.disabled = False
            loop_start_btn.disabled = state["loop_running"]
        return completed

    def _start_execute_current_plan() -> None:
        import threading

        threading.Thread(
            target=lambda: _execute_current_plan_blocking(auto_untick_confirm=True),
            daemon=True,
        ).start()

    @exec_btn.on_click
    def _(_) -> None:
        _start_execute_current_plan()

    @query_exec_btn.on_click
    def _(_) -> None:
        if state["executing"]:
            exec_status.content = "❌ already executing"
            return
        if not confirm_chk.value:
            exec_status.content = ("❌ tick 'Arm motion (confirm)' first — "
                                   "this moves the real robot")
            return
        exec_status.content = "🔵 querying policy ..."
        if _query_policy_plan():
            _start_execute_current_plan()

    @loop_start_btn.on_click
    def _(_) -> None:
        if state["loop_running"] or state["executing"]:
            exec_status.content = "❌ already running"
            return
        if not confirm_chk.value:
            exec_status.content = ("❌ tick 'Arm motion (confirm)' first — "
                                   "this moves the real robot")
            return

        import threading
        state["loop_running"] = True
        loop_start_btn.disabled = True
        loop_stop_btn.disabled = False
        query_btn.disabled = True
        exec_btn.disabled = True
        query_exec_btn.disabled = True

        def _loop_runner():
            cycle = 0
            try:
                while state["loop_running"] and confirm_chk.value:
                    cycle += 1
                    exec_status.content = f"🔵 loop cycle {cycle}: querying policy ..."
                    if not _query_policy_plan():
                        state["loop_running"] = False
                        break
                    if not state["loop_running"]:
                        break
                    ok = _execute_current_plan_blocking(auto_untick_confirm=False)
                    if not ok:
                        state["loop_running"] = False
                        break
                    if loop_delay > 0:
                        time.sleep(loop_delay)
                if not confirm_chk.value:
                    exec_status.content = "🟡 loop stopped — confirm unchecked"
                elif not state["abort_execute"]:
                    exec_status.content = "🟡 loop stopped"
            except Exception as e:
                traceback.print_exc()
                exec_status.content = f"❌ loop failed: `{e}`"
            finally:
                state["loop_running"] = False
                state["abort_execute"] = False
                loop_start_btn.disabled = False
                loop_stop_btn.disabled = True
                query_btn.disabled = False
                exec_btn.disabled = False
                query_exec_btn.disabled = False
                abort_btn.disabled = True

        threading.Thread(target=_loop_runner, daemon=True).start()

    @loop_stop_btn.on_click
    def _(_) -> None:
        state["loop_running"] = False
        state["abort_execute"] = True
        exec_status.content = "🟡 stopping loop ..."

    return state


def main():
    args = parse_args()

    if args.vla_camera is None:
        args.vla_camera = (args.view_camera
                           if args.view_camera != "all" else "25322639")
    if args.vla_cameras is None:
        args.vla_cameras = args.vla_camera
    policy_camera_serials = parse_camera_serials(args.vla_cameras)
    if args.view_camera != "all" and args.vla_camera != args.view_camera:
        print(f"[vla] WARNING: viewer shows {args.view_camera} but VLA gets "
              f"{args.vla_camera} — these usually should match.")

    arm = get_arm(args.arm) if args.arm else None
    hand = (get_hand(args.hand, ip=not args.hand_no_ip)
            if args.hand else None)

    rcc = dc = cam_source = None
    if not args.no_camera:
        rcc, dc = bring_up_cameras(args.fps)
        # Decode only what we'll display + send to VLA (saves big CPU).
        if args.view_camera == "all":
            decode_serials = None  # everything
        else:
            decode_serials = {args.view_camera, args.vla_camera, *policy_camera_serials}
        cam_source = CameraFrameSource(dc, decode_serials=decode_serials)

    vis = ViserViewer(port_number=args.port)
    vis.add_floor()

    if args.arm or args.hand:
        urdf_path = get_robot_urdf_path(arm_name=args.arm, hand_name=args.hand)
        print(f"[vla] adding robot URDF: {urdf_path}")
        vis.add_robot("robot", urdf_path)

    # Separate FK module — used by the GUI callback in a different thread than
    # ViserRobotModule's own internal yourdfpy URDF.
    fk_rm = None
    if args.arm == "xarm":
        fk_rm = RobotModule(get_robot_urdf_path(arm_name="xarm", hand_name=args.hand))

    vis.start_viewer(use_thread=True)

    # ── Quit button (always available — triggers the same cleanup path as
    # Ctrl-C: stops camera daemon, disconnects arm/hand, closes VLA socket).
    exit_event = Event()
    with vis.server.gui.add_folder("Session"):
        quit_btn   = vis.server.gui.add_button("Quit")
        quit_label = vis.server.gui.add_markdown("_running_")

    @quit_btn.on_click
    def _(_) -> None:
        quit_label.content = "🟡 shutting down ..."
        exit_event.set()

    # ── VLA setup (optional) ────────────────────────────────────────────────
    vla_client = None
    if args.vla_server:
        if args.arm != "xarm" or args.hand != "inspire":
            raise SystemExit(
                "[vla] --vla-server requires --arm xarm --hand inspire "
                "(VLA was trained on this combo).")
        if arm is None or hand is None:
            raise SystemExit(
                "[vla] --vla-server requires both arm and hand connected.")
        if cam_source is None:
            raise SystemExit(
                "[vla] --vla-server requires the camera stream "
                "(remove --no-camera).")
        client_kwargs = {}
        if args.vla_protocol == "gr00t":
            client_kwargs.update(
                arm_basis=args.gr00t_arm_basis,
                hand_basis=args.gr00t_hand_basis,
                fk_model=fk_rm,
            )
        vla_client = make_policy_client(
            args.vla_protocol, args.vla_server, **client_kwargs)
        print(f"[vla] connected to {args.vla_protocol} policy server {args.vla_server}")
        setup_vla_gui(
            vis, vla_client, fk_rm, arm, hand, cam_source,
            default_task=args.vla_task,
            default_camera_serial=args.vla_camera,
            exec_control_hz=args.exec_control_hz,
            action_mode=args.vla_action_mode,
            loop_delay=args.loop_delay,
            policy_camera_serials=policy_camera_serials,
            exec_max_step_m=args.exec_max_step_m,
            exec_max_step_rot=args.exec_max_step_rot,
            safety_limits={
                "max_step_pos_m": args.safe_max_step_m,
                "max_step_rot_rad": args.safe_max_step_rot,
                "max_start_pos_m": args.safe_max_start_m,
                "max_start_rot_rad": args.safe_max_start_rot,
                "workspace_min": parse_vec3(args.safe_workspace_min),
                "workspace_max": parse_vec3(args.safe_workspace_max),
                "hand_min_rad": np.zeros(6),
                "hand_max_rad": INSPIRE_LIMIT_VEC,
            },
        )

    # ── Separate cv2 image-viewer window ────────────────────────────────────
    show_view = (not args.no_view) and (cam_source is not None)
    view_window_name = None
    record_writer = None
    record_fps = args.record_view_fps if args.record_view_fps is not None else args.rate
    if show_view:
        if args.view_camera == "all":
            view_window_name = "VLA cameras (all)"
        else:
            view_window_name = f"VLA camera {args.view_camera}"
        cv2.namedWindow(view_window_name, cv2.WINDOW_NORMAL)
        placeholder = np.full((480, 640, 3), 30, dtype=np.uint8)
        cv2.putText(placeholder, "Waiting for stream...",
                    (40, 240), cv2.FONT_HERSHEY_SIMPLEX, 1.0,
                    (200, 200, 200), 2, cv2.LINE_AA)
        cv2.putText(placeholder, f"serial={args.view_camera}",
                    (40, 290), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                    (200, 200, 200), 2, cv2.LINE_AA)
        cv2.imshow(view_window_name, placeholder)
        cv2.waitKey(1)
        print(f"[vla] cv2 image viewer: '{view_window_name}' "
              f"(camera={args.view_camera}, scale={args.view_scale})")
        if args.record_view:
            record_dir = os.path.dirname(args.record_view)
            if record_dir:
                os.makedirs(record_dir, exist_ok=True)
            print(f"[vla] recording cv2 viewer to {args.record_view} "
                  f"at {record_fps:.1f} fps")

    print(f"[vla] viser is live at http://0.0.0.0:{args.port}")
    if vla_client:
        print("[vla] use the 'VLA' GUI folder: Query Policy → Execute Plan.")
    print("[vla] Ctrl-C, 'q' in image window, or 'Quit' button in viser to exit.")

    live_frames = {}
    if arm is not None:
        live_frames["urdf_link6"] = vis.server.scene.add_frame(
            "/live_frames/urdf_link6",
            position=(0, 0, 0),
            wxyz=(1, 0, 0, 0),
            show_axes=True,
            axes_length=0.06,
            axes_radius=0.004,
        )
        live_frames["urdf_wrist"] = vis.server.scene.add_frame(
            "/live_frames/urdf_wrist",
            position=(0, 0, 0),
            wxyz=(1, 0, 0, 0),
            show_axes=True,
            axes_length=0.08,
            axes_radius=0.005,
        )
        live_frames["sdk_tcp"] = vis.server.scene.add_frame(
            "/live_frames/sdk_tcp",
            position=(0, 0, 0),
            wxyz=(1, 0, 0, 0),
            show_axes=True,
            axes_length=0.12,
            axes_radius=0.006,
        )

    period = 1.0 / args.rate
    # exit_event was created earlier (Quit button binds to it)
    last_diag_print = 0.0
    DIAG_INTERVAL = 3.0

    try:
        while not exit_event.is_set():
            tic = time.perf_counter()

            # ── Live robot state → viser ────────────────────────────────────
            cfg_parts = []
            arm_qpos = None
            if arm is not None:
                arm_data = arm.get_data()
                arm_qpos = np.asarray(arm_data['qpos'], dtype=float)
                cfg_parts.append(arm_qpos)
            else:
                if args.hand:
                    cfg_parts.append(np.zeros(6))
            if hand is not None:
                cfg_parts.append(live_hand_qpos_for_urdf(args.hand, hand))
            if cfg_parts:
                cfg = np.concatenate(cfg_parts)
                try:
                    vis.robot_dict["robot"].update_cfg(cfg)
                except Exception as e:
                    print(f"[vla] update_cfg failed (cfg shape={cfg.shape}): {e}")
            if arm is not None and arm_qpos is not None:
                try:
                    link6_pos, link6_rotvec = fk_link6(fk_rm, arm_qpos[:6])
                    link6_T = pose_vec_to_homo(link6_pos, link6_rotvec)
                    wrist_T = link6_T @ ARM_TO_HAND_OFFSET
                    update_frame_handle(live_frames["urdf_link6"], link6_T)
                    update_frame_handle(live_frames["urdf_wrist"], wrist_T)

                    ret, sdk_pose = arm.arm.get_position_aa(is_radian=True)
                    if ret == 0:
                        sdk_pos_m = np.asarray(sdk_pose[:3], dtype=float) / 1000.0
                        sdk_rotvec = np.asarray(sdk_pose[3:], dtype=float)
                        sdk_T = pose_vec_to_homo(sdk_pos_m, sdk_rotvec)
                        update_frame_handle(live_frames["sdk_tcp"], sdk_T)
                except Exception as e:
                    if (time.time() - last_diag_print) > DIAG_INTERVAL:
                        last_diag_print = time.time()
                        print(f"[vla] live frame update failed: {e}")

            # ── cv2 image-viewer ────────────────────────────────────────────
            if show_view:
                display_frame = None
                if args.view_camera == "all":
                    frames = cam_source.latest_all()
                    if frames:
                        from paradex.image.merge import merge_image
                        merged = merge_image(frames, {k: k for k in frames})
                        if args.view_scale != 1.0:
                            merged = cv2.resize(merged, None,
                                fx=args.view_scale, fy=args.view_scale,
                                interpolation=cv2.INTER_NEAREST)
                        display_frame = merged
                        cv2.imshow(view_window_name, merged)
                else:
                    bgr = cam_source.latest(args.view_camera)
                    if bgr is not None:
                        if args.view_scale != 1.0:
                            bgr = cv2.resize(bgr, None,
                                fx=args.view_scale, fy=args.view_scale,
                                interpolation=cv2.INTER_NEAREST)
                        display_frame = bgr
                        cv2.imshow(view_window_name, bgr)

                if args.record_view and display_frame is not None:
                    if record_writer is None:
                        h, w = display_frame.shape[:2]
                        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                        record_writer = cv2.VideoWriter(
                            args.record_view, fourcc, record_fps, (w, h))
                        if not record_writer.isOpened():
                            raise RuntimeError(
                                f"failed to open video writer: {args.record_view}")
                    record_writer.write(display_frame)

                # Diagnostic if no frame yet on selected serial.
                if (args.view_camera != "all"
                        and cam_source.latest(args.view_camera) is None
                        and (time.time() - last_diag_print) > DIAG_INTERVAL):
                    last_diag_print = time.time()
                    avail = cam_source.available_serials()
                    print(f"[vla] waiting for frame on serial "
                          f"'{args.view_camera}'; available: {avail}")

                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    exit_event.set()

            elapsed = time.perf_counter() - tic
            time.sleep(max(0.0, period - elapsed))

    except KeyboardInterrupt:
        print("\n[vla] Ctrl-C received, stopping ...")
    except Exception:
        traceback.print_exc()
    finally:
        if record_writer is not None:
            try: record_writer.release()
            except Exception: pass
        if show_view:
            try: cv2.destroyAllWindows()
            except Exception: pass
        if cam_source is not None:
            cam_source.stop()
        if vla_client is not None:
            vla_client.close()
        if rcc is not None:
            if args.stop_camera_on_exit:
                teardown_cameras(rcc, dc)
            else:
                release_camera_owner(rcc, dc)
        if arm is not None:
            try: arm.end(set_break=False)
            except Exception as e: print(f"[vla] arm.end err: {e}")
        if hand is not None:
            try: hand.end()
            except Exception as e: print(f"[vla] hand.end err: {e}")
        print("[vla] done.")


if __name__ == "__main__":
    main()
