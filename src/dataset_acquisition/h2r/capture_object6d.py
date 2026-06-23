from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parents[3]))
PROJECT_ROOT = Path(__file__).parents[3]

import argparse
import datetime
import json
import os
import time
from typing import Any, Dict

import numpy as np
import trimesh
import zmq

from paradex.calibration.utils import load_current_C2R, save_current_camparam
from paradex.utils.path import shared_dir
from paradex.io.camera_system.remote_camera_controller import remote_camera_controller
from paradex.visualization.robot import RobotModule
from paradex.visualization.visualizer.viser import ViserViewer

import pose_normalize

DEFAULT_ROBOT_URDF = str(
    PROJECT_ROOT / "rsc" / "curobo" / "content" / "assets" / "robot"
    / "inspire_description" / "xarm_inspire_f1_new.urdf"
)
DEFAULT_CUROBO_ROBOT_CFG = str(
    PROJECT_ROOT / "rsc" / "curobo" / "content" / "configs" / "robot"
    / "xarm_inspire_f1_new.yml"
)
# Same xarm home pose used by the taeksoo pipeline (HOME_DEG -> rad applied
# only to the 6 arm joints; hand stays at zero).
HOME_DEG = [-84.9, -25.5, -41.1, 118.8, -1.9, 158.1]

# Make sibling helper modules importable.
_SRC_DIR = str(PROJECT_ROOT / "src")
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)


def _send_rpc_once(addr: str, req: dict, timeout_ms: int = 300000) -> dict:
    ctx = zmq.Context()
    sock = ctx.socket(zmq.REQ)
    sock.setsockopt(zmq.RCVTIMEO, timeout_ms)
    sock.setsockopt(zmq.SNDTIMEO, timeout_ms)
    sock.setsockopt(zmq.LINGER, 0)
    try:
        sock.connect(addr)
        sock.send_json(req)
        return sock.recv_json()
    finally:
        sock.close()
        ctx.term()


def _to_shared_data_path(abs_path: str) -> str:
    norm = os.path.normpath(abs_path)
    marker = f"{os.sep}shared_data{os.sep}"
    if marker in norm:
        tail = norm.split(marker, 1)[1]
        return os.path.join("shared_data", tail).replace(os.sep, "/")
    if norm.endswith(f"{os.sep}shared_data"):
        return "shared_data"
    raise ValueError(f"path is not under shared_data: {abs_path}")


def _to_4x4(mat: np.ndarray) -> np.ndarray:
    mat = np.asarray(mat, dtype=np.float32)
    if mat.shape == (4, 4):
        return mat
    if mat.shape == (3, 4):
        out = np.eye(4, dtype=np.float32)
        out[:3, :] = mat
        return out
    raise ValueError(f"matrix must be 4x4 or 3x4, got {mat.shape}")


def _extract_pose_response(res: Dict[str, Any]) -> Dict[str, Any]:
    pose = res.get("object_6d", res)
    out = {
        "pose_world": pose.get("pose_world"),
        "pose_left_cam": pose.get("pose_left_cam"),
        "R_world": pose.get("R_world"),
        "t_world": pose.get("t_world"),
    }
    if out["pose_world"] is None:
        raise RuntimeError(f"invalid response, pose_world missing: {res}")
    return out


def _resolve_mesh_path(mesh_name: str, mesh_root_dir: str) -> str:
    # local load uses the viser-aligned mesh
    return os.path.join(mesh_root_dir, mesh_name, f"{mesh_name}_viser.obj")


def _add_debug_images_to_gui(
    vis,
    debug_dir: str,
    *,
    anchor_xyz=(1.0, 1.0, 0.0),
    plane_width: float = 1.0,
    spacing: float = 0.05,
) -> int:
    """Show RPC-side mask overlay PNGs both in the GUI sidebar AND as large
    floating planes above the scene so they are readable without squinting.

    The 6D pose server writes <abs_save_path>/debug/mask_overlay_*.png on each
    inference call. We add a small thumbnail in the GUI for quick toggling, and
    place full-resolution image planes above the workspace as a billboard the
    user can rotate around."""
    import glob
    import cv2

    if not os.path.isdir(debug_dir):
        return 0
    paths = sorted(glob.glob(os.path.join(debug_dir, "*.png"))
                   + glob.glob(os.path.join(debug_dir, "*.jpg")))
    if not paths:
        return 0

    images = []
    for p in paths:
        bgr = cv2.imread(p, cv2.IMREAD_COLOR)
        if bgr is None:
            continue
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        # left-right flip so the plane normal flip done in scene.add_image does
        # not produce a mirrored image when viewed from the front.
        rgb = np.ascontiguousarray(rgb[:, ::-1, :])
        images.append((p, rgb))
    if not images:
        return 0

    # GUI thumbnails (sidebar): keep them as a quick reference / on-off toggle.
    folder = vis.server.gui.add_folder(f"Debug ({len(images)})")
    with folder:
        for p, img in images:
            vis.server.gui.add_image(img, label=os.path.basename(p))

    # Scene billboards: stack horizontally along +X above anchor_xyz. Plane is
    # rotated rot_x(-90deg) so the image faces +Y AND is right-side up from the
    # default front view (the +rot_x(+90) variant was upside-down).
    anchor = np.asarray(anchor_xyz, dtype=float)
    wxyz_face_neg_y = (float(np.cos(np.pi / 4)), float(-np.sin(np.pi / 4)), 0.0, 0.0)
    x_cursor = -0.5 * (len(images) - 1) * (plane_width + spacing)
    for p, img in images:
        h, w = img.shape[:2]
        rw = float(plane_width)
        rh = float(plane_width) * (float(h) / float(max(w, 1)))
        pos = (anchor[0] + x_cursor, anchor[1], anchor[2] + 0.5 * rh)
        vis.server.scene.add_image(
            name=f"/debug_images/{os.path.basename(p)}",
            image=img,
            render_width=rw,
            render_height=rh,
            position=pos,
            wxyz=wxyz_face_neg_y,
        )
        x_cursor += plane_width + spacing

    print(f"[debug] added {len(images)} image(s) from {debug_dir} "
          f"(sidebar + scene planes, plane_width={plane_width}m)")
    return len(images)


def _popup_debug_images(debug_dir: str) -> int:
    """Open the RPC-side mask overlay PNGs in the OS default image viewer as
    standalone popup windows, independent of the viser scene."""
    import glob
    import subprocess

    if not os.path.isdir(debug_dir):
        return 0
    paths = sorted(glob.glob(os.path.join(debug_dir, "*.png"))
                   + glob.glob(os.path.join(debug_dir, "*.jpg")))
    if not paths:
        return 0
    for p in paths:
        try:
            subprocess.Popen(["xdg-open", p],
                             stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        except Exception as e:
            print(f"[debug] popup failed for {p}: {e}")
    print(f"[debug] opened {len(paths)} debug image(s) as OS popups from {debug_dir}")
    return len(paths)


def _load_mesh(mesh_path: str) -> trimesh.Trimesh:
    geom = trimesh.load(mesh_path, force="mesh", process=False)
    if isinstance(geom, trimesh.Trimesh):
        return geom
    if isinstance(geom, trimesh.Scene):
        geoms = [g for g in geom.geometry.values() if isinstance(g, trimesh.Trimesh)]
        if not geoms:
            raise ValueError(f"no mesh geometry in scene: {mesh_path}")
        return trimesh.util.concatenate(geoms)
    if isinstance(geom, list):
        return trimesh.util.concatenate(geom)
    raise ValueError(f"unsupported mesh type: {type(geom)}")


def _solve_ik_chain_curobo(robot_cfg: Dict[str, Any], target_poses: np.ndarray,
                           seed_arm_qpos: np.ndarray, device: str = "cuda:0"):
    """Per-waypoint IK with continuity warm-start. Sets up curobo IKSolver
    once and calls solve_single in a loop, feeding the previous solution as
    the seed for the next.

    target_poses: (W, 4, 4) eef targets in robot frame
    seed_arm_qpos: (active_dof,) initial seed
    returns: (W, active_dof) IK solutions; failures hold the previous solution.
    """
    sys.path.insert(0, str(PROJECT_ROOT / "curobo" / "src"))
    import torch
    from curobo.types.base import TensorDeviceType
    from curobo.types.math import Pose
    from curobo.wrap.reacher.ik_solver import IKSolver, IKSolverConfig
    from object6d.capture_rcc import _rotmat_to_wxyz

    dev = torch.device(device if ("cuda" in str(device) and torch.cuda.is_available()) else "cpu")
    tensor_args = TensorDeviceType(device=dev)
    ik_cfg = IKSolverConfig.load_from_robot_config(
        robot_cfg, None, tensor_args=tensor_args,
        num_seeds=32, position_threshold=0.005, rotation_threshold=0.1,
        self_collision_check=True, self_collision_opt=True,
        use_cuda_graph=False,
    )
    ik = IKSolver(ik_cfg)

    W = int(target_poses.shape[0])
    dof = int(np.asarray(seed_arm_qpos).reshape(-1).shape[0])
    out = np.zeros((W, dof), dtype=np.float32)
    last = np.asarray(seed_arm_qpos, dtype=np.float32).copy()

    n_succ = 0
    for i in range(W):
        T = target_poses[i]
        wxyz = _rotmat_to_wxyz(T[:3, :3])
        pose = Pose(
            position=torch.tensor(T[:3, 3], device=dev, dtype=torch.float32).view(1, 3),
            quaternion=torch.tensor(wxyz, device=dev, dtype=torch.float32).view(1, 4),
        )
        seed_t = torch.tensor(last, device=dev, dtype=torch.float32).view(1, 1, dof)
        retract_t = torch.tensor(last, device=dev, dtype=torch.float32).view(1, dof)
        result = ik.solve_single(pose, retract_config=retract_t, seed_config=seed_t)
        succ = bool(result.success.flatten()[0].item()) if hasattr(result.success, "flatten") else bool(result.success)
        if succ:
            sol = result.solution.detach().cpu().numpy().reshape(-1)[:dof]
            last = sol.astype(np.float32)
            n_succ += 1
        out[i] = last
    print(f"[IK chain] {n_succ}/{W} succeeded")
    return out


# Order on the wire (matches /right/joint_states' name list and what the
# ros2_control position_controller actually consumes -- thumb_1 FIRST, then
# thumb_2). NOTE: state_receiver._publish_hand_command reverses the array AND
# swaps thumb_1<->thumb_2, so we publish directly to its cmd_pubs[side]
# instead of going through move().
HAND_MSG_NAMES = [
    "right_thumb_1_joint", "right_thumb_2_joint",
    "right_index_1_joint", "right_middle_1_joint",
    "right_ring_1_joint",  "right_little_1_joint",
]
# Per-joint raw motor values for the inspire f1 hand (right side). These are
# the hardware-side endpoints; URDF qpos is linearly mapped between them.
HAND_RAW_OPEN = {
    "right_thumb_1_joint":  1800.0,
    "right_thumb_2_joint":  1350.0,
    "right_index_1_joint":  1740.0,
    "right_middle_1_joint": 1740.0,
    "right_ring_1_joint":   1740.0,
    "right_little_1_joint": 1740.0,
}
HAND_RAW_CLOSE = {
    "right_thumb_1_joint":   600.0,
    "right_thumb_2_joint":  1100.0,
    "right_index_1_joint":   900.0,
    "right_middle_1_joint":  900.0,
    "right_ring_1_joint":    900.0,
    "right_little_1_joint":  900.0,
}
# URDF joint upper limits used to normalize qpos -> [0,1] before lerp.
HAND_Q_UPPER = {
    "right_thumb_1_joint":  2.0944,
    "right_thumb_2_joint":  0.4746,
    "right_index_1_joint":  1.5286,
    "right_middle_1_joint": 1.5286,
    "right_ring_1_joint":   1.5286,
    "right_little_1_joint": 1.5286,
}
HAND_RAW_CLIP = (0.0, 1800.0)


def _hand_qpos_to_inspire_msg(full_qpos: np.ndarray, full_joint_names) -> np.ndarray:
    """URDF rad -> 6 inspire_f1 raw motor counts in the controller's wire
    order [thumb_1, thumb_2, index, middle, ring, little] (matches the
    /right/joint_states name list). Linear interpolation between per-joint
    open/close raw endpoints, normalized by the URDF joint upper limit."""
    idx = {n: i for i, n in enumerate(full_joint_names)}
    lo, hi = HAND_RAW_CLIP
    out = np.zeros(6, dtype=np.float64)
    for i, name in enumerate(HAND_MSG_NAMES):
        q = float(full_qpos[idx[name]])
        q_upper = HAND_Q_UPPER[name]
        t = float(np.clip(q / q_upper, 0.0, 1.0))
        raw = HAND_RAW_OPEN[name] + (HAND_RAW_CLOSE[name] - HAND_RAW_OPEN[name]) * t
        out[i] = float(np.clip(np.rint(raw), lo, hi))
    return out


def _wait_xarm_at_target(xarm, target_rad: np.ndarray, tol_rad: float, timeout: float) -> bool:
    deadline = time.time() + float(timeout)
    while time.time() < deadline:
        data = xarm.get_data()
        cur = data.get("qpos")
        if cur is not None and np.all(np.isfinite(cur)):
            if np.max(np.abs(np.asarray(cur) - target_rad)) < tol_rad:
                return True
        time.sleep(0.05)
    return False


def _execute_on_real_robot(args, pipeline_result: Dict[str, Any], robot_module: RobotModule,
                          abs_save_path: str) -> None:
    """Stream the planned trajectory to the real xarm + inspire_f1, with
    optional camera recording via CaptureSession. Mirrors taeksoo_send.py's
    safety procedure (home first, wait for arrival, confirm prompt, streaming
    loop, cleanup on any error)."""
    from paradex.io.robot_controller import get_arm, get_hand
    from paradex.calibration.utils import save_current_C2R

    full_traj = np.asarray(pipeline_result["full_traj"], dtype=np.float64)
    Tn = int(full_traj.shape[0])
    full_joint_names = list(robot_module.get_joint_names())

    # arm: joint1..joint6 indices in URDF order
    arm_idx = [full_joint_names.index(f"joint{i}") for i in range(1, 7)]
    arm_traj = full_traj[:, arm_idx]  # (T,6) rad

    # hand: raw counts in WIRE order [thumb_1, thumb_2, index, middle, ring, little]
    hand_msgs = np.zeros((Tn, 6), dtype=np.float64)
    for t in range(Tn):
        hand_msgs[t] = _hand_qpos_to_inspire_msg(full_traj[t], full_joint_names)

    # Grasp squeeze: run the whole grasp sequence (approach + hand_interp +
    # replay) with the planned hand values UNCHANGED. Once the grasp is done we
    # insert an explicit squeeze step -- a gradual hand-only ramp that lowers
    # every joint except thumb_1 by --grasp_squeeze raw -- and then the
    # grasp_hold + lift phases keep that tighter pose.
    n_a = int(pipeline_result.get("n_approach", 0))
    n_b = int(pipeline_result.get("n_hand_interp", 0))
    n_r = int(pipeline_result.get("n_replay", 0))
    squeeze_start = n_a + n_b + n_r
    squeeze_amt = float(args.grasp_squeeze)
    squeeze_steps = int(args.squeeze_steps)
    do_squeeze = squeeze_amt != 0.0 and 0 < squeeze_start < Tn
    if do_squeeze:
        # hold + lift frames keep the squeezed pose (thumb_1 / col 0 untouched).
        hand_msgs[squeeze_start:, 1:] = np.clip(
            hand_msgs[squeeze_start:, 1:] - squeeze_amt, 0.0, 1800.0
        )
        print(f"[exec] grasp squeeze -{squeeze_amt:.0f} over {squeeze_steps} "
              f"frames after frame {squeeze_start} (thumb_1 excluded)")

    # Per-frame streaming period. The pregrasp->grasp replay segment is slowed
    # to --replay_rate_hz so the hand does not close on the object too fast;
    # every other phase runs at --rate_hz.
    periods = np.full(Tn, 1.0 / float(args.rate_hz), dtype=np.float64)
    replay_lo, replay_hi = n_a + n_b, n_a + n_b + n_r
    if replay_hi > replay_lo:
        periods[replay_lo:replay_hi] = 1.0 / float(args.replay_rate_hz)
    print(f"[exec] replay segment frames {replay_lo}..{replay_hi - 1} "
          f"@ {args.replay_rate_hz} Hz (rest @ {args.rate_hz} Hz)")

    home_rad = np.deg2rad(np.asarray(args.home_deg, dtype=np.float64))
    if not np.allclose(arm_traj[0], home_rad, atol=1e-3):
        print(f"[exec] WARNING: traj[0] arm != HOME; traj[0]={arm_traj[0]}")
    print(f"[exec] T={Tn}, rate={args.rate_hz} Hz (~{Tn / args.rate_hz:.1f}s)")
    print(f"[exec] arm[0] deg: {np.rad2deg(arm_traj[0]).round(2).tolist()}")
    print(f"[exec] hand[0] raw  [thumb1,thumb2,idx,mid,ring,little]: {hand_msgs[0].tolist()}")
    print(f"[exec] hand[{Tn-1}] raw [thumb1,thumb2,idx,mid,ring,little]: {hand_msgs[-1].tolist()}")

    # Optional video capture (CaptureSession) -- lives next to the planning
    # artefacts inside abs_save_path/video.
    cs = None
    capture_dir_abs = None
    if args.record_video:
        from paradex.dataset_acqusition.capture import CaptureSession
        capture_rel_path = os.path.join(_to_shared_data_path(abs_save_path).replace("shared_data/", ""),
                                        "video", datetime.datetime.now().strftime("%Y%m%d_%H%M%S"))
        capture_dir_abs = os.path.join(shared_dir, capture_rel_path)
        cs = CaptureSession(camera=True, realsense=bool(args.realsense))
        cs.start(capture_rel_path)
        print(f"[exec] recording video -> {capture_dir_abs}/raw")
        print("[exec] pre-roll: 3s before motion ...")
        time.sleep(3.0)

    print("[exec] init xarm (servo_api=angle_j)...")
    xarm = get_arm("xarm", servo_api="angle_j")
    print(f"[exec] init inspire_f1 (hand_side={args.hand_side})...")
    hand = get_hand("inspire_f1", hand_side=args.hand_side)

    # The state_receiver's move() reverses the array AND swaps thumb_1<->thumb_2.
    # We bypass it and publish to its cmd_pubs[side] directly with the wire-order
    # array [thumb_1, thumb_2, index, middle, ring, little].
    from std_msgs.msg import Float64MultiArray
    hand_pub = hand.cmd_pubs[args.hand_side]

    def _send_hand(raw_wire: np.ndarray) -> None:
        msg = Float64MultiArray()
        msg.data = [float(x) for x in raw_wire]
        hand_pub.publish(msg)

    try:
        # Phase 0: home + open hand, wait for arrival.
        print(f"[exec] commanding HOME deg={list(np.rad2deg(home_rad).round(2))} + hand[0]={hand_msgs[0].tolist()}")
        xarm.move(home_rad, is_servo=True)
        _send_hand(hand_msgs[0])
        ok = _wait_xarm_at_target(xarm, home_rad,
                                  np.deg2rad(float(args.home_tol_deg)),
                                  float(args.home_timeout))
        if not ok:
            print(f"[exec] WARNING: arm did not reach HOME within {args.home_timeout}s")
        if xarm.is_error() or hand.is_error():
            raise RuntimeError("controller in error state before streaming")

        # Phase 1: grasp sequence -- stream frames up to the end of replay with
        # the planned hand values unchanged. Per-frame period from `periods`.
        grasp_end = squeeze_start if do_squeeze else Tn
        print(f"[exec] streaming grasp sequence: frames 0..{grasp_end - 1} ...")
        t0 = time.perf_counter()
        clock = 0.0
        for t in range(grasp_end):
            xarm.move(arm_traj[t], is_servo=True)
            _send_hand(hand_msgs[t])
            if xarm.is_error() or hand.is_error():
                raise RuntimeError(f"controller error at frame {t}")
            clock += periods[t]
            sleep_for = t0 + clock - time.perf_counter()
            if sleep_for > 0:
                time.sleep(sleep_for)

        # Phase 1.5: explicit squeeze -- arm frozen at the last grasp pose,
        # hand ramps from the planned grasp pose to the squeezed pose.
        if do_squeeze:
            hand_from = hand_msgs[squeeze_start - 1].copy()
            hand_to = hand_msgs[squeeze_start].copy()
            arm_hold = arm_traj[squeeze_start - 1]
            print(f"[exec] squeezing hand over {squeeze_steps} frames ...")
            for s in range(1, squeeze_steps + 1):
                a = float(s) / float(squeeze_steps)
                cmd = (1.0 - a) * hand_from + a * hand_to
                xarm.move(arm_hold, is_servo=True)
                _send_hand(cmd)
                if xarm.is_error() or hand.is_error():
                    raise RuntimeError(f"controller error during squeeze step {s}")
                time.sleep(1.0 / float(args.rate_hz))

            # Phase 2: hold + lift with the squeezed hand pose.
            print(f"[exec] streaming hold + lift: frames {squeeze_start}..{Tn - 1} ...")
            t0 = time.perf_counter()
            clock = 0.0
            for t in range(squeeze_start, Tn):
                xarm.move(arm_traj[t], is_servo=True)
                _send_hand(hand_msgs[t])
                if xarm.is_error() or hand.is_error():
                    raise RuntimeError(f"controller error at frame {t}")
                clock += periods[t]
                sleep_for = t0 + clock - time.perf_counter()
                if sleep_for > 0:
                    time.sleep(sleep_for)
        print("[exec] streaming done")
        if cs is not None:
            print("[exec] holding final pose for 10s of post-roll video ...")
            time.sleep(10.0)
    finally:
        try: xarm.end(set_break=False)
        except Exception as e: print(f"[exec] xarm.end error: {e}")
        try: hand.end()
        except Exception as e: print(f"[exec] hand.end error: {e}")
        if cs is not None:
            try: cs.stop()
            except Exception as e: print(f"[exec] cs.stop error: {e}")
            try: save_current_C2R(capture_dir_abs)
            except Exception as e: print(f"[exec] save_current_C2R error: {e}")
            try: cs.end()
            except Exception as e: print(f"[exec] cs.end error: {e}")
            print(f"[exec] capture saved to {capture_dir_abs}")


def _run_h2r_pipeline(args, mesh_name: str, pose_robot: np.ndarray,
                     robot_module: RobotModule, abs_save_path: str) -> Dict[str, Any]:
    """End-to-end h2r pipeline (xarm_inspire_f1):

       Phase A: HOME -> wp0 (pregrasp) via curobo MotionGen
       Phase B: per-waypoint IK to track the canonical wrist trajectory while
                replaying finger qpos from result.npz

    The two phases are concatenated into a single (Na+W, n_dof) trajectory in
    URDF joint order. The viewer is intentionally NOT touched here: caller
    receives the full trajectory and registers it with add_traj after viser is
    launched, so the long curobo planning step does not block the GUI."""
    # Curobo builds retract_config on its module-default cuda device (cuda:0)
    # during MotionGenConfig.load_from_robot_config, then forward_bound_pos_warp
    # fails when tensor_args.device is anything else. Constrain
    # CUDA_VISIBLE_DEVICES before importing torch/curobo so cuda:0 always maps
    # to the desired physical GPU, and tell our helpers to use cuda:0.
    curobo_device = str(args.curobo_device)
    if curobo_device.startswith("cuda:"):
        gpu_id = curobo_device.split(":", 1)[1]
        os.environ.setdefault("CUDA_VISIBLE_DEVICES", gpu_id)
        print(f"[pipe] CUDA_VISIBLE_DEVICES={os.environ['CUDA_VISIBLE_DEVICES']} "
              f"(curobo will use cuda:0 internally)")
        curobo_device = "cuda:0"

    from object6d.capture_rcc import (
        _run_curobo_plan,
        _normalize_curobo_robot_cfg_paths,
        _normalize_curobo_robot_cfg_paths_with_cfg_file,
        _rotmat_to_wxyz,
    )
    from paradex.utils.file_io import load_yaml
    from scipy.spatial.transform import Rotation

    # ---- 1. load pre-recorded canonical grasp trajectory
    res_path = os.path.join(args.result_root, mesh_name, str(args.scene), "result.npz")
    if not os.path.exists(res_path):
        raise FileNotFoundError(f"no packed result: {res_path}")
    d = np.load(res_path, allow_pickle=True)
    key = f"{args.src}_pregrasp_traj"
    if key not in d.files:
        raise KeyError(f"{key} missing in {res_path}")
    traj_canon = d[key][args.particle]  # (W, 18) = wrist[6] + finger[12]
    jnames_traj = [str(x) for x in d["revolute_joint_names"]]
    W = int(traj_canon.shape[0])
    print(f"[pipe] {mesh_name} scene={args.scene} src={args.src} "
          f"particle={args.particle}  W={W} (wrist+finger)")

    # ---- 2. canonical wrist -> robot frame (object already at pose_robot)
    wrist_T_robot = np.zeros((W, 4, 4), dtype=np.float32)
    for i, w in enumerate(traj_canon):
        Wc = np.eye(4, dtype=np.float32)
        Wc[:3, :3] = Rotation.from_euler("XYZ", w[3:6]).as_matrix()
        Wc[:3, 3] = w[:3]
        wrist_T_robot[i] = (pose_robot @ Wc).astype(np.float32)

    # ---- 3. HOME qpos in full URDF joint order. The hand is pre-shaped to the
    # FIRST replay finger pose from the start, so there is no open->grasp
    # interpolation after reaching pregrasp (the fingers are already in the
    # grasp config during the whole approach).
    full_joint_names = list(robot_module.get_joint_names())
    n_dof = int(robot_module.get_num_joints())
    name_to_full_idx = {n: i for i, n in enumerate(full_joint_names)}
    name_to_finger_idx = {n: i for i, n in enumerate(jnames_traj)}
    home_qpos_full = np.zeros(n_dof, dtype=np.float32)
    home_qpos_full[:6] = np.deg2rad(np.asarray(args.home_deg, dtype=np.float32))
    for jn, fi in name_to_finger_idx.items():
        if jn in name_to_full_idx:
            home_qpos_full[name_to_full_idx[jn]] = float(traj_canon[0, 6 + fi])

    # ---- 4. curobo robot_cfg (xarm_inspire_f1; cspace=12, active=6 arm)
    cfg_path = os.path.expanduser(args.curobo_robot_cfg)
    if not os.path.exists(cfg_path):
        raise FileNotFoundError(f"curobo cfg not found: {cfg_path}")
    robot_cfg = _normalize_curobo_robot_cfg_paths_with_cfg_file(load_yaml(cfg_path), cfg_path)
    robot_cfg = _normalize_curobo_robot_cfg_paths(robot_cfg)
    # The cfg locks the 6 hand joints (open). Re-lock them to the pre-shaped
    # grasp pose so Phase A collision checking uses the actual hand shape the
    # arm will carry during the approach.
    lock_joints = robot_cfg["robot_cfg"]["kinematics"].get("lock_joints")
    if isinstance(lock_joints, dict):
        for jn in lock_joints:
            if jn in name_to_full_idx:
                lock_joints[jn] = float(home_qpos_full[name_to_full_idx[jn]])

    # ---- 5. world (table + current object as collision mesh)
    table_size = np.array([float(x) for x in args.table_size_xyz.split(",")], dtype=np.float32)
    table_pos = np.array([float(x) for x in args.table_pos_xyz.split(",")], dtype=np.float32)
    obj_wxyz = _rotmat_to_wxyz(pose_robot[:3, :3])
    world_dict = {
        "cuboid": {
            "table": {
                "dims": table_size.tolist(),
                "pose": [float(table_pos[0]), float(table_pos[1]), float(table_pos[2]),
                         1.0, 0.0, 0.0, 0.0],
            }
        },
        "mesh": {
            "object": {
                "file_path": _resolve_mesh_path(mesh_name, os.path.expanduser(args.mesh_root_dir)),
                "pose": [float(pose_robot[0, 3]), float(pose_robot[1, 3]), float(pose_robot[2, 3]),
                         float(obj_wxyz[0]), float(obj_wxyz[1]), float(obj_wxyz[2]), float(obj_wxyz[3])],
            }
        },
    }

    # ---- 6. Phase A: HOME -> pregrasp via curobo MotionGen
    cspace_jn = list(robot_cfg["robot_cfg"]["kinematics"]["cspace"]["joint_names"])
    start_qpos_cspace = np.zeros(len(cspace_jn), dtype=np.float32)
    for ci, jn in enumerate(cspace_jn):
        if jn in name_to_full_idx:
            start_qpos_cspace[ci] = home_qpos_full[name_to_full_idx[jn]]
    print("[pipe] Phase A: planning HOME -> wp0 (pregrasp)")
    plan = _run_curobo_plan(
        robot_cfg=robot_cfg,
        world_cfg_dict=world_dict,
        start_qpos=start_qpos_cspace,
        target_eef_pose_robot=wrist_T_robot[0],
        device=curobo_device,
        seeds=int(args.curobo_seeds),
        self_collision=True,
    )
    if not plan.get("success"):
        print(f"[pipe] Phase A FAILED: {plan.get('status')} {plan.get('error')}")
        return
    approach_active = np.asarray(plan["trajectory"], dtype=np.float32)  # (Na, active_dof)
    active_jn = list(plan.get("active_joint_names", []) or cspace_jn)
    Na = int(approach_active.shape[0])
    print(f"[pipe] Phase A done: {Na} waypoints, active_dof={approach_active.shape[1]}")

    # ---- 7. Phase B: per-frame IK to follow wrist trajectory
    print(f"[pipe] Phase B: per-waypoint IK across {W} waypoints")
    replay_active = _solve_ik_chain_curobo(
        robot_cfg=robot_cfg,
        target_poses=wrist_T_robot,
        seed_arm_qpos=approach_active[-1],
        device=curobo_device,
    )

    # ---- 8. expand active qpos to full URDF order; overlay hand qpos
    name_to_active_idx = {n: i for i, n in enumerate(active_jn)}

    # Phase A: arm from approach plan, hand stays at the pre-shaped grasp pose.
    approach_full = np.tile(home_qpos_full, (Na, 1))
    for jn, ai in name_to_active_idx.items():
        if jn in name_to_full_idx:
            approach_full[:, name_to_full_idx[jn]] = approach_active[:, ai]

    # Phase B: arm from replay IK, hand from canonical traj by joint name.
    replay_full = np.tile(approach_full[-1], (W, 1))
    for jn, ai in name_to_active_idx.items():
        if jn in name_to_full_idx:
            replay_full[:, name_to_full_idx[jn]] = replay_active[:, ai]
    for jn, fi in name_to_finger_idx.items():
        if jn in name_to_full_idx:
            replay_full[:, name_to_full_idx[jn]] = traj_canon[:, 6 + fi].astype(np.float32)

    # Hand pre-close bridge: after reaching the pregrasp arm pose (end of
    # Phase A), interpolate hand from open -> replay[0] hand pose over
    # ``--hand_interp_steps`` frames before starting the wrist-tracking replay.
    # Arm joints stay constant during the bridge so only the fingers move.
    nh = int(max(0, args.hand_interp_steps))
    bridge_full = np.zeros((0, n_dof), dtype=np.float32)
    q_a_end = approach_full[-1]
    q_b_start = replay_full[0]
    if nh > 0 and not np.allclose(q_a_end, q_b_start, atol=1e-6):
        # Include both endpoints in the linspace, then drop them: endpoint 0 is
        # already approach[-1] and endpoint 1 is replay[0], so emitting the
        # interior (nh-1) points gives a smooth ramp without duplicating either.
        if nh >= 2:
            alphas = np.linspace(0.0, 1.0, nh + 1, dtype=np.float32)[1:-1]
        else:
            alphas = np.array([0.5], dtype=np.float32)
        bridge_full = (1.0 - alphas[:, None]) * q_a_end[None, :] + alphas[:, None] * q_b_start[None, :]

    Nb = int(bridge_full.shape[0])

    # ---- 8bc. Hold the grasp pose still for a few frames so the gripper has
    # time to settle (mostly visual; arm and hand do not move during the dwell).
    Nh = int(max(0, args.grasp_hold_frames))
    hold_full = np.tile(replay_full[-1], (Nh, 1)) if Nh > 0 else np.zeros((0, n_dof), dtype=np.float32)

    # ---- 8c. Phase C: lift +Z after grasp (curobo MotionGen, object removed
    # from collision world since we are holding it). Start qpos is the last
    # replay arm config; hand is held at the last grasp pose throughout.
    lift_h = float(args.lift_height_m)
    Nc = 0
    lift_full = np.zeros((0, n_dof), dtype=np.float32)
    if lift_h > 0:
        print(f"[pipe] Phase C: lift +{lift_h * 100:.1f}cm in robot +Z")
        lift_target = wrist_T_robot[-1].copy()
        lift_target[2, 3] += lift_h

        cspace_idx = {c: ci for ci, c in enumerate(cspace_jn)}
        start_lift_cspace = start_qpos_cspace.copy()
        for ai, jn in enumerate(active_jn):
            if jn in cspace_idx:
                start_lift_cspace[cspace_idx[jn]] = replay_active[-1, ai]

        # Phase C is a short straight-up motion holding the object; we don't
        # want curobo to reject the start pose because the gripper is touching
        # the object/table. Replace the world with a single tiny dummy cuboid
        # parked 10 m away so curobo's primitive collision checker has at least
        # one obstacle (it errors out on a fully empty world).
        lift_world = {
            "cuboid": {
                "_dummy_far": {
                    "dims": [0.01, 0.01, 0.01],
                    "pose": [10.0, 10.0, 10.0, 1.0, 0.0, 0.0, 0.0],
                }
            },
            "mesh": {},
        }
        lift_plan = _run_curobo_plan(
            robot_cfg=robot_cfg,
            world_cfg_dict=lift_world,
            start_qpos=start_lift_cspace,
            target_eef_pose_robot=lift_target,
            device=curobo_device,
            seeds=int(args.curobo_seeds),
            self_collision=True,
        )
        if lift_plan.get("success"):
            lift_active = np.asarray(lift_plan["trajectory"], dtype=np.float32)
            # Drop the first sample so we don't repeat replay's last frame.
            if lift_active.shape[0] > 1:
                lift_active = lift_active[1:]
            Nc = int(lift_active.shape[0])
            lift_full = np.tile(replay_full[-1], (Nc, 1))
            for jn, ai in name_to_active_idx.items():
                if jn in name_to_full_idx:
                    lift_full[:, name_to_full_idx[jn]] = lift_active[:, ai]
            print(f"[pipe] Phase C done: {Nc} waypoints")
        else:
            print(f"[pipe] Phase C FAILED: {lift_plan.get('status')} {lift_plan.get('error')}")

    full_traj = np.concatenate([approach_full, bridge_full, replay_full, hold_full, lift_full], axis=0)

    # ---- 9. save artefacts next to other capture outputs
    np.save(os.path.join(abs_save_path, "h2r_pipeline_traj.npy"), full_traj)
    with open(os.path.join(abs_save_path, "h2r_pipeline_meta.json"), "w") as f:
        json.dump({"n_approach": Na, "n_hand_interp": Nb, "n_replay": W,
                   "n_grasp_hold": Nh, "n_lift": Nc, "lift_height_m": lift_h,
                   "joint_names": full_joint_names,
                   "active_joint_names": active_jn,
                   "home_deg": list(args.home_deg),
                   "hand_interp_steps_requested": int(args.hand_interp_steps),
                   "grasp_hold_frames_requested": int(args.grasp_hold_frames)}, f, indent=2)
    print(f"[pipe] saved h2r_pipeline_traj.npy shape={full_traj.shape}")

    # ---- 10. caller-facing result; caller adds add_traj after viser starts
    cursor = 0
    ranges = []
    for name, n in [("approach", Na), ("hand_interp", Nb), ("replay", W),
                    ("grasp_hold", Nh), ("lift", Nc)]:
        if n > 0:
            ranges.append((name, cursor, cursor + n - 1))
            cursor += n
    range_str = ", ".join(f"{n}[{lo}..{hi}]" for n, lo, hi in ranges)
    print(f"[pipe] planning done; trajectory shape={full_traj.shape}, "
          f"ranges {range_str}. (viser will open next)")
    return {"full_traj": full_traj, "range_str": range_str,
            "n_approach": Na, "n_hand_interp": Nb, "n_replay": W,
            "n_grasp_hold": Nh, "n_lift": Nc}


def _replay_pregrasp(args, vis, mesh_name: str, pose_robot: np.ndarray) -> None:
    """Add our pre-grasp -> grasp motion ON TOP of the existing scene (the
    xarm_inspire_f1 robot at qpos 0 + the object at pose_robot, both already
    placed by main). The inspire_f1 hand (inspire_float.urdf, floating base) is
    animated along the trajectory.

    The grasp is optimized in the object-CANONICAL frame; the object is shown
    at pose_robot, so the SAME pose_robot transform registers the hand with it
    (hand and object were registered in the canonical frame)."""
    import time

    res_path = os.path.join(args.result_root, mesh_name,
                            str(args.scene), "result.npz")
    if not os.path.exists(res_path):
        raise FileNotFoundError(f"no packed result: {res_path}")
    d = np.load(res_path, allow_pickle=True)
    key = f"{args.src}_pregrasp_traj"
    if key not in d.files:
        raise KeyError(f"{key} missing in {res_path}")
    traj = d[key][args.particle]                           # (W,18) canonical
    jnames = [str(x) for x in d["revolute_joint_names"]]
    print(f"[replay] {mesh_name} scene={args.scene} src={args.src} "
          f"particle={args.particle}  {len(traj)} waypoints")

    # object overlays on obj_verts (canonical -> pose_robot, same as the mesh):
    #   contact map  -- robot/human contact, hot colormap
    #   partition6   -- which of the 6 finger groups each point belongs to
    # a GUI dropdown switches the coloring.
    if "obj_verts" in d.files:
        from matplotlib import cm
        ov = np.asarray(d["obj_verts"], np.float64)
        ov_robot = (ov @ pose_robot[:3, :3].T
                    + pose_robot[:3, 3]).astype(np.float32)
        ckey = f"{args.src}_contact"
        cmap = (np.clip(np.asarray(d[ckey], np.float64), 0.0, 1.0)
                if ckey in d.files else None)
        part6 = (np.asarray(d["partition6"], np.int64).reshape(-1)
                 if "partition6" in d.files else None)
        PART_RGB = np.array([[153, 153, 153], [228, 26, 28], [55, 126, 184],
                             [77, 175, 74], [255, 127, 0], [152, 78, 163]],
                            dtype=np.uint8)

        def draw_overlay(mode):
            if mode == "partition6" and part6 is not None:
                rgb = PART_RGB[np.clip(part6, 0, 5)]
            elif cmap is not None:
                rgb = (cm.hot(cmap)[:, :3] * 255).astype(np.uint8)
            else:
                return
            vis.server.scene.add_point_cloud(
                "/object_overlay", points=ov_robot, colors=rgb,
                point_size=0.004)

        gui_color = vis.server.gui.add_dropdown(
            "object color", ["contact", "partition6"],
            initial_value="contact")
        gui_color.on_update(lambda _: draw_overlay(gui_color.value))
        draw_overlay("contact")
        if cmap is not None:
            print(f"[replay] contact({args.src}): "
                  f"{int((cmap > 0.3).sum())}/{len(cmap)} pts > 0.3")

        # also paint the object MESH itself by partition6: each mesh vertex
        # inherits the nearest obj_vert's partition label. This overlays the
        # default solid mesh added by vis.add_object().
        if part6 is not None:
            from scipy.spatial import cKDTree
            mesh_path_ = _resolve_mesh_path(
                mesh_name, os.path.expanduser(args.mesh_root_dir))
            obj_mesh = _load_mesh(mesh_path_).copy()
            nn = cKDTree(ov).query(np.asarray(obj_mesh.vertices))[1]
            rgb_v = PART_RGB[np.clip(part6[nn], 0, 5)]
            rgba_v = np.concatenate(
                [rgb_v, np.full((len(rgb_v), 1), 255, np.uint8)], axis=1)
            obj_mesh.apply_transform(pose_robot)
            obj_mesh.visual.vertex_colors = rgba_v
            vis.server.scene.add_mesh_trimesh("/object_part_mesh",
                                              mesh=obj_mesh)
            print(f"[replay] partition6 mesh: "
                  f"{len(obj_mesh.vertices)} vertices colored")

    # inspire_f1 hand = RH56F1_R.urdf (no floating base; 12 finger joints + 6
    # mimic). We strip <mimic> and drive all 12 finger joints directly from the
    # trajectory (exact, mimic-consistent). The wrist 6-DOF is NOT a urdf joint
    # -> the per-waypoint wrist pose is applied to the robot's root frame.
    import re
    from scipy.spatial.transform import Rotation
    hand_urdf_path = os.path.expanduser(args.hand_urdf)
    urdf_dir = os.path.dirname(hand_urdf_path)
    txt = re.sub(r"<mimic\b.*?/>", "", open(hand_urdf_path).read(), flags=re.S)
    # package://RH56F1_R/meshes/*.STL -> absolute (meshes live in ../meshes/)
    mesh_abs = os.path.abspath(os.path.join(urdf_dir, "..", "meshes"))
    txt = txt.replace("package://RH56F1_R/meshes/", f"{mesh_abs}/")
    hand_urdf_fixed = os.path.join(urdf_dir, "RH56F1_R_replay.urdf")
    with open(hand_urdf_fixed, "w") as f:
        f.write(txt)

    vis.add_robot("hand", hand_urdf_fixed, pose=pose_robot,
                  include_arm_meshes=True)
    hm = RobotModule(hand_urdf_fixed)            # 12 actuated finger joints
    hand_robot = vis.robot_dict["hand"]

    def finger_cfg(w):
        fv = {jn: w[6 + i] for i, jn in enumerate(jnames)}
        return np.array([fv.get(n, 0.0) for n in hm.get_joint_names()],
                        dtype=np.float32)

    def set_wrist(w):
        # canonical wrist pose -> robot frame (object is also at pose_robot)
        Wc = np.eye(4)
        Wc[:3, :3] = Rotation.from_euler("XYZ", w[3:6]).as_matrix()
        Wc[:3, 3] = w[:3]
        Wr = pose_robot @ Wc
        if hasattr(hand_robot, "_visual_root_frame"):
            hand_robot._visual_root_frame.position = Wr[:3, 3]
            qx, qy, qz, qw = Rotation.from_matrix(Wr[:3, :3]).as_quat()
            hand_robot._visual_root_frame.wxyz = (qw, qx, qy, qz)

    print("[replay] viser running (waypoint 0=pregrasp .. last=grasp). Ctrl+C.")
    wp = 0
    try:
        while True:
            w = traj[wp]
            set_wrist(w)
            hand_robot.update_cfg(finger_cfg(w))
            vis.update()
            wp = (wp + 1) % len(traj)
            time.sleep(0.12)
    except KeyboardInterrupt:
        pass


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", "--mesh_name", dest="mesh_name", required=True)
    parser.add_argument("--save_path", default="capture/h2r/object_6d", help="relative path under shared_data")
    parser.add_argument("--rpc_addr", default="tcp://192.168.0.14:5570")
    parser.add_argument("--rpc_timeout_ms", type=int, default=300000)
    parser.add_argument("--rcc_entry", default="image_main.py")
    parser.add_argument("--mesh_root_dir", default=os.path.join(shared_dir, "mesh_blender"))
    parser.add_argument("--c2r_path", default=None, help="optional C2R.npy override")
    parser.add_argument("--robot_urdf", default=DEFAULT_ROBOT_URDF)
    parser.add_argument("--no_vis", action="store_true")
    parser.add_argument("--viser_addr", default="tcp://localhost:5572",
                        help="ZMQ REQ address of a running viser_server. Push "
                             "the scene + trajectory to it (non-blocking) so the "
                             "execute prompt fires immediately. Empty string => "
                             "fall back to launching a local ViserViewer "
                             "(blocking; prompt fires after Ctrl+C).")
    parser.add_argument("--debug_plane_width", type=float, default=1.0,
                        help="floating debug image plane width (meters)")
    parser.add_argument("--debug_anchor_x", type=float, default=0.5,
                        help="x offset (m) of debug image planes (shift right)")
    parser.add_argument("--debug_anchor_y", type=float, default=-1.0,
                        help="y offset (m) of debug image planes")
    parser.add_argument("--debug_anchor_z", type=float, default=0.0,
                        help="ground height (m) of debug image planes; plane "
                             "center auto-lifted by half its height so the "
                             "bottom edge sits at this z")
    parser.add_argument("--debug_image_mode", default="popup",
                        choices=["popup", "viser", "both", "none"],
                        help="how to show RPC mask overlay PNGs: popup = OS "
                             "image viewer window(s); viser = floating planes "
                             "in the 3D scene; both; none")
    parser.add_argument(
        "--normalize", default="none", choices=pose_normalize.MODES,
        help="per-object 6D pose normalization applied to the RC object pose "
             "(same technique as the H2R grasp pipeline): none / x_roll "
             "(cylindrical) / axis_sign (box) / sphere")
    # --- pre-grasp -> grasp motion replay (our H2R result) ---
    parser.add_argument("--scene", type=int, default=None,
                        help="capture scene/clip number; enables pre-grasp "
                             "motion replay for that clip's stored result")
    parser.add_argument("--src", default="robot", choices=("robot", "human"),
                        help="which contact source's grasp to replay")
    parser.add_argument("--particle", type=int, default=0,
                        help="particle index 0..7, 0 = lowest energy (best)")
    parser.add_argument(
        "--result_root",
        default=os.path.join(shared_dir, "capture", "h2r", "0518"),
        help="root holding <obj>/<scene>/result.npz (pack_results.py output)")
    parser.add_argument(
        "--hand_urdf",
        default=str(PROJECT_ROOT / "rsc" / "robot" / "inspire_f1" / "URDF-2"
                    / "URDF-2" / "RH56F1_R" / "urdf" / "RH56F1_R.urdf"),
        help="inspire_f1 hand URDF (RH56F1_R) for the legacy motion replay")
    # --- new: full h2r pipeline (HOME -> pregrasp -> grasp replay) ---
    parser.add_argument("--pipeline", action="store_true",
                        help="run the curobo h2r pipeline (HOME -> pregrasp via "
                             "MotionGen, per-waypoint IK to follow the wrist "
                             "trajectory). Requires --scene. Trajectory is "
                             "added to the viewer for playback via add_traj.")
    parser.add_argument("--home_deg", nargs=6, type=float, default=HOME_DEG,
                        help="xarm home joint angles in degrees (6 values)")
    parser.add_argument("--curobo_robot_cfg", default=DEFAULT_CUROBO_ROBOT_CFG,
                        help="curobo robot config yaml (xarm_inspire_f1)")
    parser.add_argument("--curobo_device", default="cuda:1")
    parser.add_argument("--curobo_seeds", type=int, default=128)
    parser.add_argument("--table_size_xyz", default="2.4,1.6,0.1")
    parser.add_argument("--table_pos_xyz", default="0.0,0.0,-0.05")
    parser.add_argument("--hand_interp_steps", type=int, default=60,
                        help="bridge frames inserted between Phase A end (arm "
                             "at pregrasp, hand open) and Phase B start (hand "
                             "at first replay finger qpos). Arm is held; only "
                             "fingers ramp. 0 disables the bridge.")
    parser.add_argument("--grasp_hold_frames", type=int, default=30,
                        help="dwell frames inserted after the grasp replay "
                             "(arm + hand held still) before the lift phase. "
                             "0 disables.")
    parser.add_argument("--grasp_squeeze", type=float, default=40.0,
                        help="after the grasp sequence runs, ramp every hand "
                             "joint except thumb_1 down by this raw amount so "
                             "the grasp tightens. 0 disables.")
    parser.add_argument("--squeeze_steps", type=int, default=20,
                        help="number of frames for the explicit squeeze ramp "
                             "(arm frozen, hand-only). At rate_hz this is the "
                             "squeeze duration.")
    parser.add_argument("--lift_height_m", type=float, default=0.07,
                        help="post-grasp lift in robot +Z (meters). 0 disables "
                             "Phase C. Planned via curobo MotionGen with the "
                             "object removed from the collision world.")
    # --- real-robot execution after the viser session ---
    parser.add_argument("--rate_hz", type=float, default=30.0,
                        help="streaming rate to xarm + inspire_f1 (Hz)")
    parser.add_argument("--replay_rate_hz", type=float, default=24.0,
                        help="streaming rate for the pregrasp->grasp replay "
                             "segment only (Hz). Lower = slower close.")
    parser.add_argument("--home_tol_deg", type=float, default=2.0)
    parser.add_argument("--home_timeout", type=float, default=20.0)
    parser.add_argument("--hand_side", default="right", choices=["right", "left"])
    parser.add_argument("--record_video", action="store_true", default=True,
                        help="record video via CaptureSession during execution")
    parser.add_argument("--no_record_video", dest="record_video",
                        action="store_false")
    parser.add_argument("--realsense", action="store_true",
                        help="also record realsense depth (passed to CaptureSession)")
    parser.add_argument("--yes", action="store_true",
                        help="skip the post-viser execute confirmation prompt")
    args = parser.parse_args()

    # mesh existence check (RPC server only needs mesh_name)
    mesh_path = _resolve_mesh_path(args.mesh_name, os.path.expanduser(args.mesh_root_dir))
    if not os.path.exists(mesh_path):
        raise FileNotFoundError(f"mesh not found: {mesh_path}")

    date_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    # When a scene is specified we group runs under it so re-captures of the
    # same scene stay together. The contact source (robot/human) is appended
    # right after the scene number so video + plan artefacts stay separated.
    if args.scene is not None:
        rel_save_path = os.path.join(args.save_path, args.mesh_name,
                                     str(args.scene), args.src, date_str)
    else:
        rel_save_path = os.path.join(args.save_path, args.mesh_name, date_str)
    abs_save_path = os.path.join(shared_dir, rel_save_path)
    os.makedirs(abs_save_path, exist_ok=True)

    # 1) capture image
    rcc = remote_camera_controller(args.rcc_entry)
    save_current_camparam(abs_save_path)
    print(f"Capturing image to {rel_save_path}")
    try:
        rcc.start("image", False, f"shared_data/{rel_save_path}/raw")
        rcc.stop()
    finally:
        rcc.end()
    print(f"image_dir: {abs_save_path}")

    # 2) 6d pose inference via RPC
    req = {
        "command": "infer",
        "image_path": _to_shared_data_path(abs_save_path),
        "mesh_name": args.mesh_name,
    }
    res = _send_rpc_once(args.rpc_addr, req, timeout_ms=args.rpc_timeout_ms)
    pose_out = _extract_pose_response(res)

    object_json_path = os.path.join(abs_save_path, "object_6d.json")
    with open(object_json_path, "w", encoding="utf-8") as f:
        json.dump(pose_out, f, ensure_ascii=False, indent=2)
    print(f"saved: {object_json_path}")
    print(json.dumps(pose_out, ensure_ascii=False, indent=2))

    # 3) coordinate transform: world -> robot frame
    if args.c2r_path is not None:
        c2r = np.load(os.path.expanduser(args.c2r_path))
    else:
        c2r = load_current_C2R()
    c2r = _to_4x4(c2r)
    np.save(os.path.join(abs_save_path, "C2R.npy"), c2r)

    pose_world = _to_4x4(np.asarray(pose_out["pose_world"], dtype=np.float32))

    # optional per-object 6D pose normalization (same technique the H2R grasp
    # pipeline applies). Selected by --normalize; rolls/canonicalizes the RC
    # object pose so the object orientation is consistent across captures.
    pose_world_raw = pose_world.copy()
    if args.normalize != "none":
        # T_align = identity: the RC 6D pose is estimated against <name>_viser.obj
        # (the viser-aligned mesh), so it is ALREADY in the aligned frame -- no
        # inv(T_align) here (build_scene_assets needs it only because its pose
        # is in the ORIGINAL mesh frame).
        pose_world = _to_4x4(pose_normalize.normalize_object_pose(
            args.normalize, pose_world, c2r, np.eye(4)).astype(np.float32))
        print(f"[normalize] mode={args.normalize} (T_align=identity; RC pose "
              f"is already in the viser.obj frame)")

    r2c = np.linalg.inv(c2r)
    pose_robot = r2c @ pose_world  # same convention as visualize_all.py / capture_rcc.py

    robot_pose_json_path = os.path.join(abs_save_path, "object_6d_robot.json")
    robot_out = {
        "pose_robot": pose_robot.tolist(),
        "R_robot": pose_robot[:3, :3].tolist(),
        "t_robot": pose_robot[:3, 3].tolist(),
        "pose_world": pose_world.tolist(),
        "pose_world_raw": pose_world_raw.tolist(),
        "normalize_mode": args.normalize,
        "mesh_name": args.mesh_name,
        "mesh_path": mesh_path,
    }
    with open(robot_pose_json_path, "w", encoding="utf-8") as f:
        json.dump(robot_out, f, ensure_ascii=False, indent=2)
    print(f"saved: {robot_pose_json_path}")
    print("pose_robot t:", pose_robot[:3, 3].round(4).tolist())

    # 4) visualize robot (qpos=0) + object together in robot frame
    if args.no_vis:
        return

    robot_urdf_path = os.path.expanduser(args.robot_urdf)
    if not os.path.exists(robot_urdf_path):
        raise FileNotFoundError(f"robot urdf not found: {robot_urdf_path}")
    robot_module = RobotModule(robot_urdf_path)

    # Run curobo planning BEFORE creating the viser server so the long planning
    # step doesn't open a half-empty viewer to the user. The pipeline now
    # returns the full trajectory; we hand it to add_traj after viser starts.
    pipeline_result = None
    if args.pipeline:
        if args.scene is None:
            raise ValueError("--pipeline requires --scene <int>")
        pipeline_result = _run_h2r_pipeline(
            args, args.mesh_name, pose_robot, robot_module, abs_save_path,
        )

    mesh = _load_mesh(mesh_path)
    if pipeline_result is not None:
        rest_qpos = pipeline_result["full_traj"][0].astype(np.float32)
    else:
        rest_qpos = np.zeros(robot_module.get_num_joints(), dtype=np.float32)

    # RPC server writes mask overlay PNGs under <abs_save_path>/debug/. Popup
    # works in either viser mode; the in-scene plane variant needs the local
    # ViserViewer instance, so it's only added in the local branch below.
    debug_dir = os.path.join(abs_save_path, "debug")
    if args.debug_image_mode in ("popup", "both"):
        _popup_debug_images(debug_dir)

    if args.viser_addr:
        from paradex.visualization.visualizer.viser_client import ViserClient
        vc = ViserClient(args.viser_addr)
        try:
            vc.ping()
        except Exception as e:
            raise RuntimeError(
                f"viser_server unreachable at {args.viser_addr}: {e}. "
                f"Start it with `python -m paradex.visualization.visualizer.viser_server "
                f"--robot_urdf {robot_urdf_path}` or rerun with --viser_addr ''."
            ) from e
        vc.clear_objects()
        vc.set_object(args.mesh_name, mesh, pose_robot, opacity=1.0)
        vc.set_qpos(rest_qpos, name="robot")
        if pipeline_result is not None:
            vc.set_traj(pipeline_result["full_traj"], name="h2r_pipeline", robot_name="robot")
            print(f"[pipe] pushed {pipeline_result['full_traj'].shape[0]} frames to "
                  f"viser_server; ranges {pipeline_result['range_str']}.")
        else:
            print(f"[viser] pushed object + start qpos to viser_server at {args.viser_addr}")
        vc.close()
    else:
        vis = ViserViewer()
        vis.add_floor(height=0.0)
        vis.add_robot("robot", robot_urdf_path)
        vis.robot_dict["robot"].update_cfg(rest_qpos)
        vis.add_object(args.mesh_name, mesh, pose_robot, opacity=1.0)
        if args.debug_image_mode in ("viser", "both"):
            _add_debug_images_to_gui(
                vis, debug_dir,
                anchor_xyz=(float(args.debug_anchor_x), float(args.debug_anchor_y),
                            float(args.debug_anchor_z)),
                plane_width=float(args.debug_plane_width),
            )
        if pipeline_result is not None:
            vis.add_traj("h2r_pipeline", robot_traj={"robot": pipeline_result["full_traj"]})
            print(f"[pipe] added {pipeline_result['full_traj'].shape[0]} frames to "
                  f"viewer; Playback ranges {pipeline_result['range_str']}.")
        elif args.scene is not None:
            _replay_pregrasp(args, vis, args.mesh_name, pose_robot)
            return
        print("Viser running. Press Ctrl+C to exit.")
        try:
            while True:
                vis.update()
        except KeyboardInterrupt:
            pass

    # After the viser session closes, optionally execute the planned trajectory
    # on the real xarm + inspire_f1. Always confirm first (unless --yes).
    if pipeline_result is not None:
        if args.yes:
            ans = "y"
        else:
            ans = ""
            while ans not in ("y", "n"):
                try:
                    ans = input(
                        f"\n[exec] Execute {pipeline_result['full_traj'].shape[0]}-frame "
                        f"trajectory on REAL robot @ {args.rate_hz} Hz "
                        f"(record_video={args.record_video})? [y/n] "
                    ).strip().lower()
                except EOFError:
                    ans = "n"
                    break
                if ans not in ("y", "n"):
                    print("[exec] please answer 'y' or 'n'")
        if ans == "y":
            _execute_on_real_robot(args, pipeline_result, robot_module, abs_save_path)
        else:
            print("[exec] aborted by user")


if __name__ == "__main__":
    main()
