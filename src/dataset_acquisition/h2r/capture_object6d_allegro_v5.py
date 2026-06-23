"""
h2r capture + execute for the xarm + allegro_v5 stack.

Mirrors capture_object6d.py (xarm + inspire_f1) but swaps the robot to
xarm_allegro_v5 (6-DOF arm + 16-DOF allegro_v5 right hand). The pipeline
shape, RPC handshake, viser preview, and confirm-then-execute flow are the
same; the things that change are:
    - default URDF / curobo cfg
    - hand command path: AllegroController (radian, 16 joints) instead of
      the inspire_f1 raw-motor publisher
    - no raw-motor conversion table; squeeze is an optional radian offset
"""

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

DEFAULT_ROBOT_URDF = str(PROJECT_ROOT / "rsc" / "robot" / "xarm_allegro_v5.urdf")
DEFAULT_CUROBO_ROBOT_CFG = str(
    PROJECT_ROOT / "rsc" / "curobo" / "content" / "configs" / "robot"
    / "xarm_allegro_v5.yml"
)
# Same xarm home pose used by the taeksoo / inspire_f1 pipeline.
HOME_DEG = [-84.9, -25.5, -41.1, 118.8, -1.9, 158.1]

# 16 allegro_v5 hand joints in URDF order (= AllegroController wire order).
ALLEGRO_HAND_JOINTS = [f"joint_{i}_0" for i in range(16)]
# Thumb abduction joint -- excluded from the squeeze ramp so squeezing does
# not change which direction the thumb points.
ALLEGRO_THUMB_ABDUCTION = "joint_12_0"

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
    return os.path.join(mesh_root_dir, mesh_name, f"{mesh_name}_viser.obj")


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


def _popup_debug_images(debug_dir: str) -> int:
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


def _unwrap_continuous_joints(qpos: np.ndarray, joint_names, target_names=("joint1", "joint4", "joint6")):
    """For joints with ±2π range, IK may pick equivalent solutions that differ
    by 2π between consecutive waypoints (visually = a 360° flip). Snap each
    frame's chosen-joint value to the 2π-equivalent angle closest to the
    previous frame's, eliminating the wrap-around without changing the EE pose.
    """
    out = qpos.copy()
    idx = [joint_names.index(n) for n in target_names if n in joint_names]
    if not idx:
        return out
    for j in idx:
        for t in range(1, out.shape[0]):
            d = out[t, j] - out[t - 1, j]
            k = int(round(d / (2.0 * np.pi)))
            if k != 0:
                out[t, j] -= 2.0 * np.pi * k
    return out


def _solve_ik_chain_curobo(robot_cfg: Dict[str, Any], target_poses: np.ndarray,
                           seed_arm_qpos: np.ndarray, device: str = "cuda:0",
                           num_seeds: int = 4, null_space_weight: float = 5.0):
    """Per-waypoint IK with strong continuity constraint.

    Two changes from the inspire_f1 version that suppress big jumps between
    consecutive waypoints:
      1. cspace.null_space_weight is forced to ``null_space_weight`` so curobo
         actively penalises distance from retract_config (= previous solution).
      2. num_seeds drops to 4 so the solver doesn't wander to a far-away basin
         just because some random seed happens to satisfy the pose threshold.
    If IK fails, the previous solution is reused (full freeze, never a jump).
    """
    sys.path.insert(0, str(PROJECT_ROOT / "curobo" / "src"))
    import copy
    import torch
    from curobo.types.base import TensorDeviceType
    from curobo.types.math import Pose
    from curobo.wrap.reacher.ik_solver import IKSolver, IKSolverConfig
    from object6d.capture_rcc import _rotmat_to_wxyz

    robot_cfg = copy.deepcopy(robot_cfg)
    cspace = robot_cfg["robot_cfg"]["kinematics"]["cspace"]
    n_cs = len(cspace["joint_names"])
    cspace["null_space_weight"] = [float(null_space_weight)] * n_cs

    dev = torch.device(device if ("cuda" in str(device) and torch.cuda.is_available()) else "cpu")
    tensor_args = TensorDeviceType(device=dev)
    ik_cfg = IKSolverConfig.load_from_robot_config(
        robot_cfg, None, tensor_args=tensor_args,
        num_seeds=num_seeds, position_threshold=0.005, rotation_threshold=0.1,
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


def _extract_hand_command(full_qpos: np.ndarray, full_joint_names) -> np.ndarray:
    """URDF rad -> 16 allegro_v5 hand joints in AllegroController wire order
    (joint_0_0 .. joint_15_0). Joints not present in the URDF qpos fall back
    to 0 with a one-time warning."""
    idx = {n: i for i, n in enumerate(full_joint_names)}
    out = np.zeros(16, dtype=np.float64)
    missing = []
    for i, name in enumerate(ALLEGRO_HAND_JOINTS):
        if name in idx:
            out[i] = float(full_qpos[idx[name]])
        else:
            missing.append(name)
    if missing and not getattr(_extract_hand_command, "_warned", False):
        print(f"[exec] WARNING: missing allegro joints in qpos, defaulting to 0: {missing}")
        _extract_hand_command._warned = True
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


def _wait_hand_connected(hand, timeout: float = 5.0) -> bool:
    deadline = time.time() + float(timeout)
    while time.time() < deadline:
        if hand.connection_event.is_set():
            return True
        time.sleep(0.05)
    return False


def _execute_on_real_robot(args, pipeline_result: Dict[str, Any], robot_module: RobotModule,
                          abs_save_path: str) -> None:
    from paradex.io.robot_controller import get_arm, get_hand
    from paradex.calibration.utils import save_current_C2R

    full_traj = np.asarray(pipeline_result["full_traj"], dtype=np.float64)
    Tn = int(full_traj.shape[0])
    full_joint_names = list(robot_module.get_joint_names())

    arm_idx = [full_joint_names.index(f"joint{i}") for i in range(1, 7)]
    arm_traj = full_traj[:, arm_idx]  # (T,6) rad

    # hand: 16 allegro joints in wire order, radian
    hand_cmds = np.zeros((Tn, 16), dtype=np.float64)
    for t in range(Tn):
        hand_cmds[t] = _extract_hand_command(full_traj[t], full_joint_names)

    # Optional grasp squeeze: ramp every hand joint except the thumb abduction
    # by --grasp_squeeze radians AFTER the planned grasp finishes (approach +
    # hand_interp + replay), then hold the squeezed pose for grasp_hold + lift.
    n_a = int(pipeline_result.get("n_approach", 0))
    n_b = int(pipeline_result.get("n_hand_interp", 0))
    n_r = int(pipeline_result.get("n_replay", 0))
    n_release = int(pipeline_result.get("n_release", 0))
    # Release frames sit at the very end of full_traj. They must NOT play
    # before the post-roll video hold, so we stop the main streaming at
    # release_start and stream the release frames separately after the hold.
    release_start = Tn - n_release if n_release > 0 else Tn
    squeeze_start = n_a + n_b + n_r
    squeeze_amt = float(args.grasp_squeeze)
    squeeze_steps = int(args.squeeze_steps)
    do_squeeze = squeeze_amt != 0.0 and 0 < squeeze_start < Tn
    thumb_abd_idx = ALLEGRO_HAND_JOINTS.index(ALLEGRO_THUMB_ABDUCTION)
    squeeze_mask = np.ones(16, dtype=bool)
    squeeze_mask[thumb_abd_idx] = False
    if do_squeeze:
        hand_cmds[squeeze_start:, squeeze_mask] += squeeze_amt
        print(f"[exec] grasp squeeze +{squeeze_amt:.3f} rad over {squeeze_steps} "
              f"frames after frame {squeeze_start} ({ALLEGRO_THUMB_ABDUCTION} excluded)")

    # Per-frame streaming period; replay segment can be slower.
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
    print(f"[exec] hand[0]  rad: {np.round(hand_cmds[0], 3).tolist()}")
    print(f"[exec] hand[-1] rad: {np.round(hand_cmds[-1], 3).tolist()}")

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
    print(f"[exec] init allegro_v5 (hand_side={args.hand_side})...")
    hand = get_hand("allegro_v5", hand_side=args.hand_side)
    if not _wait_hand_connected(hand, timeout=5.0):
        print("[exec] WARNING: allegro_v5 state not yet received; commands may be dropped initially")

    try:
        # Phase 0: home + open hand (frame 0 of trajectory), wait for arm arrival.
        print(f"[exec] commanding HOME deg={list(np.rad2deg(home_rad).round(2))} + hand[0]={np.round(hand_cmds[0], 3).tolist()}")
        xarm.move(home_rad, is_servo=True)
        hand.move(hand_cmds[0])
        ok = _wait_xarm_at_target(xarm, home_rad,
                                  np.deg2rad(float(args.home_tol_deg)),
                                  float(args.home_timeout))
        if not ok:
            print(f"[exec] WARNING: arm did not reach HOME within {args.home_timeout}s")
        if xarm.is_error() or hand.is_error():
            raise RuntimeError("controller in error state before streaming")

        # Phase 1: grasp sequence streaming. Stop before the release frames so
        # the post-roll video hold captures the lifted/held pose.
        grasp_end = squeeze_start if do_squeeze else release_start
        print(f"[exec] streaming grasp sequence: frames 0..{grasp_end - 1} ...")
        t0 = time.perf_counter()
        clock = 0.0
        for t in range(grasp_end):
            xarm.move(arm_traj[t], is_servo=True)
            hand.move(hand_cmds[t])
            if xarm.is_error() or hand.is_error():
                raise RuntimeError(f"controller error at frame {t}")
            clock += periods[t]
            sleep_for = t0 + clock - time.perf_counter()
            if sleep_for > 0:
                time.sleep(sleep_for)

        # Phase 1.5: explicit squeeze ramp -- arm frozen, hand-only.
        if do_squeeze:
            hand_from = hand_cmds[squeeze_start - 1].copy()
            hand_to = hand_cmds[squeeze_start].copy()
            arm_hold = arm_traj[squeeze_start - 1]
            print(f"[exec] squeezing hand over {squeeze_steps} frames ...")
            for s in range(1, squeeze_steps + 1):
                a = float(s) / float(squeeze_steps)
                cmd = (1.0 - a) * hand_from + a * hand_to
                xarm.move(arm_hold, is_servo=True)
                hand.move(cmd)
                if xarm.is_error() or hand.is_error():
                    raise RuntimeError(f"controller error during squeeze step {s}")
                time.sleep(1.0 / float(args.rate_hz))

            # Phase 2: hold + lift with squeezed hand pose (stops before release).
            print(f"[exec] streaming hold + lift: frames {squeeze_start}..{release_start - 1} ...")
            t0 = time.perf_counter()
            clock = 0.0
            for t in range(squeeze_start, release_start):
                xarm.move(arm_traj[t], is_servo=True)
                hand.move(hand_cmds[t])
                if xarm.is_error() or hand.is_error():
                    raise RuntimeError(f"controller error at frame {t}")
                clock += periods[t]
                sleep_for = t0 + clock - time.perf_counter()
                if sleep_for > 0:
                    time.sleep(sleep_for)
        print("[exec] streaming done (pre-release)")
        if cs is not None:
            print("[exec] holding final pose for 10s of post-roll video ...")
            time.sleep(10.0)

        # Phase D: release index/middle/ring. Runs AFTER the post-roll hold
        # so the video shows the object held first, then dropped.
        if n_release > 0:
            print(f"[exec] releasing fingers: frames {release_start}..{Tn - 1} ...")
            t0 = time.perf_counter()
            clock = 0.0
            for t in range(release_start, Tn):
                xarm.move(arm_traj[t], is_servo=True)
                hand.move(hand_cmds[t])
                if xarm.is_error() or hand.is_error():
                    raise RuntimeError(f"controller error at release frame {t}")
                clock += periods[t]
                sleep_for = t0 + clock - time.perf_counter()
                if sleep_for > 0:
                    time.sleep(sleep_for)
            print("[exec] release done")
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
    """Same two-phase pipeline as capture_object6d.py: curobo MotionGen plans
    HOME -> pregrasp, then per-waypoint IK tracks the canonical wrist trajectory
    while finger qpos is replayed from result.npz by JOINT NAME. Result.npz must
    therefore have revolute_joint_names matching allegro_v5 joint names for the
    fingers to move; otherwise the hand is held at zeros."""
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

    res_path = os.path.join(args.result_root, mesh_name, str(args.scene), "result_allegro.npz")
    if not os.path.exists(res_path):
        raise FileNotFoundError(f"no packed result: {res_path}")
    d = np.load(res_path, allow_pickle=True)
    # result_allegro.npz layout:
    #   <src>_traj_q_all   : (8, W, 22) = particles x waypoints x (wrist[6] + 16 allegro fingers)
    #   <src>_traj_q       : (W, 22)    = the best-particle trajectory (already selected)
    #   <src>_best_particle: scalar     = best particle index
    # The finger joint order is the URDF order: joint_0_0 .. joint_15_0.
    key_all = f"{args.src}_traj_q_all"
    key_best = f"{args.src}_traj_q"
    if args.particle is None:
        if key_best not in d.files:
            raise KeyError(f"{key_best} missing in {res_path}")
        traj_canon = np.asarray(d[key_best])
        particle_used = (int(d[f"{args.src}_best_particle"])
                         if f"{args.src}_best_particle" in d.files else -1)
    else:
        if key_all not in d.files:
            raise KeyError(f"{key_all} missing in {res_path}")
        traj_canon = np.asarray(d[key_all])[int(args.particle)]
        particle_used = int(args.particle)
    jnames_traj = list(ALLEGRO_HAND_JOINTS)  # 16 finger joints in URDF order
    W = int(traj_canon.shape[0])
    n_finger_traj = traj_canon.shape[1] - 6
    print(f"[pipe] {mesh_name} scene={args.scene} src={args.src} "
          f"particle={particle_used}  W={W} (wrist + {n_finger_traj} finger joints)")
    if n_finger_traj != len(jnames_traj):
        raise ValueError(f"result_allegro.npz finger DOF ({n_finger_traj}) != "
                         f"expected 16 allegro joints")

    wrist_T_robot = np.zeros((W, 4, 4), dtype=np.float32)
    for i, w in enumerate(traj_canon):
        Wc = np.eye(4, dtype=np.float32)
        Wc[:3, :3] = Rotation.from_euler("XYZ", w[3:6]).as_matrix()
        Wc[:3, 3] = w[:3]
        wrist_T_robot[i] = (pose_robot @ Wc).astype(np.float32)

    full_joint_names = list(robot_module.get_joint_names())
    n_dof = int(robot_module.get_num_joints())
    name_to_full_idx = {n: i for i, n in enumerate(full_joint_names)}
    name_to_finger_idx = {n: i for i, n in enumerate(jnames_traj)}
    home_qpos_full = np.zeros(n_dof, dtype=np.float32)
    home_qpos_full[:6] = np.deg2rad(np.asarray(args.home_deg, dtype=np.float32))
    matched_finger = 0
    for jn, fi in name_to_finger_idx.items():
        if jn in name_to_full_idx:
            home_qpos_full[name_to_full_idx[jn]] = float(traj_canon[0, 6 + fi])
            matched_finger += 1
    print(f"[pipe] result.npz -> URDF finger mapping matched {matched_finger}/{len(jnames_traj)} joints")

    cfg_path = os.path.expanduser(args.curobo_robot_cfg)
    if not os.path.exists(cfg_path):
        raise FileNotFoundError(f"curobo cfg not found: {cfg_path}")
    robot_cfg = _normalize_curobo_robot_cfg_paths_with_cfg_file(load_yaml(cfg_path), cfg_path)
    robot_cfg = _normalize_curobo_robot_cfg_paths(robot_cfg)
    # Pre-shape the locked hand joints to the FIRST replay finger pose so the
    # arm carries the actual grasp pose through the approach.
    lock_joints = robot_cfg["robot_cfg"]["kinematics"].get("lock_joints")
    if isinstance(lock_joints, dict):
        for jn in lock_joints:
            if jn in name_to_full_idx:
                lock_joints[jn] = float(home_qpos_full[name_to_full_idx[jn]])

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
    approach_active = np.asarray(plan["trajectory"], dtype=np.float32)
    active_jn = list(plan.get("active_joint_names", []) or cspace_jn)
    Na = int(approach_active.shape[0])
    print(f"[pipe] Phase A done: {Na} waypoints, active_dof={approach_active.shape[1]}")

    print(f"[pipe] Phase B: per-waypoint IK across {W} waypoints "
          f"(num_seeds={args.ik_num_seeds}, null_space_weight={args.ik_null_space_weight})")
    replay_active = _solve_ik_chain_curobo(
        robot_cfg=robot_cfg,
        target_poses=wrist_T_robot,
        seed_arm_qpos=approach_active[-1],
        device=curobo_device,
        num_seeds=int(args.ik_num_seeds),
        null_space_weight=float(args.ik_null_space_weight),
    )
    active_jn_tmp = list(plan.get("active_joint_names", []) or cspace_jn)
    # Include the Phase A endpoint so Phase B's first frame is unwrapped
    # relative to it (avoids a 2π swing across the bridge frames).
    _stitched = np.concatenate([approach_active[-1:], replay_active], axis=0)
    replay_active = _unwrap_continuous_joints(_stitched, active_jn_tmp)[1:]

    name_to_active_idx = {n: i for i, n in enumerate(active_jn)}

    approach_full = np.tile(home_qpos_full, (Na, 1))
    for jn, ai in name_to_active_idx.items():
        if jn in name_to_full_idx:
            approach_full[:, name_to_full_idx[jn]] = approach_active[:, ai]

    replay_full = np.tile(approach_full[-1], (W, 1))
    for jn, ai in name_to_active_idx.items():
        if jn in name_to_full_idx:
            replay_full[:, name_to_full_idx[jn]] = replay_active[:, ai]
    for jn, fi in name_to_finger_idx.items():
        if jn in name_to_full_idx:
            replay_full[:, name_to_full_idx[jn]] = traj_canon[:, 6 + fi].astype(np.float32)

    nh = int(max(0, args.hand_interp_steps))
    bridge_full = np.zeros((0, n_dof), dtype=np.float32)
    q_a_end = approach_full[-1]
    q_b_start = replay_full[0]
    if nh > 0 and not np.allclose(q_a_end, q_b_start, atol=1e-6):
        if nh >= 2:
            alphas = np.linspace(0.0, 1.0, nh + 1, dtype=np.float32)[1:-1]
        else:
            alphas = np.array([0.5], dtype=np.float32)
        bridge_full = (1.0 - alphas[:, None]) * q_a_end[None, :] + alphas[:, None] * q_b_start[None, :]
    Nb = int(bridge_full.shape[0])

    Nh = int(max(0, args.grasp_hold_frames))
    hold_full = np.tile(replay_full[-1], (Nh, 1)) if Nh > 0 else np.zeros((0, n_dof), dtype=np.float32)

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

    # ---- Phase D: release -- open index/middle/ring after lift so the
    # object drops. Thumb stays closed; arm stays at lift end-pose.
    Nd = int(max(0, args.release_steps))
    release_full = np.zeros((0, n_dof), dtype=np.float32)
    if Nd > 0:
        last = lift_full[-1] if Nc > 0 else (hold_full[-1] if Nh > 0 else replay_full[-1])
        target = last.copy()
        # joint_0_0 .. joint_11_0 = index + middle + ring (12 joints). Thumb
        # (joint_12_0..joint_15_0) excluded so it can keep supporting until
        # the other fingers have cleared the object.
        release_joints = [f"joint_{i}_0" for i in range(12)]
        for jn in release_joints:
            if jn in name_to_full_idx:
                target[name_to_full_idx[jn]] = 0.0
        if Nd >= 2:
            alphas = np.linspace(0.0, 1.0, Nd + 1, dtype=np.float32)[1:]
        else:
            alphas = np.array([1.0], dtype=np.float32)
        release_full = (1.0 - alphas[:, None]) * last[None, :] + alphas[:, None] * target[None, :]
        print(f"[pipe] Phase D: release index/middle/ring over {Nd} frames")

    full_traj = np.concatenate(
        [approach_full, bridge_full, replay_full, hold_full, lift_full, release_full], axis=0)

    np.save(os.path.join(abs_save_path, "h2r_pipeline_traj.npy"), full_traj)
    with open(os.path.join(abs_save_path, "h2r_pipeline_meta.json"), "w") as f:
        json.dump({"n_approach": Na, "n_hand_interp": Nb, "n_replay": W,
                   "n_grasp_hold": Nh, "n_lift": Nc, "n_release": Nd,
                   "lift_height_m": lift_h,
                   "joint_names": full_joint_names,
                   "active_joint_names": active_jn,
                   "home_deg": list(args.home_deg),
                   "hand_interp_steps_requested": int(args.hand_interp_steps),
                   "grasp_hold_frames_requested": int(args.grasp_hold_frames),
                   "release_steps_requested": int(args.release_steps)}, f, indent=2)
    print(f"[pipe] saved h2r_pipeline_traj.npy shape={full_traj.shape}")

    cursor = 0
    ranges = []
    for name, n in [("approach", Na), ("hand_interp", Nb), ("replay", W),
                    ("grasp_hold", Nh), ("lift", Nc), ("release", Nd)]:
        if n > 0:
            ranges.append((name, cursor, cursor + n - 1))
            cursor += n
    range_str = ", ".join(f"{n}[{lo}..{hi}]" for n, lo, hi in ranges)
    print(f"[pipe] planning done; trajectory shape={full_traj.shape}, "
          f"ranges {range_str}.")
    return {"full_traj": full_traj, "range_str": range_str,
            "n_approach": Na, "n_hand_interp": Nb, "n_replay": W,
            "n_grasp_hold": Nh, "n_lift": Nc, "n_release": Nd}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", "--mesh_name", dest="mesh_name", required=True)
    parser.add_argument("--save_path", default="capture/test_allegro",
                        help="relative path under shared_data")
    parser.add_argument("--rpc_addr", default="tcp://192.168.0.14:5570")
    parser.add_argument("--rpc_timeout_ms", type=int, default=300000)
    parser.add_argument("--rcc_entry", default="image_main.py")
    parser.add_argument("--mesh_root_dir", default=os.path.join(shared_dir, "mesh_blender"))
    parser.add_argument("--c2r_path", default=None, help="optional C2R.npy override")
    parser.add_argument("--robot_urdf", default=DEFAULT_ROBOT_URDF)
    parser.add_argument("--no_vis", action="store_true")
    parser.add_argument("--viser_addr", default="tcp://localhost:5572",
                        help="ZMQ REQ address of a running viser_server. "
                             "Empty string => fall back to launching a local "
                             "ViserViewer (blocking).")
    parser.add_argument("--debug_image_mode", default="popup",
                        choices=["popup", "none"],
                        help="how to show RPC mask overlay PNGs: popup = OS "
                             "image viewer window(s); none")
    parser.add_argument(
        "--normalize", default="none", choices=pose_normalize.MODES,
        help="per-object 6D pose normalization applied to the RC object pose")
    # --- pre-grasp -> grasp motion replay (our H2R result) ---
    parser.add_argument("--scene", type=int, default=None,
                        help="capture scene/clip number; enables pre-grasp "
                             "motion replay for that clip's stored result")
    parser.add_argument("--src", default="robot", choices=("robot", "human"),
                        help="which contact source's grasp to replay")
    parser.add_argument("--particle", type=int, default=None,
                        help="particle index 0..7. Default = use <src>_traj_q "
                             "(already-selected best particle). Override to "
                             "pick a specific particle from <src>_traj_q_all.")
    parser.add_argument(
        "--result_root",
        default=os.path.join(shared_dir, "capture", "h2r", "0518"),
        help="root holding <obj>/<scene>/result.npz (pack_results.py output)")
    parser.add_argument("--pipeline", action="store_true",
                        help="run the curobo h2r pipeline; requires --scene")
    parser.add_argument("--home_deg", nargs=6, type=float, default=HOME_DEG,
                        help="xarm home joint angles in degrees (6 values)")
    parser.add_argument("--curobo_robot_cfg", default=DEFAULT_CUROBO_ROBOT_CFG,
                        help="curobo robot config yaml (xarm_allegro_v5)")
    parser.add_argument("--curobo_device", default="cuda:1")
    parser.add_argument("--curobo_seeds", type=int, default=128)
    parser.add_argument("--ik_num_seeds", type=int, default=8,
                        help="curobo IK num_seeds for the per-waypoint chain. "
                             "Lower => stays close to previous solution.")
    parser.add_argument("--ik_null_space_weight", type=float, default=1.0,
                        help="penalty on distance from retract_config (= last "
                             "IK solution). Higher => stronger continuity but "
                             "can fight target tracking; lower if arm starts "
                             "drifting sideways instead of toward the target.")
    parser.add_argument("--table_size_xyz", default="2.4,1.6,0.1")
    parser.add_argument("--table_pos_xyz", default="0.0,0.0,-0.05")
    parser.add_argument("--hand_interp_steps", type=int, default=60,
                        help="bridge frames between Phase A end (arm at pregrasp, "
                             "hand open) and Phase B start (hand at first replay "
                             "finger qpos). Arm is held; fingers ramp.")
    parser.add_argument("--grasp_hold_frames", type=int, default=30,
                        help="dwell frames after grasp replay (arm + hand still)")
    parser.add_argument("--grasp_squeeze", type=float, default=0.0,
                        help="add this many radians to every hand joint except "
                             "joint_12_0 (thumb abduction) AFTER the grasp "
                             "sequence finishes, ramped over --squeeze_steps. "
                             "0 disables.")
    parser.add_argument("--squeeze_steps", type=int, default=20)
    parser.add_argument("--lift_height_m", type=float, default=0.12)
    parser.add_argument("--release_steps", type=int, default=30,
                        help="after lift, open index/middle/ring (joint_0_0.."
                             "joint_11_0) to 0 rad over this many frames so "
                             "the object is released. thumb (joint_12_0.._15_0) "
                             "and arm stay still. 0 disables.")
    # --- real-robot execution ---
    parser.add_argument("--rate_hz", type=float, default=30.0)
    parser.add_argument("--replay_rate_hz", type=float, default=15.0)
    parser.add_argument("--home_tol_deg", type=float, default=2.0)
    parser.add_argument("--home_timeout", type=float, default=20.0)
    parser.add_argument("--hand_side", default="right", choices=["right", "left"])
    parser.add_argument("--record_video", action="store_true", default=True)
    parser.add_argument("--no_record_video", dest="record_video", action="store_false")
    parser.add_argument("--realsense", action="store_true")
    parser.add_argument("--yes", action="store_true",
                        help="skip the post-viser execute confirmation prompt")
    args = parser.parse_args()

    mesh_path = _resolve_mesh_path(args.mesh_name, os.path.expanduser(args.mesh_root_dir))
    if not os.path.exists(mesh_path):
        raise FileNotFoundError(f"mesh not found: {mesh_path}")

    date_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
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
    pose_world_raw = pose_world.copy()
    if args.normalize != "none":
        pose_world = _to_4x4(pose_normalize.normalize_object_pose(
            args.normalize, pose_world, c2r, np.eye(4)).astype(np.float32))
        print(f"[normalize] mode={args.normalize}")

    r2c = np.linalg.inv(c2r)
    pose_robot = r2c @ pose_world

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

    if args.no_vis:
        return

    robot_urdf_path = os.path.expanduser(args.robot_urdf)
    if not os.path.exists(robot_urdf_path):
        raise FileNotFoundError(f"robot urdf not found: {robot_urdf_path}")
    robot_module = RobotModule(robot_urdf_path)

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

    debug_dir = os.path.join(abs_save_path, "debug")
    if args.debug_image_mode == "popup":
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
                f"--robot_urdf {robot_urdf_path}` or rerun with --viser_addr ''.") from e
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
        if pipeline_result is not None:
            vis.add_traj("h2r_pipeline", robot_traj={"robot": pipeline_result["full_traj"]})
            print(f"[pipe] added {pipeline_result['full_traj'].shape[0]} frames to "
                  f"viewer; Playback ranges {pipeline_result['range_str']}.")
        print("Viser running. Press Ctrl+C to exit.")
        try:
            while True:
                vis.update()
        except KeyboardInterrupt:
            pass

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
