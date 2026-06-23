"""Full xarm trajectory: home -> frame 0 (MG) -> frame N-1 (per-frame IK).

Phase A (approach): cuRobo MotionGen plan_single_js from a fixed home qpos to
frame-0 IK target, with a longer horizon. Produces a smooth approach trajectory
sampled at a fixed dt.

Phase B (replay): per-frame continuity-biased IK output (already smooth, max
~0.08 rad/step). One sample per input frame.

Stitch [Phase A interpolated] + [Phase B per-frame] into a single qpos array
and visualize with MANO + cameras. MANO/per-frame index is shown as
``frame_idx - len(Phase A)`` and clipped to 0 during the approach.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation as R

sys.path.append(str(Path(__file__).resolve().parents[2]))

from paradex.calibration.utils import load_current_C2R, load_current_camparam
from paradex.visualization.visualizer.viser import ViserViewer

from src.planner.taeksoo_xarm_ik import (
    SOURCE_SERIAL, URDF_PATH, ARM_JOINTS, OPERATOR2MANO_RIGHT,
    _to4x4, _transform_pts, _palm_pose_in_robot, _build_curobo_cfg,
)

HOME_DEG = [-84.9, -25.5, -41.1, 118.8, -1.9, 158.1]


def _solve_ik_smooth(targets_pos, targets_quat, retract_default):
    """Per-frame IK with retract+seed = prev_q (continuity-biased)."""
    import torch
    from curobo.types.base import TensorDeviceType
    from curobo.types.math import Pose
    from curobo.wrap.reacher.ik_solver import IKSolver, IKSolverConfig

    tensor_args = TensorDeviceType(
        device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    )
    cfg = _build_curobo_cfg()
    ik_cfg = IKSolverConfig.load_from_robot_config(
        cfg, None, tensor_args=tensor_args, use_cuda_graph=False, num_seeds=32,
    )
    ik = IKSolver(ik_cfg)
    joint_names = list(ik.kinematics.joint_names) if hasattr(ik, "kinematics") else list(ARM_JOINTS)
    Tn = targets_pos.shape[0]
    n_dof = len(joint_names)
    q_out = np.full((Tn, n_dof), np.nan, dtype=np.float32)
    success = np.zeros(Tn, dtype=bool)
    cur_q = retract_default.copy()
    for t in range(Tn):
        goal = Pose(
            position=tensor_args.to_device(targets_pos[t : t + 1].astype(np.float32)),
            quaternion=tensor_args.to_device(targets_quat[t : t + 1].astype(np.float32)),
        )
        retract_t = tensor_args.to_device(cur_q.reshape(1, -1))
        seed_t = tensor_args.to_device(cur_q.reshape(1, 1, -1))
        res = ik.solve_single(goal, retract_config=retract_t, seed_config=seed_t)
        sol = getattr(getattr(res, "js_solution", None), "position", None)
        if sol is None:
            sol = getattr(res, "solution", None)
        arr = sol.detach().cpu().numpy().reshape(-1, n_dof)
        cur_q = arr[0].astype(np.float32)
        q_out[t] = cur_q
        ok = getattr(res, "success", True)
        success[t] = bool(ok.item()) if hasattr(ok, "item") else bool(ok)
    return q_out, success, joint_names


def _plan_approach(home_q, q_goal, joint_names, plan_dt, approach_tsteps):
    """Return interpolated qpos sequence from home_q to q_goal."""
    import torch
    from curobo.types.base import TensorDeviceType
    from curobo.types.state import JointState
    from curobo.wrap.reacher.motion_gen import (
        MotionGen, MotionGenConfig, MotionGenPlanConfig,
    )

    tensor_args = TensorDeviceType(
        device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    )
    cfg = _build_curobo_cfg()
    mg_cfg = MotionGenConfig.load_from_robot_config(
        cfg, None,
        tensor_args=tensor_args,
        use_cuda_graph=False,
        trajopt_tsteps=int(approach_tsteps),
        trajopt_dt=float(plan_dt),
        js_trajopt_dt=float(plan_dt),
        interpolation_dt=float(plan_dt),
        optimize_dt=False,
        num_trajopt_seeds=4,
        num_ik_seeds=16,
    )
    mg = MotionGen(mg_cfg)
    print(f"[approach] warmup tsteps={approach_tsteps} dt={plan_dt}...")
    mg.warmup(parallel_finetune=False)

    home_state = JointState.from_position(
        tensor_args.to_device(home_q.reshape(1, -1).astype(np.float32)),
        joint_names=joint_names,
    )
    goal_state = JointState.from_position(
        tensor_args.to_device(q_goal.reshape(1, -1).astype(np.float32)),
        joint_names=joint_names,
    )
    plan_cfg = MotionGenPlanConfig(
        max_attempts=8, timeout=10.0, enable_graph=False, enable_opt=True,
    )
    res = mg.plan_single_js(home_state, goal_state, plan_cfg)
    ok_attr = getattr(res, "success", False)
    ok = bool(ok_attr.item()) if hasattr(ok_attr, "item") else bool(ok_attr)
    if not ok:
        raise RuntimeError(f"approach plan failed: status={getattr(res, 'status', None)}")

    traj = res.get_interpolated_plan() if hasattr(res, "get_interpolated_plan") else getattr(res, "optimized_plan", None)
    pos = traj.position.detach().cpu().numpy()
    if pos.ndim == 3:
        pos = pos[0]
    return pos.astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default=str(Path(__file__).parent / "out_taeksoo"))
    ap.add_argument("--side", default="right", choices=["right"])
    ap.add_argument("--home-deg", nargs=6, type=float, default=HOME_DEG)
    ap.add_argument("--approach-tsteps", type=int, default=28)
    ap.add_argument("--plan-dt", type=float, default=0.033)
    ap.add_argument("--show-cameras", action="store_true", default=True)
    ap.add_argument("--no-cameras", dest="show_cameras", action="store_false")
    ap.add_argument("--cam-scale", type=float, default=0.05)
    ap.add_argument("--mano-color", nargs=3, type=int, default=[90, 200, 255])
    args = ap.parse_args()

    out = Path(args.out_dir)
    qpos_hand = np.load(out / f"qpos_{args.side}.npy")
    joints_world = np.load(out / f"joints_world_{args.side}.npy")
    verts_world = np.load(out / f"verts_world_{args.side}.npy")
    faces = np.load(out / f"faces_{args.side}.npy")
    rt_joint_names = json.loads((out / f"joint_names_{args.side}.json").read_text())
    Tn = qpos_hand.shape[0]
    print(f"[full] T={Tn}")

    c2r = _to4x4(load_current_C2R())
    robot_from_world = np.linalg.inv(c2r)
    joints_robot = np.stack([_transform_pts(joints_world[t], robot_from_world) for t in range(Tn)])
    verts_robot = np.stack([_transform_pts(verts_world[t], robot_from_world) for t in range(Tn)])

    targets_pos = np.zeros((Tn, 3), dtype=np.float32)
    targets_quat = np.zeros((Tn, 4), dtype=np.float32)
    for t in range(Tn):
        p, q = _palm_pose_in_robot(joints_robot[t])
        targets_pos[t] = p
        targets_quat[t] = q

    home_q = np.deg2rad(np.asarray(args.home_deg, dtype=np.float32))
    print(f"[full] home_q (rad): {home_q}")

    qpos_replay, ik_ok, joint_names = _solve_ik_smooth(targets_pos, targets_quat, home_q)
    print(f"[full] IK replay: {int(ik_ok.sum())}/{Tn}")

    qpos_approach = _plan_approach(
        home_q, qpos_replay[0], joint_names,
        plan_dt=args.plan_dt, approach_tsteps=args.approach_tsteps,
    )
    print(f"[full] approach trajectory: {qpos_approach.shape[0]} sub-frames")

    qpos_full = np.concatenate([qpos_approach, qpos_replay], axis=0)
    n_approach = qpos_approach.shape[0]
    print(f"[full] total frames: {qpos_full.shape[0]} ({n_approach} approach + {Tn} replay)")
    np.save(out / "qpos_arm_full.npy", qpos_full)
    np.save(out / "qpos_arm_approach.npy", qpos_approach)
    np.save(out / "n_approach.npy", np.array([n_approach]))

    # ---------- visualization ----------
    viewer = ViserViewer(scene_title="taeksoo full")
    server = viewer.server

    if args.show_cameras:
        intrinsic, extrinsic = load_current_camparam()
        for serial, E_3x4 in extrinsic.items():
            E = _to4x4(E_3x4)
            cam_in_robot = robot_from_world @ np.linalg.inv(E)
            K = intrinsic[serial]["intrinsics_undistort"]
            h = float(intrinsic[serial]["height"]); w = float(intrinsic[serial]["width"])
            vfov = 2.0 * np.arctan(h / (2.0 * float(K[1, 1])))
            quat_xyzw = R.from_matrix(cam_in_robot[:3, :3]).as_quat()
            is_src = serial == SOURCE_SERIAL
            server.scene.add_camera_frustum(
                name=f"/cameras/{serial}",
                fov=float(vfov), aspect=float(w / h),
                scale=float(args.cam_scale * (1.4 if is_src else 1.0)),
                color=(255, 80, 80) if is_src else (160, 160, 160),
                position=cam_in_robot[:3, 3], wxyz=quat_xyzw[[3, 0, 1, 2]],
            )

    viewer.add_robot(name="xarm_allegro", urdf_path=URDF_PATH)
    robot = viewer.robot_dict["xarm_allegro"]
    urdf_joint_names = list(robot.urdf.get_joint_names())
    name_to_hand_idx = {n: i for i, n in enumerate(rt_joint_names)}
    name_to_arm_idx = {n: i for i, n in enumerate(joint_names)}

    Ttot = qpos_full.shape[0]
    qpos_full_urdf = np.zeros((Ttot, len(urdf_joint_names)), dtype=np.float64)
    # Hand qpos for approach phase: hold frame-0 hand pose; for replay phase: per-frame.
    hand_traj = np.concatenate(
        [np.tile(qpos_hand[0:1], (n_approach, 1)), qpos_hand], axis=0
    )
    for j, name in enumerate(urdf_joint_names):
        if name in name_to_arm_idx:
            qpos_full_urdf[:, j] = qpos_full[:, name_to_arm_idx[name]]
        elif name in name_to_hand_idx:
            qpos_full_urdf[:, j] = hand_traj[:, name_to_hand_idx[name]]

    viewer.add_traj("taeksoo_full", robot_traj={"xarm_allegro": qpos_full_urdf})

    faces_i32 = np.asarray(faces, dtype=np.int32)
    mano_color = tuple(int(c) for c in args.mano_color)
    state = {"handle": None}

    def _set_mano(t_full: int) -> None:
        # Map full-trajectory index back to per-frame index; clip to 0 during approach.
        t_replay = max(0, t_full - n_approach)
        t_replay = min(t_replay, Tn - 1)
        if state["handle"] is not None:
            state["handle"].remove()
        state["handle"] = server.scene.add_mesh_simple(
            name="/mano/mesh",
            vertices=verts_robot[t_replay].astype(np.float32),
            faces=faces_i32,
            color=mano_color,
            opacity=0.55,
        )

    @viewer.gui_timestep.on_update
    def _(_):
        _set_mano(int(viewer.gui_timestep.value))

    _set_mano(0)
    print("[viz] open URL")
    viewer.start_viewer()


if __name__ == "__main__":
    main()
