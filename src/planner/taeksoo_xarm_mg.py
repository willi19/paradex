"""Per-frame xarm MotionGen (short trajopt) variant of taeksoo_xarm_ik.py.

Runs cuRobo MotionGen between consecutive frames with a small
``trajopt_tsteps`` to enforce kinematic continuity. Take the final state of
each plan as the qpos for that frame; seed the next plan with it. If a plan
fails for some frame, fall back to seed (= previous qpos).

Defaults to trajopt_tsteps=1 per the experiment requested.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation as R

sys.path.append(str(Path(__file__).resolve().parents[2]))

from paradex.calibration.utils import load_current_C2R, load_current_camparam
from paradex.visualization.visualizer.viser import ViserViewer

from src.planner.taeksoo_xarm_ik import (
    SOURCE_SERIAL,
    URDF_PATH,
    ARM_JOINTS,
    OPERATOR2MANO_RIGHT,
    _to4x4,
    _transform_pts,
    _palm_pose_in_robot,
    _build_curobo_cfg,
)


def _solve_motiongen(
    targets_pos: np.ndarray,
    targets_quat: np.ndarray,
    trajopt_tsteps: int,
    plan_dt: float,
):
    """Two-stage: (1) per-frame pose-IK to get q* targets,
    (2) plan_single_js between consecutive q's for smooth trajectory."""
    import torch
    from curobo.types.base import TensorDeviceType
    from curobo.types.math import Pose
    from curobo.types.state import JointState
    from curobo.wrap.reacher.ik_solver import IKSolver, IKSolverConfig
    from curobo.wrap.reacher.motion_gen import (
        MotionGen, MotionGenConfig, MotionGenPlanConfig,
    )

    tensor_args = TensorDeviceType(
        device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    )
    cfg = _build_curobo_cfg()

    # ---- (1) IK per frame ----
    ik_cfg = IKSolverConfig.load_from_robot_config(
        cfg, None, tensor_args=tensor_args, use_cuda_graph=False, num_seeds=32,
    )
    ik_solver = IKSolver(ik_cfg)
    joint_names = (
        list(ik_solver.kinematics.joint_names)
        if hasattr(ik_solver, "kinematics") and hasattr(ik_solver.kinematics, "joint_names")
        else list(ARM_JOINTS)
    )
    print(f"[mg] joint names: {joint_names}")

    Tn = targets_pos.shape[0]
    n_dof = len(joint_names)
    q_target = np.zeros((Tn, n_dof), dtype=np.float32)
    ik_ok = np.zeros(Tn, dtype=bool)
    retract_default = np.asarray(
        cfg["robot_cfg"]["kinematics"]["cspace"]["retract_config"], dtype=np.float32
    )
    cur_q = retract_default.copy()
    for t in range(Tn):
        goal = Pose(
            position=tensor_args.to_device(targets_pos[t : t + 1].astype(np.float32)),
            quaternion=tensor_args.to_device(targets_quat[t : t + 1].astype(np.float32)),
        )
        # Bias toward prev_q via retract_config (null-space cost) + seed_config.
        retract_t = tensor_args.to_device(cur_q.reshape(1, -1))
        seed_t = tensor_args.to_device(cur_q.reshape(1, 1, -1))
        res = ik_solver.solve_single(goal, retract_config=retract_t, seed_config=seed_t)
        sol = getattr(getattr(res, "js_solution", None), "position", None)
        if sol is None:
            sol = getattr(res, "solution", None)
        if sol is None:
            q_target[t] = cur_q
            continue
        arr = sol.detach().cpu().numpy().reshape(-1, n_dof)
        cur_q = arr[0].astype(np.float32)
        q_target[t] = cur_q
        ik_ok_t = getattr(res, "success", True)
        ik_ok[t] = bool(ik_ok_t.item()) if hasattr(ik_ok_t, "item") else bool(ik_ok_t)
    print(f"[mg] IK targets: {int(ik_ok.sum())}/{Tn}")
    del ik_solver

    # ---- (2) JS trajopt between consecutive q's ----
    mg_cfg = MotionGenConfig.load_from_robot_config(
        cfg, None,
        tensor_args=tensor_args,
        use_cuda_graph=False,
        trajopt_tsteps=int(trajopt_tsteps),
        trajopt_dt=float(plan_dt),
        js_trajopt_dt=float(plan_dt),
        interpolation_dt=float(plan_dt),
        optimize_dt=False,
        num_trajopt_seeds=4,
        num_ik_seeds=16,
    )
    mg = MotionGen(mg_cfg)
    print(f"[mg] warming up (trajopt_tsteps={trajopt_tsteps}, dt={plan_dt})...")
    mg.warmup(parallel_finetune=False)

    q_out = np.full((Tn, n_dof), np.nan, dtype=np.float32)
    success = np.zeros(Tn, dtype=bool)
    q_out[0] = q_target[0]
    success[0] = True
    cur_q = q_target[0].copy()

    plan_cfg = MotionGenPlanConfig(
        max_attempts=4, timeout=2.0, enable_graph=False, enable_opt=True,
    )

    for t in range(1, Tn):
        cur_state = JointState.from_position(
            tensor_args.to_device(cur_q.reshape(1, -1)), joint_names=joint_names,
        )
        goal_state = JointState.from_position(
            tensor_args.to_device(q_target[t].reshape(1, -1)), joint_names=joint_names,
        )
        res = mg.plan_single_js(cur_state, goal_state, plan_cfg)
        ok_attr = getattr(res, "success", False)
        ok = bool(ok_attr.item()) if hasattr(ok_attr, "item") else bool(ok_attr)
        if not ok:
            status = getattr(res, "status", None)
            if t < 5 or t % 20 == 0:
                print(f"[mg] frame {t}: plan_js failed (status={status})")
            q_out[t] = cur_q
            continue

        traj = getattr(res, "optimized_plan", None)
        if traj is None and hasattr(res, "get_interpolated_plan"):
            traj = res.get_interpolated_plan()
        if traj is None or not hasattr(traj, "position"):
            q_out[t] = cur_q
            continue

        pos = traj.position.detach().cpu().numpy()
        if pos.ndim == 3:
            pos = pos[0]
        last = pos[-1].astype(np.float32)
        q_out[t] = last
        success[t] = True
        cur_q = last

    return q_out, success, joint_names


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default=str(Path(__file__).parent / "out_taeksoo"))
    ap.add_argument("--side", default="right", choices=["right"])
    ap.add_argument("--show-cameras", action="store_true", default=True)
    ap.add_argument("--no-cameras", dest="show_cameras", action="store_false")
    ap.add_argument("--cam-scale", type=float, default=0.05)
    ap.add_argument("--mano-color", nargs=3, type=int, default=[90, 200, 255])
    ap.add_argument("--trajopt-tsteps", type=int, default=1)
    ap.add_argument("--plan-dt", type=float, default=0.033)
    args = ap.parse_args()

    out = Path(args.out_dir)
    qpos_hand = np.load(out / f"qpos_{args.side}.npy")
    joints_world = np.load(out / f"joints_world_{args.side}.npy")
    verts_world = np.load(out / f"verts_world_{args.side}.npy")
    faces = np.load(out / f"faces_{args.side}.npy")
    rt_joint_names = json.loads((out / f"joint_names_{args.side}.json").read_text())
    Tn = qpos_hand.shape[0]
    print(f"[mg] T={Tn}")

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

    qpos_arm, success, joint_names = _solve_motiongen(
        targets_pos, targets_quat,
        trajopt_tsteps=args.trajopt_tsteps,
        plan_dt=args.plan_dt,
    )
    print(f"[mg] solved {int(success.sum())}/{Tn}")
    np.save(out / f"qpos_arm_mg_ts{args.trajopt_tsteps}.npy", qpos_arm)
    np.save(out / f"mg_success_ts{args.trajopt_tsteps}.npy", success)

    # ---------- visualization ----------
    viewer = ViserViewer(scene_title=f"taeksoo MG ts={args.trajopt_tsteps}")
    server = viewer.server

    if args.show_cameras:
        intrinsic, extrinsic = load_current_camparam()
        for serial, E_3x4 in extrinsic.items():
            E = _to4x4(E_3x4)
            cam_in_world = np.linalg.inv(E)
            cam_in_robot = robot_from_world @ cam_in_world
            K = intrinsic[serial]["intrinsics_undistort"]
            h = float(intrinsic[serial]["height"])
            w = float(intrinsic[serial]["width"])
            vfov = 2.0 * np.arctan(h / (2.0 * float(K[1, 1])))
            quat_xyzw = R.from_matrix(cam_in_robot[:3, :3]).as_quat()
            is_src = serial == SOURCE_SERIAL
            server.scene.add_camera_frustum(
                name=f"/cameras/{serial}",
                fov=float(vfov),
                aspect=float(w / h),
                scale=float(args.cam_scale * (1.4 if is_src else 1.0)),
                color=(255, 80, 80) if is_src else (160, 160, 160),
                position=cam_in_robot[:3, 3],
                wxyz=quat_xyzw[[3, 0, 1, 2]],
            )
        print(f"[viz] {len(extrinsic)} cameras")

    viewer.add_robot(name="xarm_allegro", urdf_path=URDF_PATH)
    robot = viewer.robot_dict["xarm_allegro"]
    urdf_joint_names = list(robot.urdf.get_joint_names())

    name_to_hand_idx = {n: i for i, n in enumerate(rt_joint_names)}
    name_to_arm_idx = {n: i for i, n in enumerate(joint_names)}
    qpos_full = np.zeros((Tn, len(urdf_joint_names)), dtype=np.float64)
    for j, name in enumerate(urdf_joint_names):
        if name in name_to_arm_idx:
            qpos_full[:, j] = qpos_arm[:, name_to_arm_idx[name]]
        elif name in name_to_hand_idx:
            qpos_full[:, j] = qpos_hand[:, name_to_hand_idx[name]]

    viewer.add_traj("taeksoo_mg", robot_traj={"xarm_allegro": qpos_full})

    faces_i32 = np.asarray(faces, dtype=np.int32)
    mano_color = tuple(int(c) for c in args.mano_color)
    state = {"handle": None}

    def _set_mano(t: int) -> None:
        if state["handle"] is not None:
            state["handle"].remove()
        state["handle"] = server.scene.add_mesh_simple(
            name="/mano/mesh",
            vertices=verts_robot[t].astype(np.float32),
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
