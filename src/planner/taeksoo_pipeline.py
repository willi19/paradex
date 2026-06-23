"""End-to-end pipeline: MANO predictions -> allegro retarget -> xarm IK +
approach -> viser visualization.

Equivalent to running:
    taeksoo.py            # MANO -> allegro_v5 vector retarget
    taeksoo_xarm_full.py  # C2R -> per-frame IK + approach -> viz

But in a single process, reusing helpers from each module.
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

from src.planner.taeksoo import (
    SOURCE_SERIAL as MANO_SOURCE_SERIAL,
    ALLEGRO_V5_DIR,
    _VECTOR_TEMPLATE,
    _POSITION_TEMPLATE,
    _DEFAULT_VECTOR_SCALING,
    _DEFAULT_LOW_PASS_ALPHA,
    _e_inv_4x4,
    _transform_pts as _transform_pts_taeksoo,
    _forward_mano,
    _estimate_mano_wrist_frame as _estimate_wrist_taeksoo,
    OPERATOR2MANO_RIGHT,
)
from src.planner.taeksoo_xarm_ik import (
    URDF_PATH,
    ARM_JOINTS,
    _to4x4,
    _transform_pts,
    _palm_pose_in_robot,
    _build_curobo_cfg,
)
from src.planner.taeksoo_xarm_full import (
    HOME_DEG,
    _solve_ik_smooth,
    _plan_approach,
)


def _retarget_hand(joints_cam0: np.ndarray, valid: np.ndarray, retarget_type: str,
                   work_dir: Path, scaling_factor: float = _DEFAULT_VECTOR_SCALING,
                   low_pass_alpha: float = _DEFAULT_LOW_PASS_ALPHA,
                   n_warmup: int = 0):
    """Run dex-retargeting per-frame on (T,21,3) MANO joints (cam0 frame).

    If ``n_warmup > 0``, prepends that many extra retarget() calls using the
    first valid frame's reference. Since the retargeter starts at last_qpos=0
    (fully open), this lets the built-in low-pass filter ramp the qpos from
    open to the first real pose. Returns (qpos[n_warmup+T_valid_lead,16], ...)
    of length ``n_warmup + T``.
    """
    from dex_retargeting.retargeting_config import RetargetingConfig

    if retarget_type == "vector":
        yaml_str = _VECTOR_TEMPLATE.format(
            scaling_factor=float(scaling_factor),
            low_pass_alpha=float(low_pass_alpha),
        )
    else:
        yaml_str = _POSITION_TEMPLATE.format(low_pass_alpha=float(low_pass_alpha))
    RetargetingConfig.set_default_urdf_dir(str(ALLEGRO_V5_DIR))
    yml_path = work_dir / f"allegro_v5_right_{retarget_type}.yml"
    yml_path.write_text(yaml_str)
    rt = RetargetingConfig.load_from_file(str(yml_path)).build()
    joint_names = list(rt.optimizer.robot.dof_joint_names)
    # Force optimizer seed AND low-pass filter state to fully-open (zeros) so
    # the saved trajectory begins at qpos=0 and ramps up via the filter on
    # subsequent calls. Without this both default to mid-of-limits.
    rt.last_qpos = np.zeros_like(rt.last_qpos)
    if rt.filter is not None:
        rt.filter.y = np.zeros(rt.optimizer.robot.dof, dtype=np.float32)
        rt.filter.is_init = True

    indices = rt.optimizer.target_link_human_indices
    rtype = rt.optimizer.retargeting_type
    n_fixed = len(rt.optimizer.idx_pin2fixed)
    fixed_qpos = np.zeros((n_fixed,), dtype=np.float32)

    def _ref_at(t):
        kp3 = joints_cam0[t].astype(np.float32)
        kp3 = kp3 - kp3[0:1, :]
        wrist_rot = _estimate_wrist_taeksoo(kp3)
        joint_pos = kp3 @ wrist_rot @ OPERATOR2MANO_RIGHT.astype(np.float32)
        if rtype == "POSITION":
            return joint_pos[indices, :]
        return joint_pos[indices[1, :], :] - joint_pos[indices[0, :], :]

    Tn = joints_cam0.shape[0]
    n_dof = len(joint_names)
    n_warmup = int(max(0, n_warmup))
    qpos = np.full((n_warmup + Tn, n_dof), np.nan, dtype=np.float32)

    # Warmup: feed the first valid frame's reference n_warmup times. The
    # retargeter starts at last_qpos = 0 (fully open) and the low-pass filter
    # ramps it toward the target.
    first_valid = next((t for t in range(Tn) if bool(valid[t])), None)
    if n_warmup > 0 and first_valid is not None:
        ref0 = _ref_at(first_valid)
        for k in range(n_warmup):
            q = rt.retarget(ref0, fixed_qpos=fixed_qpos)
            qpos[k] = np.asarray(q, dtype=np.float32)

    last = qpos[n_warmup - 1] if n_warmup > 0 else None
    for t in range(Tn):
        if not bool(valid[t]):
            if last is not None:
                qpos[n_warmup + t] = last
            continue
        ref = _ref_at(t)
        q = rt.retarget(ref, fixed_qpos=fixed_qpos)
        last = np.asarray(q, dtype=np.float32)
        qpos[n_warmup + t] = last
    return qpos, joint_names


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bundle", default=str(Path(__file__).parent / "collab_bundle" / "predictions.npz"))
    ap.add_argument("--out-dir", default=str(Path(__file__).parent / "out_taeksoo"))
    ap.add_argument("--mano-model-dir", default="/home/temp_id/hamer-mediapipe/_DATA/data/mano")
    ap.add_argument("--source-serial", default=MANO_SOURCE_SERIAL)
    ap.add_argument("--retarget-type", default="vector", choices=["vector", "position"])
    ap.add_argument("--scaling-factor", type=float, default=_DEFAULT_VECTOR_SCALING,
                    help="Vector retargeting human-vector scale. "
                    "Higher -> fingers extend more (less curl).")
    ap.add_argument("--low-pass-alpha", type=float, default=_DEFAULT_LOW_PASS_ALPHA,
                    help="dex-retargeting low-pass alpha. "
                    "Smaller -> stronger filter / slower response (e.g. 0.05).")
    ap.add_argument("--home-deg", nargs=6, type=float, default=HOME_DEG)
    ap.add_argument("--approach-tsteps", type=int, default=28)
    ap.add_argument("--plan-dt", type=float, default=0.033)
    ap.add_argument("--show-cameras", action="store_true", default=True)
    ap.add_argument("--no-cameras", dest="show_cameras", action="store_false")
    ap.add_argument("--cam-scale", type=float, default=0.05)
    ap.add_argument("--mano-color", nargs=3, type=int, default=[90, 200, 255])
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    side = "right"

    # ---- 1. MANO forward in cam0 frame ----
    bundle = dict(np.load(args.bundle, allow_pickle=True))
    Tn = int(bundle[f"{side}_trans"].shape[0])
    valid_key = f"pred_valid_{side}" if f"pred_valid_{side}" in bundle else f"{side}_valid"
    valid = np.asarray(bundle[valid_key], dtype=bool)
    print(f"[pipe] T={Tn} side={side}")

    joints_cam0, verts_cam0, faces = _forward_mano(bundle, side, args.mano_model_dir, Tn)

    # ---- 2. cam0 -> calib world ----
    intrinsic, extrinsic = load_current_camparam()
    if args.source_serial not in extrinsic:
        raise KeyError(f"source serial {args.source_serial} not in current calibration")
    E_world_from_cam0 = _e_inv_4x4(extrinsic[args.source_serial])
    joints_world = np.stack([_transform_pts_taeksoo(joints_cam0[t], E_world_from_cam0) for t in range(Tn)])
    verts_world = np.stack([_transform_pts_taeksoo(verts_cam0[t], E_world_from_cam0) for t in range(Tn)])

    # ---- 3. world -> xarm base ----
    c2r = _to4x4(load_current_C2R())
    robot_from_world = np.linalg.inv(c2r)
    joints_robot = np.stack([_transform_pts(joints_world[t], robot_from_world) for t in range(Tn)])
    verts_robot = np.stack([_transform_pts(verts_world[t], robot_from_world) for t in range(Tn)])

    # ---- 4. per-frame IK targets (palm_link in xarm base frame) ----
    targets_pos = np.zeros((Tn, 3), dtype=np.float32)
    targets_quat = np.zeros((Tn, 4), dtype=np.float32)
    for t in range(Tn):
        p, q = _palm_pose_in_robot(joints_robot[t])
        targets_pos[t] = p
        targets_quat[t] = q

    # ---- 5. continuity-biased IK + MG approach ----
    home_q = np.deg2rad(np.asarray(args.home_deg, dtype=np.float32))
    print(f"[pipe] home_q (rad): {home_q}")
    qpos_replay, ik_ok, joint_names = _solve_ik_smooth(targets_pos, targets_quat, home_q)
    print(f"[pipe] IK replay: {int(ik_ok.sum())}/{Tn}")

    qpos_approach = _plan_approach(
        home_q, qpos_replay[0], joint_names,
        plan_dt=args.plan_dt, approach_tsteps=args.approach_tsteps,
    )
    n_approach = qpos_approach.shape[0]
    qpos_full = np.concatenate([qpos_approach, qpos_replay], axis=0)
    print(f"[pipe] full arm traj: {qpos_full.shape[0]} ({n_approach} approach + {Tn} replay)")
    np.save(out_dir / "qpos_arm_full.npy", qpos_full)
    np.save(out_dir / "qpos_arm_approach.npy", qpos_approach)
    np.save(out_dir / "qpos_arm.npy", qpos_replay)
    np.save(out_dir / "n_approach.npy", np.array([n_approach]))
    np.save(out_dir / "joints_robot.npy", joints_robot)
    np.save(out_dir / "verts_robot.npy", verts_robot)

    # ---- 6. dex-retargeting with n_warmup=n_approach ----
    # Retargeter starts at qpos=0 (open hand). Feeding frame-0's reference for
    # n_approach iterations makes the low-pass filter ramp from open to the
    # first real pose, so saved hand traj matches arm traj length.
    qpos_hand_full, rt_joint_names = _retarget_hand(
        joints_cam0, valid, args.retarget_type, out_dir,
        scaling_factor=args.scaling_factor,
        low_pass_alpha=args.low_pass_alpha,
        n_warmup=n_approach,
    )
    qpos_hand = qpos_hand_full[n_approach:]
    print(f"[pipe] retarget done: full hand qpos shape={qpos_hand_full.shape}")
    np.save(out_dir / f"qpos_{side}.npy", qpos_hand)
    np.save(out_dir / "qpos_hand_full.npy", qpos_hand_full)
    (out_dir / f"joint_names_{side}.json").write_text(json.dumps(rt_joint_names))
    np.save(out_dir / f"joints_world_{side}.npy", joints_world)
    np.save(out_dir / f"verts_world_{side}.npy", verts_world)
    np.save(out_dir / f"faces_{side}.npy", faces)
    np.save(out_dir / f"valid_{side}.npy", valid)

    # ---- 7. viser visualization ----
    viewer = ViserViewer(scene_title="taeksoo pipeline")
    server = viewer.server

    if args.show_cameras:
        for serial, E_3x4 in extrinsic.items():
            E = _to4x4(E_3x4)
            cam_in_robot = robot_from_world @ np.linalg.inv(E)
            K = intrinsic[serial]["intrinsics_undistort"]
            h = float(intrinsic[serial]["height"]); w = float(intrinsic[serial]["width"])
            vfov = 2.0 * np.arctan(h / (2.0 * float(K[1, 1])))
            quat_xyzw = R.from_matrix(cam_in_robot[:3, :3]).as_quat()
            is_src = serial == args.source_serial
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
    for j, name in enumerate(urdf_joint_names):
        if name in name_to_arm_idx:
            qpos_full_urdf[:, j] = qpos_full[:, name_to_arm_idx[name]]
        elif name in name_to_hand_idx:
            qpos_full_urdf[:, j] = qpos_hand_full[:, name_to_hand_idx[name]]

    viewer.add_traj("taeksoo_pipeline", robot_traj={"xarm_allegro": qpos_full_urdf})

    faces_i32 = np.asarray(faces, dtype=np.int32)
    mano_color = tuple(int(c) for c in args.mano_color)
    state = {"handle": None}

    def _set_mano(t_full: int) -> None:
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
