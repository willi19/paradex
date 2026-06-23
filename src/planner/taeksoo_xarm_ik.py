"""Per-frame xarm IK + visualization.

Pipeline:
  1. Load taeksoo.py outputs (joints_world, verts_world, qpos for allegro_v5).
  2. Apply latest C2R to bring everything from calib world frame into xarm
     base (robot) frame. Cameras too.
  3. For each frame, build target palm_link pose (xarm base frame) from MANO
     middle MCP + dex-retargeting operator orientation.
  4. Run cuRobo IK (arm-only, 6 DoF; allegro joints locked) per frame, using
     previous q as seed. No collision check.
  5. Save xarm trajectory.
  6. Visualize MANO mesh + xarm_allegro_v5 URDF together (xarm = IK qpos,
     allegro = retargeted qpos), in xarm base frame.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from scipy.spatial.transform import Rotation as R

sys.path.append(str(Path(__file__).resolve().parents[2]))

from paradex.calibration.utils import load_current_C2R, load_current_camparam
from paradex.visualization.visualizer.viser import ViserViewer

SOURCE_SERIAL = "23029839"
URDF_PATH = "/home/temp_id/paradex/rsc/robot/xarm_allegro_v5.urdf"

OPERATOR2MANO_RIGHT = np.array(
    [[0, 0, -1], [-1, 0, 0], [0, 1, 0]], dtype=np.float64
)

ARM_JOINTS = [f"joint{i}" for i in range(1, 7)]
HAND_JOINTS = [f"joint_{i}_0" for i in (0, 1, 2, 3, 12, 13, 14, 15, 4, 5, 6, 7, 8, 9, 10, 11)]


def _to4x4(m: np.ndarray) -> np.ndarray:
    m = np.asarray(m, dtype=np.float64)
    if m.shape == (4, 4):
        return m
    if m.shape == (3, 4):
        out = np.eye(4)
        out[:3, :] = m
        return out
    raise ValueError(f"bad shape {m.shape}")


def _transform_pts(pts: np.ndarray, T_4x4: np.ndarray) -> np.ndarray:
    return pts @ T_4x4[:3, :3].T + T_4x4[:3, 3]


def _estimate_mano_wrist_frame(kp3: np.ndarray) -> np.ndarray:
    pts = kp3[[0, 5, 9], :]
    x_vec = pts[0] - pts[2]
    centered = pts - pts.mean(axis=0, keepdims=True)
    _, _, v = np.linalg.svd(centered)
    normal = v[2, :]
    x = x_vec - np.sum(x_vec * normal) * normal
    x = x / (np.linalg.norm(x) + 1e-8)
    z = np.cross(x, normal)
    if np.sum(z * (pts[1] - pts[2])) < 0:
        normal *= -1
        z *= -1
    return np.stack([x, normal, z], axis=1)


def _palm_pose_in_robot(joints_robot_t: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Returns (position[3], quat_wxyz[4]) for palm_link target in xarm base frame."""
    wrist_rot = _estimate_mano_wrist_frame(joints_robot_t)
    R_robot_palm = wrist_rot @ OPERATOR2MANO_RIGHT
    pos = joints_robot_t[9]  # middle MCP, matching retargeting origin
    quat_xyzw = R.from_matrix(R_robot_palm).as_quat()
    quat_wxyz = quat_xyzw[[3, 0, 1, 2]]
    return pos, quat_wxyz


def _build_curobo_cfg() -> dict:
    """Minimal robot cfg dict for curobo IK (arm-only, no collision).
    Hand joints branch off palm_link so they're outside the IK chain
    (link_base -> palm_link); we don't need to lock them."""
    return {
        "robot_cfg": {
            "kinematics": {
                "use_usd_kinematics": False,
                "urdf_path": URDF_PATH,
                "asset_root_path": "",
                "base_link": "link_base",
                "ee_link": "palm_link",
                "link_names": ["wrist", "palm_link"],
                "collision_link_names": [],
                "collision_spheres": None,
                "collision_sphere_buffer": 0.0,
                "self_collision_ignore": {},
                "self_collision_buffer": {},
                "use_global_cumul": True,
                "mesh_link_names": [],
                "extra_collision_spheres": {},
                "cspace": {
                    "joint_names": list(ARM_JOINTS),
                    "retract_config": [0.0, -0.5, -1.0, 0.0, 1.5, 0.0],
                    "cspace_distance_weight": [1.0] * 6,
                    "null_space_weight": [1.0] * 6,
                    "max_jerk": 500.0,
                    "max_acceleration": 15.0,
                },
            }
        }
    }


def _solve_ik(
    targets_pos: np.ndarray,   # (T, 3)
    targets_quat: np.ndarray,  # (T, 4) wxyz
) -> Tuple[np.ndarray, np.ndarray]:
    """Per-frame IK; returns (qpos[T,6], success[T] bool)."""
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

    if hasattr(ik, "kinematics") and hasattr(ik.kinematics, "joint_names"):
        joint_names = list(ik.kinematics.joint_names)
    else:
        joint_names = list(ARM_JOINTS)
    print(f"[ik] solver joints: {joint_names}")

    Tn = targets_pos.shape[0]
    q_out = np.full((Tn, len(joint_names)), np.nan, dtype=np.float32)
    success = np.zeros(Tn, dtype=bool)

    retract = np.asarray(
        cfg["robot_cfg"]["kinematics"]["cspace"]["retract_config"], dtype=np.float32
    )
    current_q = retract.copy()

    for t in range(Tn):
        pos_t = tensor_args.to_device(targets_pos[t : t + 1].astype(np.float32))
        quat_t = tensor_args.to_device(targets_quat[t : t + 1].astype(np.float32))
        goal = Pose(position=pos_t, quaternion=quat_t)
        retract_t = tensor_args.to_device(current_q.reshape(1, -1))
        seed_t = tensor_args.to_device(current_q.reshape(1, 1, -1))
        try:
            res = ik.solve_single(goal, retract_config=retract_t, seed_config=seed_t)
        except Exception as e:
            print(f"[ik] frame {t}: solve failed: {type(e).__name__}: {e}")
            q_out[t] = current_q
            continue

        ok_attr = getattr(res, "success", False)
        ok = bool(ok_attr.item()) if hasattr(ok_attr, "item") else bool(ok_attr)
        sol = getattr(getattr(res, "js_solution", None), "position", None)
        if sol is None:
            sol = getattr(res, "solution", None)
        if sol is None:
            print(f"[ik] frame {t}: no solution tensor")
            q_out[t] = current_q
            continue
        arr = sol.detach().cpu().numpy().reshape(-1, len(joint_names))
        q_now = arr[0].astype(np.float32)
        q_out[t] = q_now
        success[t] = ok
        if ok:
            current_q = q_now
    return q_out, success


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default=str(Path(__file__).parent / "out_taeksoo"))
    ap.add_argument("--side", default="right", choices=["right"])
    ap.add_argument("--show-cameras", action="store_true", default=True)
    ap.add_argument("--no-cameras", dest="show_cameras", action="store_false")
    ap.add_argument("--cam-scale", type=float, default=0.05)
    ap.add_argument("--mano-color", nargs=3, type=int, default=[90, 200, 255])
    args = ap.parse_args()

    out = Path(args.out_dir)
    qpos_hand = np.load(out / f"qpos_{args.side}.npy")              # (T, 16)
    joints_world = np.load(out / f"joints_world_{args.side}.npy")    # (T, 21, 3)
    verts_world = np.load(out / f"verts_world_{args.side}.npy")      # (T, 778, 3)
    faces = np.load(out / f"faces_{args.side}.npy")
    rt_joint_names = json.loads((out / f"joint_names_{args.side}.json").read_text())
    Tn = qpos_hand.shape[0]
    print(f"[ik] T={Tn}")

    # C2R: world_from_robot (based on cam_from_world @ c2r = cam_from_robot).
    c2r = _to4x4(load_current_C2R())
    robot_from_world = np.linalg.inv(c2r)
    print(f"[ik] C2R loaded (world_from_robot); shape={c2r.shape}")

    joints_robot = np.stack(
        [_transform_pts(joints_world[t], robot_from_world) for t in range(Tn)]
    )
    verts_robot = np.stack(
        [_transform_pts(verts_world[t], robot_from_world) for t in range(Tn)]
    )

    targets_pos = np.zeros((Tn, 3), dtype=np.float32)
    targets_quat = np.zeros((Tn, 4), dtype=np.float32)
    for t in range(Tn):
        p, q = _palm_pose_in_robot(joints_robot[t])
        targets_pos[t] = p
        targets_quat[t] = q

    qpos_arm, success = _solve_ik(targets_pos, targets_quat)
    print(f"[ik] solved {int(success.sum())}/{Tn}")
    np.save(out / "qpos_arm.npy", qpos_arm)
    np.save(out / "ik_success.npy", success)
    np.save(out / "ik_target_pos.npy", targets_pos)
    np.save(out / "ik_target_quat.npy", targets_quat)
    np.save(out / "joints_robot.npy", joints_robot)
    np.save(out / "verts_robot.npy", verts_robot)

    # ---------- visualization ----------
    viewer = ViserViewer(scene_title="taeksoo xarm IK")
    server = viewer.server

    if args.show_cameras:
        intrinsic, extrinsic = load_current_camparam()
        for serial, E_3x4 in extrinsic.items():
            E = _to4x4(E_3x4)            # cam_from_world
            cam_in_world = np.linalg.inv(E)
            cam_in_robot = robot_from_world @ cam_in_world
            K = intrinsic[serial]["intrinsics_undistort"]
            h = float(intrinsic[serial]["height"])
            w = float(intrinsic[serial]["width"])
            vfov = 2.0 * np.arctan(h / (2.0 * float(K[1, 1])))
            aspect = w / h
            is_src = serial == SOURCE_SERIAL
            quat_xyzw = R.from_matrix(cam_in_robot[:3, :3]).as_quat()
            server.scene.add_camera_frustum(
                name=f"/cameras/{serial}",
                fov=float(vfov),
                aspect=float(aspect),
                scale=float(args.cam_scale * (1.4 if is_src else 1.0)),
                color=(255, 80, 80) if is_src else (160, 160, 160),
                position=cam_in_robot[:3, 3],
                wxyz=quat_xyzw[[3, 0, 1, 2]],
            )
        print(f"[viz] {len(extrinsic)} cameras (in robot frame)")

    viewer.add_robot(name="xarm_allegro", urdf_path=URDF_PATH)
    robot = viewer.robot_dict["xarm_allegro"]
    urdf_joint_names = list(robot.urdf.get_joint_names())

    name_to_hand_idx = {n: i for i, n in enumerate(rt_joint_names)}
    name_to_arm_idx = {n: i for i, n in enumerate(ARM_JOINTS)}
    qpos_full = np.zeros((Tn, len(urdf_joint_names)), dtype=np.float64)
    for j, name in enumerate(urdf_joint_names):
        if name in name_to_arm_idx:
            qpos_full[:, j] = qpos_arm[:, name_to_arm_idx[name]]
        elif name in name_to_hand_idx:
            qpos_full[:, j] = qpos_hand[:, name_to_hand_idx[name]]
        # else leave 0 (e.g. fixed joints; but yourdfpy returns only actuated)

    viewer.add_traj("taeksoo_xarm", robot_traj={"xarm_allegro": qpos_full})

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
    print("[viz] open URL printed by viser")
    viewer.start_viewer()


if __name__ == "__main__":
    main()
