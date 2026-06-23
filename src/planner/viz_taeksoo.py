"""Viser visualization: overlay MANO mesh + retargeted allegro_v5 hand.

Usage:
    python src/planner/viz_taeksoo.py [--out-dir src/planner/out_taeksoo] [--side right]

Loads taeksoo.py outputs and renders both hands in the same world frame so you
can eyeball the retargeting quality.
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

from paradex.calibration.utils import load_current_camparam
from paradex.visualization.visualizer.viser import ViserViewer

SOURCE_SERIAL = "23029839"

OPERATOR2MANO_RIGHT = np.array(
    [[0, 0, -1], [-1, 0, 0], [0, 1, 0]], dtype=np.float64
)
OPERATOR2MANO_LEFT = np.array(
    [[0, 0, -1], [1, 0, 0], [0, -1, 0]], dtype=np.float64
)

ALLEGRO_V5_URDF = {
    "right": "/home/temp_id/paradex/rsc/robot/allegro_v5/allegro_v5_right.urdf",
}


def _estimate_mano_wrist_frame(kp3: np.ndarray) -> np.ndarray:
    points = kp3[[0, 5, 9], :]
    x_vector = points[0] - points[2]
    centered = points - np.mean(points, axis=0, keepdims=True)
    _, _, v = np.linalg.svd(centered)
    normal = v[2, :]
    x = x_vector - np.sum(x_vector * normal) * normal
    x = x / (np.linalg.norm(x) + 1e-8)
    z = np.cross(x, normal)
    if np.sum(z * (points[1] - points[2])) < 0:
        normal *= -1
        z *= -1
    return np.stack([x, normal, z], axis=1)


def _add_cameras(server, intrinsic: dict, extrinsic: dict, scale: float) -> None:
    """Add camera frustums for all calibrated cameras. Highlights SOURCE_SERIAL."""
    for serial, E_3x4 in extrinsic.items():
        E = np.eye(4)
        E[:3, :] = np.asarray(E_3x4, dtype=np.float64)  # cam_from_world
        E_inv = np.linalg.inv(E)                         # world_from_cam
        pos = E_inv[:3, 3]
        wxyz = R.from_matrix(E_inv[:3, :3]).as_quat()[[3, 0, 1, 2]]
        K = intrinsic[serial]["intrinsics_undistort"]
        fy = float(K[1, 1])
        h = float(intrinsic[serial]["height"])
        w = float(intrinsic[serial]["width"])
        vfov = 2.0 * np.arctan(h / (2.0 * fy))
        aspect = w / h
        is_src = serial == SOURCE_SERIAL
        color = (255, 80, 80) if is_src else (160, 160, 160)
        server.scene.add_camera_frustum(
            name=f"/cameras/{serial}",
            fov=float(vfov),
            aspect=float(aspect),
            scale=float(scale * (1.4 if is_src else 1.0)),
            color=color,
            position=pos,
            wxyz=wxyz,
        )
        server.scene.add_label(name=f"/cameras/{serial}/label", text=serial)


def _palm_pose(
    joints_world_t: np.ndarray,
    side: str,
    anchor: str = "mcp_center",
    palm_offset: np.ndarray = np.zeros(3),
) -> np.ndarray:
    """4x4 pose for allegro palm_link.

    Orientation: dex-retargeting operator (palm) frame in world.
    Position: depends on `anchor`:
      - "wrist"      : MANO joint 0
      - "mcp_center" : mean of index/middle/ring/pinky MCPs (5,9,13,17)
    palm_offset is added in palm-local frame.
    """
    op = OPERATOR2MANO_RIGHT if side == "right" else OPERATOR2MANO_LEFT
    wrist_rot = _estimate_mano_wrist_frame(joints_world_t)
    R_world_palm = wrist_rot @ op
    if anchor == "wrist":
        anchor_pos = joints_world_t[0]
    elif anchor == "middle_mcp":
        anchor_pos = joints_world_t[9]
    elif anchor == "mcp_center":
        anchor_pos = joints_world_t[[5, 9, 13, 17]].mean(axis=0)
    else:
        raise ValueError(f"unknown anchor: {anchor}")
    pose = np.eye(4)
    pose[:3, :3] = R_world_palm
    pose[:3, 3] = anchor_pos + R_world_palm @ np.asarray(palm_offset, dtype=np.float64)
    return pose


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default=str(Path(__file__).parent / "out_taeksoo"))
    ap.add_argument("--side", default="right", choices=["right", "left"])
    ap.add_argument("--mano-color", nargs=3, type=int, default=[90, 200, 255])
    ap.add_argument("--no-cameras", dest="show_cameras", action="store_false",
                    help="Skip rendering camera frustums.")
    ap.set_defaults(show_cameras=True)
    ap.add_argument("--calib-name", default=None,
                    help="cam_param subdir; defaults to latest.")
    ap.add_argument("--cam-scale", type=float, default=0.05,
                    help="Camera frustum scale (m).")
    ap.add_argument(
        "--anchor", default="middle_mcp",
        choices=["wrist", "middle_mcp", "mcp_center"],
        help="Which MANO point to align robot palm_link to "
        "(must match retargeting origin in taeksoo.py yaml).",
    )
    ap.add_argument(
        "--palm-offset", nargs=3, type=float, default=[0.0, 0.0, 0.0],
        help="Extra palm-local offset (m) added on top of anchor.",
    )
    args = ap.parse_args()

    out = Path(args.out_dir)
    qpos = np.load(out / f"qpos_{args.side}.npy")                # (T, n_dof)
    joints_world = np.load(out / f"joints_world_{args.side}.npy")  # (T, 21, 3)
    verts_world = np.load(out / f"verts_world_{args.side}.npy")    # (T, 778, 3)
    faces = np.load(out / f"faces_{args.side}.npy")
    valid = np.load(out / f"valid_{args.side}.npy")
    rt_joint_names = json.loads((out / f"joint_names_{args.side}.json").read_text())

    T = qpos.shape[0]
    print(f"[viz] T={T} side={args.side} valid={int(valid.sum())}")

    viewer = ViserViewer(scene_title=f"taeksoo {args.side}")
    if args.show_cameras:
        intrinsic, extrinsic = load_current_camparam(args.calib_name)
        _add_cameras(viewer.server, intrinsic, extrinsic, args.cam_scale)
        print(f"[viz] {len(extrinsic)} cameras added (source={SOURCE_SERIAL} red)")
    viewer.add_robot(name="allegro_v5", urdf_path=ALLEGRO_V5_URDF[args.side])
    robot = viewer.robot_dict["allegro_v5"]
    urdf_joint_names = list(robot.urdf.get_joint_names())

    # qpos -> URDF joint order remap (retargeter orders joints differently).
    name_to_idx = {n: i for i, n in enumerate(rt_joint_names)}
    remap = np.array(
        [name_to_idx[n] for n in urdf_joint_names if n in name_to_idx],
        dtype=np.int64,
    )
    urdf_dof = [n for n in urdf_joint_names if n in name_to_idx]
    if len(urdf_dof) != len(urdf_joint_names):
        missing = set(urdf_joint_names) - set(urdf_dof)
        print(f"[viz] WARN: URDF joints not in retargeter qpos: {missing}")

    server = viewer.server
    faces_i32 = np.asarray(faces, dtype=np.int32)
    mano_color = tuple(int(c) for c in args.mano_color)

    qpos_urdf = np.zeros((T, len(urdf_joint_names)), dtype=np.float64)
    qpos_urdf[:, : len(remap)] = qpos[:, remap]
    viewer.add_traj("taeksoo", robot_traj={"allegro_v5": qpos_urdf})

    mano_state = {"handle": None}

    def _set_mano_mesh(t: int) -> None:
        if mano_state["handle"] is not None:
            mano_state["handle"].remove()
        mano_state["handle"] = server.scene.add_mesh_simple(
            name="/mano/mesh",
            vertices=verts_world[t].astype(np.float32),
            faces=faces_i32,
            color=mano_color,
            opacity=0.55,
        )

    palm_offset = np.asarray(args.palm_offset, dtype=np.float64)

    def _apply_extras(t: int) -> None:
        t = int(np.clip(t, 0, T - 1))
        pose = _palm_pose(joints_world[t], args.side, args.anchor, palm_offset)
        if hasattr(robot, "_visual_root_frame"):
            quat_xyzw = R.from_matrix(pose[:3, :3]).as_quat()
            robot._visual_root_frame.wxyz = quat_xyzw[[3, 0, 1, 2]]
            robot._visual_root_frame.position = pose[:3, 3]
        _set_mano_mesh(t)

    @viewer.gui_timestep.on_update
    def _(_):
        _apply_extras(int(viewer.gui_timestep.value))

    _apply_extras(0)
    print("[viz] open the URL viser printed (typically http://localhost:8080)")
    viewer.start_viewer()


if __name__ == "__main__":
    main()
