"""Collision-CHECK a saved hand-eye trajectory along its motion (no planning, no GPU).

Interpolates the joint-space path (HOME -> wp0 -> wp1 -> ...) and, at every frame, checks:
  - board vs floor  : lowest board corner below the raised floor?
  - board vs arm    : any arm-link (fr3_link0..6) surface point within `margin` of the board box?
Reports the colliding frames/segments. (Arm-arm self-collision needs an adjacency-ignore
list -> use curobo for that; the two checks here are the realistic ones for this rig.)

    python src/calibration/handeye/check_traj_collision.py --floor-z 0.037 --steps 20
"""
import argparse
import importlib.util
import os
import sys
from pathlib import Path

import numpy as np
import yourdfpy

sys.path.append(str(Path(__file__).resolve().parents[3]))
from paradex.utils.system import config_dir
from paradex.robot.utils import get_robot_urdf_path

_pf_spec = importlib.util.spec_from_file_location("pf", Path(__file__).parent / "plan_handeye_flat.py")
pf = importlib.util.module_from_spec(_pf_spec); _pf_spec.loader.exec_module(pf)

EEF_LINK = "fr3_link8"
HOME = np.array([0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785])
HALF = np.array([pf.BOARD_W / 2, pf.BOARD_H / 2, pf.BOARD_T / 2])


def board_pose(fk, q):
    fk.update_cfg(q)
    T = np.array(fk.get_transform(EEF_LINK, fk.base_link))
    T[:3, 3] = T[:3, 3] + T[:3, :3] @ np.array([0, 0, pf.BOARD_Z_FROM_EEF])
    return T


def dist_to_board(T_board, pts_world):
    """Min distance from world points to the board box (0 if inside)."""
    local = (pts_world - T_board[:3, 3]) @ T_board[:3, :3]     # world->board frame
    d = np.maximum(np.abs(local) - HALF, 0.0)
    return np.linalg.norm(d, axis=1).min()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", default="franka")
    ap.add_argument("--floor-z", type=float, default=0.037)
    ap.add_argument("--margin", type=float, default=0.01, help="board-arm clearance (m)")
    ap.add_argument("--steps", type=int, default=20, help="interpolation steps per segment")
    args = ap.parse_args()

    traj_dir = os.path.join(config_dir, "hecalib", args.arm)
    wps = [np.load(os.path.join(traj_dir, f)) for f in
           sorted([f for f in os.listdir(traj_dir) if f.endswith("_qpos.npy")], key=lambda f: int(f.split("_")[0]))]
    print(f"{len(wps)} waypoints from {traj_dir}")
    seq = [HOME] + wps

    urdf_path = get_robot_urdf_path(arm_name=args.arm)
    fk = yourdfpy.URDF.load(urdf_path, build_scene_graph=True)
    scene = fk.scene
    # arm-link meshes are named link0.dae .. link6.dae (skip link7/8 -> the board mounts there).
    # Each geometry node's transform in scene.graph is already the world pose after update_cfg.
    arm_geoms = []
    for node in scene.graph.nodes_geometry:
        gname = scene.graph[node][1]
        if any(gname.startswith(f"link{i}.") for i in range(7)):
            v = np.asarray(scene.geometry[gname].vertices)
            if len(v) > 250:
                v = v[np.random.default_rng(0).choice(len(v), 250, replace=False)]
            arm_geoms.append((node, v))

    def dense(a, b):
        return [(1 - k / args.steps) * a + (k / args.steps) * b for k in range(args.steps)]
    frames = []
    for i in range(len(seq) - 1):
        frames += dense(seq[i], seq[i + 1])
    frames.append(seq[-1])

    floor_bad, arm_bad = [], []
    min_floor, min_arm = 1e9, 1e9
    for fi, q in enumerate(frames):
        Tb = board_pose(fk, q)
        bmin = pf.board_min_z(Tb)
        min_floor = min(min_floor, bmin)
        if bmin < args.floor_z:
            floor_bad.append(fi)
        # arm points in world at this q
        fk.update_cfg(q)
        dmin = 1e9
        for node, v in arm_geoms:
            Tl = np.array(scene.graph[node][0])       # world pose of this mesh at q
            pw = (Tl[:3, :3] @ v.T).T + Tl[:3, 3]
            dmin = min(dmin, dist_to_board(Tb, pw))
        min_arm = min(min_arm, dmin)
        if dmin < args.margin:
            arm_bad.append(fi)

    print(f"\nchecked {len(frames)} frames (floor {args.floor_z}, arm margin {args.margin})")
    print(f"BOARD-FLOOR : min board_min_z {min_floor:.3f}  -> "
          f"{'COLLIDES at %d frame(s)' % len(floor_bad) if floor_bad else 'clear'}")
    print(f"BOARD-ARM   : min clearance  {min_arm:.3f}  -> "
          f"{'COLLIDES at %d frame(s)' % len(arm_bad) if arm_bad else 'clear'}")
    for name, bad in [("floor", floor_bad), ("arm", arm_bad)]:
        if bad:
            segs = sorted(set(b // args.steps for b in bad))
            print(f"  {name} collision in segment(s) {segs}  (segment k = move waypoint k-1 -> k, 0 = HOME->wp0)")
    if not floor_bad and not arm_bad:
        print("=> trajectory is collision-clear (board vs floor & arm) along the whole motion.")


if __name__ == "__main__":
    main()
