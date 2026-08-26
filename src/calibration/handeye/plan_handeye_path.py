"""Build a collision-checked, continuous joint path through the hand-eye waypoints.

Straight-line joint interpolation dips the board into the floor. Instead, each transition
A -> B is done in CARTESIAN space as lift -> move -> lower: raise the board to a safe
height, translate above the target, lower down. Every micro-step is IK'd (seeded from the
previous config for continuity) and collision-checked (board vs floor + board vs arm); if a
step still collides the lift height is raised and the transition retried. Output is one
dense joint trajectory + the frame indices where the arm stops to capture.

Runs in the franka env (needs pinocchio for IK). Saves to hecalib/franka_path/.

    python src/calibration/handeye/plan_handeye_path.py --floor-z 0.037 --view
"""
import argparse
import importlib.util
import os
import sys
from pathlib import Path

import numpy as np
import yourdfpy
from scipy.spatial.transform import Rotation as R, Slerp

sys.path.append(str(Path(__file__).resolve().parents[3]))
from paradex.utils.system import config_dir

_pf = importlib.util.spec_from_file_location("pf", Path(__file__).parent / "plan_handeye_flat.py")
pf = importlib.util.module_from_spec(_pf); _pf.loader.exec_module(pf)

EEF_LINK = "fr3_link8"
HOME = np.array([0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785])
HALF = np.array([pf.BOARD_W / 2, pf.BOARD_H / 2, pf.BOARD_T / 2])
OUT_DIR = os.path.join(config_dir, "hecalib", "franka_path")


def board_from_eef():
    T = np.eye(4); T[2, 3] = pf.BOARD_Z_FROM_EEF; return T


def make_collision_fns(fk):
    """Return (board_pose(q), collides(q, floor, margin)) using the arm meshes."""
    scene = fk.scene
    arm = []
    for node in scene.graph.nodes_geometry:
        g = scene.graph[node][1]
        if any(g.startswith(f"link{i}.") for i in range(7)):
            v = np.asarray(scene.geometry[g].vertices)
            if len(v) > 200:
                v = v[np.random.default_rng(0).choice(len(v), 200, replace=False)]
            arm.append((node, v))

    def board_pose(q):
        fk.update_cfg(q)
        T = np.array(fk.get_transform(EEF_LINK, fk.base_link))
        T[:3, 3] = T[:3, 3] + T[:3, :3] @ np.array([0, 0, pf.BOARD_Z_FROM_EEF])
        return T

    def collides(q, floor_z, margin):
        Tb = board_pose(q)
        if pf.board_min_z(Tb) < floor_z:
            return True
        fk.update_cfg(q)
        for node, v in arm:
            Tl = np.array(scene.graph[node][0])
            pw = (Tl[:3, :3] @ v.T).T + Tl[:3, 3]
            loc = (pw - Tb[:3, 3]) @ Tb[:3, :3]
            if np.linalg.norm(np.maximum(np.abs(loc) - HALF, 0.0), axis=1).min() < margin:
                return True
        return False

    return board_pose, collides


def cart_interp(TA, TB, n):
    """n poses from TA to TB: lerp position, slerp orientation."""
    key = Slerp([0, 1], R.from_matrix([TA[:3, :3], TB[:3, :3]]))
    out = []
    for t in np.linspace(0, 1, n):
        T = np.eye(4)
        T[:3, :3] = key(t).as_matrix()
        T[:3, 3] = (1 - t) * TA[:3, 3] + t * TB[:3, 3]
        out.append(T)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", default="franka")
    ap.add_argument("--floor-z", type=float, default=0.037)
    ap.add_argument("--margin", type=float, default=0.01)
    ap.add_argument("--lift", type=float, default=0.12, help="how high to raise the board for transit (m)")
    ap.add_argument("--cart-step", type=float, default=0.03, help="cartesian step size (m)")
    ap.add_argument("--view", action="store_true")
    ap.add_argument("--port", type=int, default=8080)
    args = ap.parse_args()

    from paradex.robot.robot_wrapper import RobotWrapper
    from paradex.robot.utils import get_robot_urdf_path
    urdf_path = get_robot_urdf_path(arm_name=args.arm)
    rw = RobotWrapper(urdf_path)
    fk = yourdfpy.URDF.load(urdf_path, build_scene_graph=True)
    board_pose, collides = make_collision_fns(fk)
    eef_from_board = np.linalg.inv(board_from_eef())

    traj_dir = os.path.join(config_dir, "hecalib", args.arm)
    wps = [np.load(os.path.join(traj_dir, f)) for f in
           sorted([f for f in os.listdir(traj_dir) if f.endswith("_qpos.npy")], key=lambda f: int(f.split("_")[0]))]
    seq = [HOME] + wps
    print(f"{len(wps)} waypoints; planning collision-free transits (lift {args.lift} m)")

    def ik(T_board, seed):
        q, ok = rw.solve_ik(T_board @ eef_from_board, EEF_LINK, q_init=seed, max_iter=300, try_num=6)
        return q if ok else None

    def transit(qA, qB, lift):
        """Cartesian lift->move->lower from qA to qB, IK'd + collision-checked. None if it fails."""
        TA, TB = board_pose(qA), board_pose(qB)
        TAu, TBu = TA.copy(), TB.copy()
        TAu[2, 3] += lift; TBu[2, 3] += lift
        path_q = [qA]
        seed = qA
        for T0, T1 in [(TA, TAu), (TAu, TBu), (TBu, TB)]:
            n = max(2, int(np.linalg.norm(T1[:3, 3] - T0[:3, 3]) / args.cart_step) + 1)
            for T in cart_interp(T0, T1, n)[1:]:
                q = ik(T, seed)
                if q is None or collides(q, args.floor_z, args.margin):
                    return None
                path_q.append(q); seed = q
        return path_q

    full, capture_idx = [seq[0]], [0]
    ok_all = True
    for i in range(len(seq) - 1):
        seg = None
        for lift in (args.lift, args.lift * 2, args.lift * 3):     # raise lift if it still collides
            seg = transit(seq[i], seq[i + 1], lift)
            if seg is not None:
                break
        if seg is None:
            print(f"  transition {i} ({'HOME' if i == 0 else f'wp{i-1}'} -> wp{i}): FAILED (no collision-free lift)")
            ok_all = False
            full.append(seq[i + 1])                                # fall back to a jump (unsafe) so indices stay
        else:
            full.extend(seg[1:])
            print(f"  transition {i}: {len(seg)} steps, collision-free")
        capture_idx.append(len(full) - 1)                          # this waypoint = stop & capture here

    full = np.array(full)
    print(f"\ndense path: {len(full)} configs, {len(capture_idx)} capture stops. "
          f"{'ALL transits collision-free' if ok_all else 'SOME transits failed (see above)'}")
    os.makedirs(OUT_DIR, exist_ok=True)
    np.save(os.path.join(OUT_DIR, "path.npy"), full)
    np.save(os.path.join(OUT_DIR, "capture_idx.npy"), np.array(capture_idx))
    print(f"saved -> {OUT_DIR}/path.npy (+ capture_idx.npy)")

    if args.view:
        _view(fk, full, capture_idx, board_pose, args)


def _view(fk, full, capture_idx, board_pose, args):
    import trimesh
    from paradex.visualization.visualizer.viser import ViserViewer
    from paradex.robot.utils import get_robot_urdf_path
    viewer = ViserViewer(port_number=args.port)
    viewer.add_grid(size=2.0, cell_size=0.1, height=args.floor_z)
    viewer.add_robot(args.arm, get_robot_urdf_path(arm_name=args.arm))
    b = trimesh.creation.box(extents=[pf.BOARD_W, pf.BOARD_H, pf.BOARD_T]); b.visual.vertex_colors = [235, 235, 235, 255]
    viewer.add_trimesh("board", b, np.eye(4))
    sl = viewer.server.gui.add_slider("frame", min=0, max=len(full) - 1, step=1, initial_value=0)
    txt = viewer.server.gui.add_text("info", initial_value="")
    caps = set(int(c) for c in capture_idx)

    def show(_=None):
        i = int(sl.value)
        viewer.robot_dict[args.arm].update_cfg(full[i])
        Tb = board_pose(full[i])
        h = viewer.obj_dict["board"]["frame"]
        h.position = Tb[:3, 3]; h.wxyz = R.from_matrix(Tb[:3, :3]).as_quat()[[3, 0, 1, 2]]
        txt.value = f"frame {i}/{len(full)-1} {'[CAPTURE]' if i in caps else ''} board_min_z={pf.board_min_z(Tb):.3f}"
    sl.on_update(show); show()
    print(f"open http://localhost:{args.port} . drag 'frame'. Ctrl-C to quit.")
    viewer.start_viewer(use_thread=False)


if __name__ == "__main__":
    main()
