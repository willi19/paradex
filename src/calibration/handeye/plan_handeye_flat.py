"""Plan a hand-eye trajectory for a FLAT 5x6 board on the franka (no cube).

The board (300 x 360 mm, 3 mm) sits on the board_adapter plate, its normal along the
flange +z. Unlike the cube, a flat board must FACE the cameras, so each sampled pose
orients the board normal toward the camera cloud (+ random roll about the normal and a
small tilt for rotation diversity). Poses are kept only if reachable (IK) and if the
board (and every link) clears the raised floor. Then farthest-point selection spreads
[R,t]. Review with the Viser slider; --write saves to hecalib/franka.

    python src/calibration/handeye/plan_handeye_flat.py --n 20 --floor-z 0.037 --write
"""
import argparse
import os
import sys
from pathlib import Path

import numpy as np
import trimesh
import yourdfpy
from scipy.spatial.transform import Rotation as R

sys.path.append(str(Path(__file__).resolve().parents[3]))
from paradex.utils.system import config_dir
from paradex.calibration.utils import handeye_calib_path
import importlib.util
_spec = importlib.util.spec_from_file_location(
    "ph", str(Path(__file__).parent / "plan_handeye_poses.py"))
ph = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ph)

REPO = Path(__file__).resolve().parents[3]
FRANKA = str(REPO / "rsc/robot/franka/franka.urdf")
FULL_URDF = str(REPO / "rsc/robot/franka/franka_adapter.urdf")
EEF_LINK = "fr3_link8"
BOARD_Z_FROM_EEF = 0.0475          # board plane along flange +z
BOARD_W, BOARD_H, BOARD_T = 0.30, 0.36, 0.003   # 5x6 @ 60mm + 3mm
MARKER_MM = 50.0                   # old board marker
OUT_DIR = os.path.join(config_dir, "hecalib", "franka")

_BOARD_CORNERS = np.array([[sx * BOARD_W / 2, sy * BOARD_H / 2, sz * BOARD_T / 2]
                           for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)])


def flat_board_pose(pos, cam_centroid, tilt_deg, rng):
    """Board at pos with normal (+z) pointing toward the cameras, + roll + small tilt."""
    z = cam_centroid - pos
    z = z / np.linalg.norm(z)
    a = np.array([0, 0, 1.0]) if abs(z[2]) < 0.9 else np.array([1.0, 0, 0])
    x = np.cross(a, z); x /= np.linalg.norm(x)
    y = np.cross(z, x)
    Rmat = np.column_stack([x, y, z])
    roll = R.from_euler("z", rng.uniform(0, 360), degrees=True).as_matrix()
    tilt = R.from_euler(rng.choice(["x", "y"]), rng.uniform(-tilt_deg, tilt_deg), degrees=True).as_matrix()
    T = np.eye(4)
    T[:3, :3] = Rmat @ roll @ tilt
    T[:3, 3] = pos
    return T


def board_min_z(T):
    return ((T[:3, :3] @ _BOARD_CORNERS.T).T + T[:3, 3])[:, 2].min()


def score_flat(T, cams):
    """How many cameras see the board face well (facing them, in FOV, marker resolves)."""
    n = T[:3, :3] @ np.array([0, 0, 1.0])
    c = T[:3, 3]
    good = 0
    for cam in cams:
        d = c - cam["C"]
        dist = np.linalg.norm(d)
        if not (ph.DIST_RANGE[0] <= dist <= ph.DIST_RANGE[1]):
            continue
        ray = d / dist
        cos_i = float(np.dot(-ray, n))
        if cos_i < np.cos(np.radians(ph.MAX_INCIDENCE)):
            continue
        pc = cam["Rwc"] @ c + cam["twc"]
        if pc[2] <= 0:
            continue
        uv = cam["K"] @ (pc / pc[2])
        if not (0 <= uv[0] < cam["w"] and 0 <= uv[1] < cam["h"]):
            continue
        if MARKER_MM * (cam["fx"] / (dist * 1000.0)) * cos_i >= ph.MARKER_MIN_PX:
            good += 1
    return good


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cam-session", default=None)
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--center", nargs=3, type=float, default=None)
    ap.add_argument("--extent", type=float, default=0.30)
    ap.add_argument("--x-min", type=float, default=0.25, help="keep the board at least this far in front of the base")
    ap.add_argument("--z-min", type=float, default=None)
    ap.add_argument("--z-max", type=float, default=0.55)
    ap.add_argument("--tilt", type=float, default=35.0)
    ap.add_argument("--min-obs", type=int, default=4)
    ap.add_argument("--floor-z", type=float, default=0.037, help="raised floor height (m)")
    ap.add_argument("--floor-margin", type=float, default=0.02)
    ap.add_argument("--write", action="store_true")
    ap.add_argument("--no-viser", action="store_true")
    ap.add_argument("--port", type=int, default=8080)
    args = ap.parse_args()

    session = args.cam_session
    if session is None:
        for s in sorted(os.listdir(handeye_calib_path), reverse=True):
            if os.path.exists(os.path.join(handeye_calib_path, s, "0", "C2R.npy")):
                session = s
                break
    cams = ph.load_cameras(session)
    cam_centroid = np.mean([c["C"] for c in cams], axis=0)
    center = np.array(args.center) if args.center else np.mean([c["C"] + c["look"] for c in cams], axis=0)
    floor_lo = args.floor_z + args.floor_margin
    z_min = args.z_min if args.z_min is not None else floor_lo + BOARD_H / 2   # board can stand up
    print(f"cameras {len(cams)} (session {session}) | center {center.round(3)} | "
          f"floor {args.floor_z} | z sample [{z_min:.2f},{args.z_max}]")

    from paradex.robot.robot_wrapper import RobotWrapper
    from paradex.robot.utils import get_robot_urdf_path
    urdf_path = get_robot_urdf_path(arm_name="franka")     # SAME urdf for IK and FK -> frames match
    rw = RobotWrapper(urdf_path)
    fk = yourdfpy.URDF.load(urdf_path, build_scene_graph=True)
    eef_from_board = np.eye(4); eef_from_board[2, 3] = -BOARD_Z_FROM_EEF

    # IK self-test: it MUST be able to recover home's own EEF pose. If this fails,
    # the problem is IK/link/frame, not reachability of the sampled poses.
    HOME = np.clip([0, -0.785, 0, -2.356, 0, 1.571, 0.785],
                   [j.limit.lower for j in fk.actuated_joints],
                   [j.limit.upper for j in fk.actuated_joints])
    fk.update_cfg(HOME)
    T_home = np.array(fk.get_transform(EEF_LINK, fk.base_link))
    _, ok_home = rw.solve_ik(T_home, EEF_LINK, max_iter=500, try_num=10)
    print(f"IK self-test (recover home {EEF_LINK}): {'OK' if ok_home else 'FAIL <- IK bug, not reachability'}",
          flush=True)

    def try_pose(T_board):
        if board_min_z(T_board) < floor_lo:
            return None
        q, ok = rw.solve_ik(T_board @ eef_from_board, EEF_LINK, max_iter=500, try_num=10)  # fast-ish
        # board vs the raised floor is checked separately (board_min_z); arm links only
        # need to clear the actual ground (base sits at z=0, below the raised platform).
        if ok and ph.links_above_floor(fk, q, -0.02):
            return q
        return None

    rng = np.random.default_rng(0)
    pool, tries = [], 0
    target = max(args.n * 4, 60)
    print(f"searching poses (min_obs={args.min_obs})...", flush=True)
    while len(pool) < target and tries < target * 150:
        tries += 1
        if tries % 100 == 0:
            print(f"  ... {len(pool)}/{target} found, {tries} tries", flush=True)
        pos = np.array([rng.uniform(args.x_min, center[0] + args.extent),   # x stays in front
                        center[1] + rng.uniform(-args.extent, args.extent),
                        rng.uniform(z_min, args.z_max)])
        T = flat_board_pose(pos, cam_centroid, args.tilt, rng)
        ng = score_flat(T, cams)
        if ng < args.min_obs:
            continue
        q = try_pose(T)
        if q is not None:
            pool.append((T, q, ng, []))
    chosen = ph.farthest_point_select(pool, args.n)
    # order into a continuous sweep: greedy nearest-neighbor in joint space from HOME,
    # so the arm moves in small steps (smooth path, low collision risk between waypoints).
    remaining = list(range(len(chosen)))
    order, cur = [], HOME
    while remaining:
        j = min(remaining, key=lambda i: np.linalg.norm(chosen[i][1] - cur))
        order.append(j); cur = chosen[j][1]; remaining.remove(j)
    chosen = [chosen[i] for i in order]
    jumps = [np.linalg.norm(chosen[i + 1][1] - chosen[i][1]) for i in range(len(chosen) - 1)]
    print(f"pool {len(pool)} ({tries} tries) -> selected {len(chosen)}, "
          f"ordered (max joint jump {max(jumps):.2f} rad)" if jumps else f"selected {len(chosen)}", flush=True)

    # INTERMEDIATE-path check: sample the straight-line joint move between consecutive
    # waypoints (incl HOME -> first) and check the board never dips below the floor.
    # This is a floor-collision CHECK only (no self/robot collision -> that needs curobo).
    def seg_board_min(qa, qb, steps=15):
        m = 1e9
        for t in np.linspace(0, 1, steps):
            fk.update_cfg((1 - t) * qa + t * qb)
            Tb = np.array(fk.get_transform(EEF_LINK, fk.base_link))
            Tb = Tb @ np.array([[1,0,0,0],[0,1,0,0],[0,0,1,BOARD_Z_FROM_EEF],[0,0,0,1]])
            m = min(m, board_min_z(Tb))
        return m
    seq = [HOME] + [q for _, q, _, _ in chosen]
    seg_mins = [seg_board_min(seq[i], seq[i + 1]) for i in range(len(seq) - 1)]
    worst = min(seg_mins)
    print(f"INTERMEDIATE-path board_min_z (straight-line moves): {worst:.3f} "
          f"(>= {floor_lo:.3f} => board clears floor en route)", flush=True)
    if worst < floor_lo:
        bad = [i for i, m in enumerate(seg_mins) if m < floor_lo]
        print(f"  WARNING: {len(bad)} transition(s) dip the board below floor (segments {bad}). "
              f"Use curobo, or add a retract via-point, before executing.", flush=True)
    if len(chosen) < args.n:
        print(f"  (got {len(chosen)} < {args.n}: try --min-obs 0, or lower --z-min / widen --extent)", flush=True)

    P = np.array([T[:3, 3] for T, _, _, _ in chosen])
    if len(P):
        print(f"position spread (m): x[{P[:,0].min():.2f},{P[:,0].max():.2f}] "
              f"y[{P[:,1].min():.2f},{P[:,1].max():.2f}] z[{P[:,2].min():.2f},{P[:,2].max():.2f}]")
        print(f"board_min_z over selected: {min(board_min_z(T) for T,_,_,_ in chosen):.3f} "
              f"(floor {args.floor_z} + margin {args.floor_margin} = {floor_lo:.3f}; must be >=)")

    if args.write:
        os.makedirs(OUT_DIR, exist_ok=True)
        for i, (T, q, ng, _) in enumerate(chosen):
            fk.update_cfg(q)
            np.save(os.path.join(OUT_DIR, f"{i}_qpos.npy"), q)
            np.save(os.path.join(OUT_DIR, f"{i}_pose.npy"),
                    np.array(fk.get_transform(EEF_LINK, fk.base_link)))
        print(f"wrote {len(chosen)} poses -> {OUT_DIR}")

    if args.no_viser:
        return
    from paradex.visualization.visualizer.viser import ViserViewer
    viewer = ViserViewer(port_number=args.port)
    viewer.add_grid(size=2.0, cell_size=0.1, height=args.floor_z)   # grid at the raised floor
    viewer.add_robot("franka", urdf_path)
    board_mesh = trimesh.creation.box(extents=[BOARD_W, BOARD_H, BOARD_T])
    board_mesh.visual.vertex_colors = [230, 230, 230, 255]
    viewer.add_trimesh("board", board_mesh, np.eye(4))
    for cam in cams:
        T = np.eye(4); T[:3, :3] = cam["Rwc"].T; T[:3, 3] = cam["C"]
        viewer.add_camera(cam["serial"], T, cam["K"], color=(0.0, 0.9, 0.3), size=0.05)
    qs = [q for _, q, _, _ in chosen]
    sl = viewer.server.gui.add_slider("pose #", min=0, max=max(len(qs) - 1, 0), step=1, initial_value=0)
    txt = viewer.server.gui.add_text("info", initial_value="")

    def show(_=None):
        i = int(sl.value)
        viewer.robot_dict["franka"].update_cfg(qs[i])
        fk.update_cfg(qs[i])
        Tb = np.array(fk.get_transform(EEF_LINK, fk.base_link)) @ np.linalg.inv(eef_from_board)
        h = viewer.obj_dict["board"]["frame"]
        h.position = Tb[:3, 3]
        h.wxyz = R.from_matrix(Tb[:3, :3]).as_quat()[[3, 0, 1, 2]]
        txt.value = f"pose {i}: obs={chosen[i][2]}, board_min_z={board_min_z(Tb):.3f}"
    sl.on_update(show)
    show()
    print("slider로 pose 넘기며 확인. 흰 판=보드, 초록=카메라. Ctrl-C 종료.")
    viewer.start_viewer(use_thread=False)


if __name__ == "__main__":
    main()
