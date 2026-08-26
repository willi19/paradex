"""Auto-plan hand-eye cube poses that the cameras actually see well, then show them.

For a cube pose (robot frame) we score how many (camera, charuco-face) pairs are a
GOOD observation: face turned toward the camera (incidence < MAX_INCIDENCE), projects
inside the image, distance in range, and — using the camera's own fx — the 30 mm
marker still resolves above MARKER_MIN_PX (foreshortened by incidence). We sample cube
poses around the camera look-center, keep the reachable ones (Franka IK on fr3_link8),
and greedily pick a set that both scores high and spreads the cube orientation (what
hand-eye AX=XB needs). Then a Viser slider steps through them for you to review.

Cameras come from a hand-eye session's C2R (robot frame). That C2R is the xarm's, so
for the franka it's APPROXIMATE — override the sampling center with --center once the
franka is mounted, or just eyeball the reachable/visible overlap here.

    python src/calibration/handeye/plan_handeye_poses.py            # plan + viser review
    python src/calibration/handeye/plan_handeye_poses.py --n 20 --write   # save to hecalib/franka
    python src/calibration/handeye/plan_handeye_poses.py --center 0.5 0 0.5 --extent 0.15
"""
import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import yourdfpy
from scipy.spatial.transform import Rotation as R

sys.path.append(str(Path(__file__).resolve().parents[3]))
from paradex.utils.system import config_dir
from paradex.calibration.utils import handeye_calib_path

REPO = Path(__file__).resolve().parents[3]
URDF = str(REPO / "rsc/robot/franka/franka_adapter.urdf")
EEF_LINK = "fr3_link8"
CUBE_Z_FROM_EEF = 0.196          # cube center along flange +z
HALF = 0.15                      # cube half-size
TILT_DEG = 45.0                  # how far the hanging cube may tilt off vertical
                                 # (bigger = faces point in more varied directions; better AX=XB)
MARKER_MM = 30.0
FACE_AXES = [np.array(a) for a in ([1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0])]  # charuco faces
OUT_DIR = os.path.join(config_dir, "hecalib", "franka")

# visibility gates
MARKER_MIN_PX = 24.0
MAX_INCIDENCE = 65.0             # deg; face too edge-on -> skip
DIST_RANGE = (0.4, 2.0)         # m


def load_cameras(session):
    base = os.path.join(handeye_calib_path, session, "0")
    inv_c2r = np.linalg.inv(np.load(os.path.join(base, "C2R.npy")))
    ext = json.load(open(os.path.join(base, "cam_param", "extrinsics.json")))
    intr = json.load(open(os.path.join(base, "cam_param", "intrinsics.json")))
    cams = []
    for s, v in ext.items():
        if s not in intr:
            continue
        E = np.array(v).reshape(3, 4)
        wfc = np.eye(4)
        wfc[:3, :3] = E[:3, :3].T
        wfc[:3, 3] = -E[:3, :3].T @ E[:3, 3]
        T = inv_c2r @ wfc                       # camera pose in robot frame
        K = np.array(intr[s]["intrinsics_undistort"]).reshape(3, 3)
        # world(robot)->cam for this frame:
        Rwc = T[:3, :3].T
        twc = -Rwc @ T[:3, 3]
        cams.append({"serial": s, "K": K, "Rwc": Rwc, "twc": twc, "C": T[:3, 3],
                     "w": intr[s]["width"], "h": intr[s]["height"], "fx": K[0, 0],
                     "look": T[:3, :3] @ np.array([0, 0, 1.0])})
    return cams


def score_pose(T_cube, cams):
    """Return (n_good_obs, list of world face-normals that were well-seen)."""
    n_good = 0
    seen_normals = []
    Rc = T_cube[:3, :3]
    p = T_cube[:3, 3]
    for a in FACE_AXES:
        nrm = Rc @ a
        c = p + HALF * nrm
        face_seen = False
        for cam in cams:
            d = c - cam["C"]
            dist = np.linalg.norm(d)
            if not (DIST_RANGE[0] <= dist <= DIST_RANGE[1]):
                continue
            ray = d / dist
            cos_i = float(np.dot(-ray, nrm))            # 1 = head-on
            if cos_i < np.cos(np.radians(MAX_INCIDENCE)):
                continue
            pc = cam["Rwc"] @ c + cam["twc"]
            if pc[2] <= 0:
                continue
            uv = cam["K"] @ (pc / pc[2])
            if not (0 <= uv[0] < cam["w"] and 0 <= uv[1] < cam["h"]):
                continue
            marker_px = MARKER_MM * (cam["fx"] / (dist * 1000.0)) * cos_i
            if marker_px >= MARKER_MIN_PX:
                n_good += 1
                face_seen = True
        if face_seen:
            seen_normals.append(nrm)
    return n_good, seen_normals


_CUBE_CORNERS = np.array([[sx, sy, sz] for sx in (-HALF, HALF)
                          for sy in (-HALF, HALF) for sz in (-HALF, HALF)])


def cube_min_z(T_cube):
    """Lowest z of the cube's 8 corners in robot frame."""
    corners = (T_cube[:3, :3] @ _CUBE_CORNERS.T).T + T_cube[:3, 3]
    return corners[:, 2].min()


def links_above_floor(fk_urdf, q, floor_z):
    """True if every link origin is at/above the floor at config q."""
    fk_urdf.update_cfg(q)
    base = fk_urdf.base_link
    for ln in fk_urdf.link_map:
        if fk_urdf.get_transform(ln, base)[2, 3] < floor_z:
            return False
    return True


def random_hanging_pose(center_xy, xy_extent, z_range, rng):
    """Cube hangs bottom-down (random yaw + small tilt). xy jittered around center_xy;
    z drawn uniformly over z_range so positions reach all the way down to the floor."""
    down = R.from_euler("x", 180, degrees=True)
    rot = (R.from_euler("z", rng.uniform(0, 360), degrees=True)
           * down
           * R.from_euler(rng.choice(["x", "y"]), rng.uniform(-TILT_DEG, TILT_DEG), degrees=True))
    T = np.eye(4)
    T[:3, :3] = rot.as_matrix()
    T[:3, 3] = [center_xy[0] + rng.uniform(-xy_extent, xy_extent),
                center_xy[1] + rng.uniform(-xy_extent, xy_extent),
                rng.uniform(z_range[0], z_range[1])]
    return T


def sample_poses(center, extent, n_pos, n_rot):
    """Grid of positions x diverse orientations."""
    g = np.linspace(-extent, extent, n_pos)
    positions = [center + np.array([x, y, z]) for x in g for y in g for z in g]
    # orientations: cube hangs "bottom face down" (local +Z -> world -Z), charuco side
    # faces point horizontally at the ring of cameras. Vary yaw + small tilts only
    # (tilts give AX=XB the rotation-axis diversity that pure yaw can't).
    down = R.from_euler("x", 180, degrees=True)          # local +Z -> world -Z
    tilts = [R.identity()]
    for ax in "xy":
        for t in (TILT_DEG, -TILT_DEG):
            tilts.append(R.from_euler(ax, t, degrees=True))
    n_yaw = max(1, n_rot // len(tilts))
    base = [R.from_euler("z", yaw, degrees=True) * down * tl
            for yaw in np.linspace(0, 360, n_yaw, endpoint=False) for tl in tilts]
    poses = []
    for pos in positions:
        for rot in base:
            T = np.eye(4)
            T[:3, :3] = rot.as_matrix()
            T[:3, 3] = pos
            poses.append(T)
    return poses


def pose_dist(Ta, Tb, t_scale=0.3):
    """Distance between two cube poses, translation and rotation normalized to ~[0,1]
    so farthest-point sampling spreads BOTH (not just rotation)."""
    td = np.linalg.norm(Ta[:3, 3] - Tb[:3, 3]) / t_scale        # ~full workspace span -> 1
    Rrel = Ta[:3, :3].T @ Tb[:3, :3]
    ang = np.arccos(np.clip((np.trace(Rrel) - 1) / 2, -1.0, 1.0)) / np.pi   # 0..1
    return td + ang


def farthest_point_select(pool, n):
    """Greedily pick n poses whose [R,t] are maximally spread (farthest-point sampling)."""
    if not pool:
        return []
    sel = [0]
    mind = [pose_dist(pool[0][0], pool[i][0]) for i in range(len(pool))]
    while len(sel) < n and len(sel) < len(pool):
        nxt = int(np.argmax(mind))
        sel.append(nxt)
        for i in range(len(pool)):
            mind[i] = min(mind[i], pose_dist(pool[nxt][0], pool[i][0]))
    return [pool[i] for i in sel]


def greedy_select(scored, n):
    """scored: list of (T_cube, qpos, n_good, seen_normals). Pick n covering diverse normals."""
    chosen = []
    covered = []  # list of unit normals already covered
    remaining = list(range(len(scored)))
    while remaining and len(chosen) < n:
        best_idx, best_gain = None, -1.0
        for idx in remaining:
            item = scored[idx]
            gain = 0.0
            for nrm in item[3]:
                if all(np.dot(nrm, c) < 0.85 for c in covered):  # not yet covered (within ~30deg)
                    gain += 1
            gain += item[2] * 0.01                                # tiebreak by raw score
            if gain > best_gain:
                best_idx, best_gain = idx, gain
        chosen.append(scored[best_idx])
        covered.extend(scored[best_idx][3])
        remaining.remove(best_idx)
    return chosen


def main():
    global TILT_DEG
    ap = argparse.ArgumentParser()
    ap.add_argument("--cam-session", default=None)
    ap.add_argument("--n", type=int, default=18)
    ap.add_argument("--center", nargs=3, type=float, default=None, help="cube sampling center (robot frame)")
    ap.add_argument("--extent", type=float, default=0.30, help="cube xy position jitter half-range (m)")
    ap.add_argument("--z-min", type=float, default=None,
                    help="lowest cube-center z to sample (default: floor + cube half -> cube nearly on the floor)")
    ap.add_argument("--z-max", type=float, default=0.45, help="highest cube-center z to sample (m)")
    ap.add_argument("--min-obs", type=int, default=4,
                    help="random mode: keep a pose only if the cube is this well-seen (good camera-face obs)")
    ap.add_argument("--n-pos", type=int, default=3, help="grid points per axis")
    ap.add_argument("--n-rot", type=int, default=24)
    ap.add_argument("--random", action="store_true",
                    help="just sample random hanging poses (skip camera-visibility scoring)")
    ap.add_argument("--tilt", type=float, default=TILT_DEG,
                    help="max cube tilt off vertical, deg (bottom face stays within this of straight-down)")
    ap.add_argument("--ik-top", type=int, default=150, help="IK only the top-scored candidates")
    ap.add_argument("--floor-z", type=float, default=0.0, help="floor height in robot frame")
    ap.add_argument("--floor-margin", type=float, default=0.05, help="keep cube this far above floor")
    ap.add_argument("--write", action="store_true")
    ap.add_argument("--no-viser", action="store_true")
    ap.add_argument("--port", type=int, default=8080)
    args = ap.parse_args()
    TILT_DEG = args.tilt

    session = args.cam_session
    if session is None:
        for s in sorted(os.listdir(handeye_calib_path), reverse=True):
            if os.path.exists(os.path.join(handeye_calib_path, s, "0", "C2R.npy")):
                session = s
                break
    cams = load_cameras(session)
    print(f"cameras: {len(cams)} (session {session})")

    center = (np.array(args.center) if args.center
              else np.mean([c["C"] + c["look"] * 1.0 for c in cams], axis=0))
    print(f"sampling center (robot frame): {center.round(3)}  extent {args.extent}")

    floor_lo = args.floor_z + args.floor_margin

    # IK setup (both modes; needs pinocchio)
    from paradex.robot.robot_wrapper import RobotWrapper
    from paradex.robot.utils import get_robot_urdf_path
    rw = RobotWrapper(get_robot_urdf_path(arm_name="franka"))
    fk_urdf = yourdfpy.URDF.load(URDF, build_scene_graph=True)   # link floor check + eef pose
    cube_from_eef = np.eye(4)
    cube_from_eef[2, 3] = CUBE_Z_FROM_EEF
    eef_from_cube = np.linalg.inv(cube_from_eef)

    def ik_floor(T_cube):
        if cube_min_z(T_cube) < floor_lo:
            return None
        # no warm-start: let solve_ik random-restart so arm configs stay diverse (no one-side lean)
        q, ok = rw.solve_ik(T_cube @ eef_from_cube, EEF_LINK)
        if ok and links_above_floor(fk_urdf, q, args.floor_z):
            return q
        return None

    if args.random:
        z_min = args.z_min if args.z_min is not None else floor_lo + HALF
        z_range = (z_min, args.z_max)
        print(f"cube-center z sampled over {z_range} (cube bottom reaches ~floor at z_min)")
        rng = np.random.default_rng(0)
        pool, tries = [], 0
        pool_target = max(args.n * 6, 120)             # oversample, then farthest-point select
        while len(pool) < pool_target and tries < pool_target * 200:
            tries += 1
            T = random_hanging_pose(center[:2], args.extent, z_range, rng)
            ng, seen = score_pose(T, cams)
            if ng < args.min_obs:
                continue
            q = ik_floor(T)
            if q is not None:
                pool.append((T, q, ng, seen))
        chosen = farthest_point_select(pool, args.n)
        print(f"random: pool {len(pool)} reachable+visible ({tries} tries) "
              f"-> farthest-point selected {len(chosen)} for max [R,t] diversity")
    else:
        cand = sample_poses(center, args.extent, args.n_pos, args.n_rot)
        scored_vis = [(T, *score_pose(T, cams)) for T in cand if cube_min_z(T) >= floor_lo]
        scored_vis = [x for x in scored_vis if x[1] > 0]
        scored_vis.sort(key=lambda x: -x[1])
        print(f"candidates {len(cand)} -> {len(scored_vis)} visible & floor-safe; IK top {args.ik_top}")
        reachable = []
        for T_cube, n_good, seen in scored_vis[:args.ik_top]:
            q = ik_floor(T_cube)
            if q is not None:
                reachable.append((T_cube, q, n_good, seen))
        print(f"reachable & floor-safe: {len(reachable)}")
        chosen = greedy_select(reachable, args.n)
    print(f"\nselected {len(chosen)} poses:")
    print(f"  {'#':>2} {'good_obs':>8} {'faces_seen':>10}  cube_pos")
    for i, (T, q, ng, seen) in enumerate(chosen):
        pos = T[:3, 3].round(3) if T is not None else "(random)"
        print(f"  {i:>2} {ng:>8} {len(seen):>10}  {pos}")

    P = np.array([T[:3, 3] for T, _, _, _ in chosen if T is not None])
    if len(P):
        print(f"\nselected position spread (m): "
              f"x {np.ptp(P[:,0]):.2f} [{P[:,0].min():.2f},{P[:,0].max():.2f}]  "
              f"y {np.ptp(P[:,1]):.2f} [{P[:,1].min():.2f},{P[:,1].max():.2f}]  "
              f"z {np.ptp(P[:,2]):.2f} [{P[:,2].min():.2f},{P[:,2].max():.2f}]")
        print(f"  mean {P.mean(0).round(3)}  (vs sampling center {center.round(3)}; "
              f"big y offset from center => IK reaches one side better)")

    if args.write:
        os.makedirs(OUT_DIR, exist_ok=True)
        for old in os.listdir(OUT_DIR):        # clear stale waypoints (else old runs mix in)
            if old.endswith(("_qpos.npy", "_pose.npy")):
                os.remove(os.path.join(OUT_DIR, old))
        for i, (T, q, ng, seen) in enumerate(chosen):
            fk_urdf.update_cfg(q)
            ee = np.array(fk_urdf.get_transform(EEF_LINK, fk_urdf.base_link))
            np.save(os.path.join(OUT_DIR, f"{i}_qpos.npy"), q)
            np.save(os.path.join(OUT_DIR, f"{i}_pose.npy"), ee)
        print(f"wrote {len(chosen)} poses -> {OUT_DIR}")

    if args.no_viser:
        return
    from paradex.visualization.visualizer.viser import ViserViewer
    viewer = ViserViewer(port_number=args.port)
    viewer.add_grid(size=2.0, cell_size=0.1, height=0.0)
    viewer.add_robot("franka", URDF)
    for cam in cams:
        T = np.eye(4)
        T[:3, :3] = cam["Rwc"].T
        T[:3, 3] = cam["C"]
        viewer.add_camera(cam["serial"], T, cam["K"], color=(0.0, 0.9, 0.3), size=0.05)
    robot = viewer.robot_dict["franka"]
    qs = [q for _, q, _, _ in chosen]
    sl = viewer.server.gui.add_slider("pose #", min=0, max=max(len(qs) - 1, 0), step=1, initial_value=0)
    txt = viewer.server.gui.add_text("info", initial_value="")

    def show(_=None):
        i = int(sl.value)
        robot.update_cfg(qs[i])
        txt.value = f"pose {i}: good_obs={chosen[i][2]}, faces={len(chosen[i][3])}"
    sl.on_update(show)
    show()
    print("step through poses with the slider. Ctrl-C to quit.")
    viewer.start_viewer(use_thread=False)


if __name__ == "__main__":
    main()
