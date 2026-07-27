"""Visualize the current camera rig + detected ChArUco board positions, and
compute the optimal ChArUco square size FOR THIS RIG from real geometry.

A hand-eye session is the cleanest source: one folder holds both the cameras
(``0/cam_param/{intrinsics,extrinsics}.json``) and the triangulated board
corners (``*/charuco_3d_corners.npy`` + ``*_ids.npy``) in the SAME metric frame.
Optionally everything is pushed into the robot base frame via ``0/C2R.npy``.

Two things it does:
  1. --analyze  (headless, no viser): from actual camera intrinsics and actual
     camera->board distances, report pixels-per-mm and the smallest ChArUco
     square that still (a) lets the 6x6 marker decode and (b) keeps checker
     corners sharp — at the WORST-placed camera, not just the average.
  2. (default)  open a Viser scene: camera frustums + board corner points.

Usage:
    python src/validate/calibration/visualize_rig.py --analyze
    python src/validate/calibration/visualize_rig.py --session 20260721_134538
    python src/validate/calibration/visualize_rig.py --to-robot        # robot frame
"""
import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

sys.path.append(str(Path(__file__).resolve().parents[3]))
from paradex.calibration.utils import handeye_calib_path
from paradex.utils.file_io import find_latest_directory
from paradex.image.aruco import boardinfo_dict

# --- board-size targets (why these numbers) ---
# A 6x6 ArUco marker is an 8x8 module grid (6 data + 2 border). Reliable decode
# wants ~3 px/module -> ~24 px across the marker. Checker subpixel corners want
# the square to span >= ~12 px. Both are evaluated at the worst-placed camera.
MARKER_DECODE_PX = 24.0
CHECKER_MIN_PX = 12.0
DEFAULT_RATIO = 0.75          # marker / checker (keep <=0.75 to avoid corner contamination)
CANDIDATE_MM = [25, 30, 40, 50, 60]

_PALETTE = [
    (0.90, 0.30, 0.30), (0.30, 0.70, 0.95), (0.35, 0.85, 0.45),
    (0.95, 0.75, 0.25), (0.75, 0.45, 0.90), (0.95, 0.55, 0.25),
]


def board_id_ranges():
    """Global-corner-id -> board mapping, matching merge_charuco_detection offsets."""
    ranges = {}
    off = 0
    for bid, cfg in boardinfo_dict.items():
        n = (cfg["numX"] - 1) * (cfg["numY"] - 1)
        ranges[bid] = (off, off + n)
        off += n
    return ranges


def board_of_id(gid, ranges):
    for bid, (lo, hi) in ranges.items():
        if lo <= gid < hi:
            return bid
    return None


def cam_center(ext_34):
    """World-space camera center from a world->cam [R|t] (3x4)."""
    R = ext_34[:3, :3]
    t = ext_34[:3, 3]
    return -R.T @ t


def world_from_cam(ext_34):
    """4x4 camera pose in world (inverse of the world->cam extrinsic)."""
    R = ext_34[:3, :3]
    t = ext_34[:3, 3]
    T = np.eye(4)
    T[:3, :3] = R.T
    T[:3, 3] = -R.T @ t
    return T


def load_session(session):
    """Load cameras + aggregated board corners from a hand-eye session.

    Returns intr(dict), ext(dict serial->3x4), board_pts(N,3), board_ids(N,), C2R or None.
    """
    root = os.path.join(handeye_calib_path, session)
    idxs = sorted(os.listdir(root), key=lambda s: (len(s), s))
    base = os.path.join(root, idxs[0])

    cp = os.path.join(base, "cam_param")
    intr_raw = json.load(open(os.path.join(cp, "intrinsics.json")))
    ext_raw = json.load(open(os.path.join(cp, "extrinsics.json")))
    intr = {s: {
        "K": np.array(v["intrinsics_undistort"]).reshape(3, 3),
        "width": v["width"], "height": v["height"],
    } for s, v in intr_raw.items()}
    ext = {s: np.array(v).reshape(3, 4) for s, v in ext_raw.items()}

    pts, ids = [], []
    for ix in idxs:
        pc = os.path.join(root, ix, "charuco_3d_corners.npy")
        pi = os.path.join(root, ix, "charuco_3d_ids.npy")
        if not (os.path.exists(pc) and os.path.exists(pi)):
            continue
        c = np.load(pc)
        i = np.load(pi)
        if len(c):
            pts.append(c)
            ids.append(i)
    board_pts = np.vstack(pts) if pts else np.zeros((0, 3))
    board_ids = np.concatenate(ids) if ids else np.zeros((0,), dtype=int)

    c2r_path = os.path.join(base, "C2R.npy")
    c2r = np.load(c2r_path) if os.path.exists(c2r_path) else None
    return intr, ext, board_pts, board_ids, c2r


def level_floor_transform(board_pts, board_ids, cam_centers, floor_board="1"):
    """4x4 that rotates the scene so the FLOOR board plane normal -> +z, floor at z=0.

    Fits a plane to the static floor board's corners (board "1", the id range
    handeye/calculate.py excludes as static). Normal is flipped to point toward the
    cameras (up). Rotation about z is left free (only "make floor horizontal").
    """
    ranges = board_id_ranges()
    rng = ranges.get(floor_board)
    fp = board_pts
    if rng is not None:
        mask = (board_ids >= rng[0]) & (board_ids < rng[1])
        if mask.sum() >= 3:
            fp = board_pts[mask]
        else:
            print(f"floor board {floor_board!r} has <3 corners; fitting plane to ALL board corners.")
    if len(fp) < 3:
        print("not enough board corners to fit a floor plane; leaving frame unchanged.")
        return np.eye(4)

    c = fp.mean(0)
    _, _, vt = np.linalg.svd(fp - c)
    normal = vt[-1]
    if cam_centers is not None and len(cam_centers):
        if np.dot(cam_centers.mean(0) - c, normal) < 0:
            normal = -normal

    z = np.array([0.0, 0.0, 1.0])
    v = np.cross(normal, z)
    s = np.linalg.norm(v)
    cth = np.dot(normal, z)
    if s < 1e-9:
        R = np.eye(3) if cth > 0 else np.diag([1.0, -1.0, -1.0])
    else:
        vx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
        R = np.eye(3) + vx + vx @ vx * ((1 - cth) / (s * s))
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = -R @ c          # floor centroid -> origin, normal -> +z
    return T


def analyze(intr, ext, board_pts, ratio):
    """Print pixels-per-mm and the smallest usable ChArUco square, from real geometry."""
    if len(board_pts) == 0:
        print("No board corners in this session — cannot analyze distances.")
        return
    board_c = board_pts.mean(0)

    rows = []  # (serial, dist_m, fx, ppm)
    for s, e in ext.items():
        if s not in intr:
            continue
        c = cam_center(e)
        d = np.linalg.norm(c - board_c)
        fx = intr[s]["K"][0, 0]
        ppm = fx / (d * 1000.0)  # pixels per mm at the board plane
        rows.append((s, d, fx, ppm))
    rows.sort(key=lambda r: r[3])  # worst (lowest ppm) first

    ppms = np.array([r[3] for r in rows])
    worst = rows[0]
    dists = np.array([r[1] for r in rows])

    print("=" * 74)
    print(f"RIG BOARD-SIZE ANALYSIS  ({len(rows)} cameras, marker ratio {ratio})")
    print("=" * 74)
    print(f"camera->board distance (m): min {dists.min():.2f}  median {np.median(dists):.2f}  max {dists.max():.2f}")
    print(f"pixels/mm at board:         min {ppms.min():.2f}  median {np.median(ppms):.2f}  max {ppms.max():.2f}")
    print(f"worst-placed camera: {worst[0]}  dist {worst[1]:.2f} m  fx {worst[2]:.0f}px  -> {worst[3]:.2f} px/mm")
    print()

    # smallest square that satisfies both constraints at the worst camera
    ppm_w = worst[3]
    s_marker = MARKER_DECODE_PX / (ratio * ppm_w)
    s_checker = CHECKER_MIN_PX / ppm_w
    s_rec = max(s_marker, s_checker)
    print(f"min square for 6x6 marker decode ({MARKER_DECODE_PX:.0f}px) at worst cam: {s_marker:5.1f} mm")
    print(f"min square for sharp checker    ({CHECKER_MIN_PX:.0f}px) at worst cam: {s_checker:5.1f} mm")
    print(f">>> recommended MINIMUM checker (worst-case safe): {np.ceil(s_rec):.0f} mm")
    print(f"    (current rig boards use 60 mm -> plenty of margin; smaller = more corners)")
    print()

    print("candidate squares — marker px seen at worst / median camera (need >= "
          f"{MARKER_DECODE_PX:.0f} to decode):")
    ppm_med = np.median(ppms)
    print(f"  {'square':>7} {'marker mm':>10} {'worst px':>9} {'median px':>10}   verdict")
    for mm in CANDIDATE_MM:
        mk = ratio * mm
        pw = mk * ppm_w
        pm = mk * ppm_med
        ok = "OK" if pw >= MARKER_DECODE_PX else "RISKY at far cams"
        print(f"  {mm:>5}mm {mk:>9.1f} {pw:>9.1f} {pm:>10.1f}   {ok}")
    print("=" * 74)


def scene_transform(board_pts, board_ids, ext, c2r, mode):
    """Pick the colmap-world -> display transform.

    mode: 'auto' (default) uses robot frame via C2R (robot base z ⊥ floor, so the
    floor comes out level); if the session has no C2R it falls back to fitting the
    floor board plane. 'robot' / 'level' / 'colmap' force one explicitly.
    """
    def robot():
        return np.linalg.inv(c2r), "robot (floor level via C2R)"  # C2R: robot->colmap world

    def level():
        cams = np.array([cam_center(e) for e in ext.values()])
        return level_floor_transform(board_pts, board_ids, cams), "floor-leveled (plane fit)"

    if mode == "colmap":
        return np.eye(4), "colmap-world"
    if mode == "robot":
        if c2r is None:
            raise SystemExit("--robot requested but session has no C2R.npy")
        return robot()
    if mode == "level":
        return level()
    # auto (default): robot frame if C2R, else fit the floor plane
    if c2r is not None:
        return robot()
    print("no C2R.npy in this session -> falling back to floor-plane fit (--level).")
    return level()


def visualize(intr, ext, board_pts, board_ids, c2r, mode, port):
    from paradex.visualization.visualizer.viser import ViserViewer  # lazy: viser needed only here

    T, frame_name = scene_transform(board_pts, board_ids, ext, c2r, mode)

    viewer = ViserViewer(port_number=port)
    viewer.add_grid(size=4.0, cell_size=0.1, height=0.0)

    for s, e in ext.items():
        wfc = T @ world_from_cam(e)
        viewer.add_camera(s, wfc, intr[s]["K"], color=(0.0, 0.9, 0.3), size=0.08)

    if len(board_pts):
        pts = (T[:3, :3] @ board_pts.T).T + T[:3, 3]
        ranges = board_id_ranges()
        board_keys = list(boardinfo_dict.keys())
        colors = np.zeros((len(pts), 3))
        for k, gid in enumerate(board_ids):
            bid = board_of_id(int(gid), ranges)
            ci = board_keys.index(bid) % len(_PALETTE) if bid in board_keys else 0
            colors[k] = _PALETTE[ci]
        try:
            viewer.server.scene.add_point_cloud(
                "/boards", points=pts.astype(np.float32),
                colors=(colors * 255).astype(np.uint8), point_size=0.006,
            )
        except Exception:
            for k, p in enumerate(pts):  # fallback
                viewer.add_sphere(f"board_{k}", p, radius=0.004, color=tuple(colors[k]))
        print(f"drew {len(pts)} board corners "
              f"(boards present: {sorted(set(board_of_id(int(g), ranges) for g in board_ids))})")

    print(f"cameras: {len(ext)} | frame: {frame_name}")
    print(f"open the Viser URL it prints below. Ctrl-C to quit.")
    viewer.start_viewer(use_thread=False)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--session", default=None, help="handeye session (default: latest)")
    ap.add_argument("--analyze", action="store_true", help="headless board-size analysis only")
    ap.add_argument("--frame", choices=["auto", "robot", "level", "colmap"], default="auto",
                    help="display frame. auto(default)=robot via C2R, floor-plane fit if no C2R; "
                         "robot=force C2R; level=fit floor board plane; colmap=raw")
    ap.add_argument("--ratio", type=float, default=DEFAULT_RATIO, help="marker/checker ratio")
    ap.add_argument("--port", type=int, default=8080)
    args = ap.parse_args()

    session = args.session or find_latest_directory(handeye_calib_path)
    print(f"session: {session}")
    intr, ext, board_pts, board_ids, c2r = load_session(session)
    print(f"loaded {len(ext)} cameras, {len(board_pts)} board corners, C2R={'yes' if c2r is not None else 'no'}")

    if args.analyze:
        analyze(intr, ext, board_pts, args.ratio)
    else:
        analyze(intr, ext, board_pts, args.ratio)  # always print the numbers too
        visualize(intr, ext, board_pts, board_ids, c2r, args.frame, args.port)


if __name__ == "__main__":
    main()
