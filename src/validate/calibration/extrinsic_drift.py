"""
Measure camera position drift in robot space across calibration sessions.

Each handeye calibration session contains:
  - C2R: COLMAP world frame -> robot frame (4x4)
  - extrinsics: per-camera world-to-camera [R|t] (3x4) in COLMAP frame

Since COLMAP's world frame is arbitrary per session, comparing C2R alone
is meaningless. Instead we compute each camera's pose in robot space:

    T_cam_in_world = inv(extrinsic)   # camera pose in COLMAP world
    T_cam_in_robot = C2R @ T_cam_in_world  # camera pose in robot space

and compare THOSE across sessions.

Usage:
    # Compare all sessions
    python extrinsic_drift.py

    # Compare specific sessions
    python extrinsic_drift.py --sessions 20260318_083843 20260326_062502

    # Use a specific session as reference (default: earliest)
    python extrinsic_drift.py --ref 20260318_083843
"""

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

sys.path.append(str(Path(__file__).resolve().parents[3]))
from paradex.utils.path import shared_dir

handeye_dir = os.path.join(shared_dir, "handeye_calibration")


def load_session(session_name):
    """Load C2R and per-camera extrinsics from a handeye calibration session.

    Convention (from handeye/calculate.py):
        C2R (saved as "robot_wrt_cam"): robot coord -> COLMAP world coord
        extrinsic [R|t] (3x4): COLMAP world coord -> camera coord

    So to get camera pose in robot space:
        T_cam_in_colmap = inv(extrinsic)          # camera pose in COLMAP world
        T_cam_in_robot  = inv(C2R) @ T_cam_in_colmap  # camera pose in robot space

    Returns:
        c2r: (4,4) robot -> COLMAP world
        cam_poses_robot: dict {serial: (4,4) camera pose in robot space}
    """
    session_dir = os.path.join(handeye_dir, session_name)
    indices = sorted(os.listdir(session_dir))
    base = os.path.join(session_dir, indices[0])

    c2r = np.load(os.path.join(base, "C2R.npy"))  # robot -> COLMAP world
    r2c_inv = np.linalg.inv(c2r)  # COLMAP world -> robot

    ext_path = os.path.join(base, "cam_param", "extrinsics.json")
    with open(ext_path) as f:
        ext_data = json.load(f)

    cam_poses_robot = {}
    for serial, vals in ext_data.items():
        # extrinsic [R|t]: COLMAP world -> camera (p_cam = R * p_world + t)
        ext_34 = np.array(vals).reshape(3, 4)
        R_wc = ext_34[:3, :3]
        t_wc = ext_34[:3, 3]

        # Camera pose in COLMAP world: inv(extrinsic)
        T_cam_colmap = np.eye(4)
        T_cam_colmap[:3, :3] = R_wc.T
        T_cam_colmap[:3, 3] = -R_wc.T @ t_wc

        # Camera pose in robot space
        T_cam_robot = r2c_inv @ T_cam_colmap
        cam_poses_robot[serial] = T_cam_robot

    return c2r, cam_poses_robot


def se3_diff(T_a, T_b):
    """Compute translation (m) and rotation (deg) difference."""
    T_rel = np.linalg.inv(T_a) @ T_b
    t_diff = np.linalg.norm(T_rel[:3, 3])
    cos_angle = np.clip((np.trace(T_rel[:3, :3]) - 1) / 2, -1.0, 1.0)
    r_diff_deg = np.degrees(np.arccos(cos_angle))
    return t_diff, r_diff_deg


def compare_camera_poses(sessions, ref_name=None):
    """Compare each camera's pose in robot space across sessions."""
    if not sessions:
        sessions = sorted(os.listdir(handeye_dir))
        sessions = [s for s in sessions
                    if os.path.isdir(os.path.join(handeye_dir, s))]

    # Load all sessions
    data = {}
    for s in sessions:
        try:
            _, cam_poses = load_session(s)
            data[s] = cam_poses
        except Exception as e:
            print(f"  WARNING: Could not load {s}: {e}")

    sessions = list(data.keys())
    if len(sessions) < 2:
        print("Need at least 2 valid sessions.")
        return

    ref = ref_name if ref_name and ref_name in data else sessions[0]

    # Common cameras
    all_serials = set(data[ref].keys())
    for s in sessions:
        all_serials &= set(data[s].keys())
    serials = sorted(all_serials)

    print(f"\n{'='*70}")
    print(f"  Camera Position Drift in Robot Space")
    print(f"  {len(sessions)} sessions, {len(serials)} common cameras")
    print(f"  Reference: {ref}")
    print(f"{'='*70}")

    # ── Per-camera table ──
    all_t_diffs = {serial: [] for serial in serials}
    all_r_diffs = {serial: [] for serial in serials}

    for serial in serials:
        T_ref = data[ref][serial]
        ref_pos = T_ref[:3, 3]
        print(f"\n  Camera {serial}  (ref pos: [{ref_pos[0]:.4f}, {ref_pos[1]:.4f}, {ref_pos[2]:.4f}] m)")
        print(f"    {'Session':<25} {'dt (mm)':>10} {'dR (deg)':>10} {'dx (mm)':>10} {'dy (mm)':>10} {'dz (mm)':>10}")
        print(f"    {'-'*25} {'-'*10} {'-'*10} {'-'*10} {'-'*10} {'-'*10}")

        for s in sessions:
            T = data[s][serial]
            t_diff, r_diff = se3_diff(T_ref, T)
            dt = (T[:3, 3] - ref_pos) * 1000
            t_mm = t_diff * 1000
            marker = " <-- ref" if s == ref else ""
            print(f"    {s:<25} {t_mm:>10.3f} {r_diff:>10.4f} {dt[0]:>10.3f} {dt[1]:>10.3f} {dt[2]:>10.3f}{marker}")
            if s != ref:
                all_t_diffs[serial].append(t_mm)
                all_r_diffs[serial].append(r_diff)

    # ── Overall summary ──
    print(f"\n{'='*70}")
    print(f"  Summary: Per-Camera Drift vs Reference ({ref})")
    print(f"{'='*70}")
    print(f"  {'Camera':<15} {'mean dt':>10} {'max dt':>10} {'std dt':>10} {'mean dR':>10} {'max dR':>10}")
    print(f"  {'':15} {'(mm)':>10} {'(mm)':>10} {'(mm)':>10} {'(deg)':>10} {'(deg)':>10}")
    print(f"  {'-'*15} {'-'*10} {'-'*10} {'-'*10} {'-'*10} {'-'*10}")

    global_t = []
    global_r = []
    for serial in serials:
        td = all_t_diffs[serial]
        rd = all_r_diffs[serial]
        if td:
            print(f"  {serial:<15} {np.mean(td):>10.3f} {np.max(td):>10.3f} {np.std(td):>10.3f} "
                  f"{np.mean(rd):>10.4f} {np.max(rd):>10.4f}")
            global_t.extend(td)
            global_r.extend(rd)

    if global_t:
        print(f"  {'-'*15} {'-'*10} {'-'*10} {'-'*10} {'-'*10} {'-'*10}")
        print(f"  {'ALL':15} {np.mean(global_t):>10.3f} {np.max(global_t):>10.3f} {np.std(global_t):>10.3f} "
              f"{np.mean(global_r):>10.4f} {np.max(global_r):>10.4f}")

    # ── Consecutive session drift ──
    if len(sessions) >= 3:
        print(f"\n{'='*70}")
        print(f"  Consecutive Session Drift (per-camera mean)")
        print(f"{'='*70}")
        print(f"  {'From -> To':<55} {'mean dt':>10} {'mean dR':>10}")
        print(f"  {'':55} {'(mm)':>10} {'(deg)':>10}")
        print(f"  {'-'*55} {'-'*10} {'-'*10}")

        for i in range(len(sessions) - 1):
            s_a, s_b = sessions[i], sessions[i + 1]
            t_list, r_list = [], []
            for serial in serials:
                t_diff, r_diff = se3_diff(data[s_a][serial], data[s_b][serial])
                t_list.append(t_diff * 1000)
                r_list.append(r_diff)
            print(f"  {s_a} -> {s_b:<25} {np.mean(t_list):>10.3f} {np.mean(r_list):>10.4f}")


def main():
    parser = argparse.ArgumentParser(
        description="Measure camera position drift in robot space across calibration sessions")
    parser.add_argument("--sessions", nargs="*", default=None,
                        help="Handeye calibration session names (default: all)")
    parser.add_argument("--ref", type=str, default=None,
                        help="Reference session name (default: earliest)")
    args = parser.parse_args()

    compare_camera_poses(args.sessions or [], ref_name=args.ref)


if __name__ == "__main__":
    main()
