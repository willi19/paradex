"""
Calibrate xArm `joint_tcp` offset from compare logs.

This script estimates a constant SE3 correction D such that:
    T_measured ~= T_fk @ D
where
    T_measured: pose from xArm get_position (after servo_cartesian_aa)
    T_fk:       URDF FK pose from observed joint state (servo_fk_pose_aa)

Then it updates `joint_tcp` origin by:
    T_tcp_new = T_tcp_old @ D^alpha

Usage:
  python src/validate/robot/calibrate_tcp_offset_from_logs.py --glob "xarm_compare_*.npz" --apply
  python src/validate/robot/calibrate_tcp_offset_from_logs.py --glob "xarm_compare_*.npz" --alpha 0.7
"""

import argparse
import glob
import os
import xml.etree.ElementTree as ET

import numpy as np
import transforms3d as t3d


def aa2h(aa):
    aa = np.asarray(aa, dtype=np.float64)
    h = np.eye(4, dtype=np.float64)
    h[:3, 3] = aa[:3] / 1000.0
    rv = aa[3:6]
    ang = np.linalg.norm(rv)
    if ang < 1e-12:
        h[:3, :3] = np.eye(3, dtype=np.float64)
    else:
        h[:3, :3] = t3d.axangles.axangle2mat(rv / ang, ang)
    return h


def h_to_xyz_rpy(h):
    xyz = h[:3, 3].copy()
    rpy = np.array(t3d.euler.mat2euler(h[:3, :3], axes="sxyz"), dtype=np.float64)
    return xyz, rpy


def rv_to_R(rv):
    rv = np.asarray(rv, dtype=np.float64)
    ang = np.linalg.norm(rv)
    if ang < 1e-12:
        return np.eye(3, dtype=np.float64)
    return t3d.axangles.axangle2mat(rv / ang, ang)


def parse_origin_xyz_rpy(s):
    return np.asarray([float(x) for x in s.strip().split()], dtype=np.float64)


def collect_deltas(npz_glob, max_pos_mm, max_rot_rad):
    paths = sorted(glob.glob(npz_glob))
    deltas_t = []
    deltas_rv = []
    used = []

    for p in paths:
        d = np.load(p)
        keys = set(d.files)
        if not {"after_servo_pose_aa", "servo_fk_pose_aa", "servo_fk_pos_err_mm", "servo_fk_rot_err_rad"}.issubset(keys):
            continue
        epos = float(d["servo_fk_pos_err_mm"])
        erot = float(d["servo_fk_rot_err_rad"])
        if not np.isfinite(epos) or not np.isfinite(erot):
            continue
        if epos > max_pos_mm or erot > max_rot_rad:
            continue

        measured = aa2h(d["after_servo_pose_aa"].astype(np.float64))
        fk = aa2h(d["servo_fk_pose_aa"].astype(np.float64))
        delta = np.linalg.inv(fk) @ measured

        t = delta[:3, 3]
        ax, ang = t3d.axangles.mat2axangle(delta[:3, :3])
        rv = np.asarray(ax, dtype=np.float64) * float(ang)

        deltas_t.append(t)
        deltas_rv.append(rv)
        used.append((p, epos, erot))

    return np.asarray(deltas_t), np.asarray(deltas_rv), used


def robust_center(arr):
    """
    Robust center:
    - median center
    - trim top 20% farthest samples
    - mean of remaining
    """
    if arr.shape[0] == 0:
        raise ValueError("No samples")
    med = np.median(arr, axis=0)
    if arr.shape[0] < 5:
        return med
    dist = np.linalg.norm(arr - med[None, :], axis=1)
    keep_n = max(3, int(np.ceil(arr.shape[0] * 0.8)))
    idx = np.argsort(dist)[:keep_n]
    return np.mean(arr[idx], axis=0)


def load_joint_tcp_origin(urdf_path):
    tree = ET.parse(urdf_path)
    root = tree.getroot()
    for joint in root.findall("joint"):
        if joint.attrib.get("name") == "joint_tcp":
            origin = joint.find("origin")
            if origin is None:
                raise RuntimeError("joint_tcp found but no <origin>")
            xyz = parse_origin_xyz_rpy(origin.attrib["xyz"])
            rpy = parse_origin_xyz_rpy(origin.attrib["rpy"])
            return tree, root, origin, xyz, rpy
    raise RuntimeError("joint_tcp not found in URDF")


def save_joint_tcp_origin(tree, origin_elem, xyz, rpy, urdf_path):
    origin_elem.attrib["xyz"] = f"{xyz[0]:.8f} {xyz[1]:.8f} {xyz[2]:.8f}"
    origin_elem.attrib["rpy"] = f"{rpy[0]:.8f} {rpy[1]:.8f} {rpy[2]:.8f}"
    tree.write(urdf_path, encoding="utf-8", xml_declaration=False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--glob", type=str, default="xarm_compare_*.npz")
    parser.add_argument("--urdf", type=str, default="rsc/robot/xarm/xarm.urdf")
    parser.add_argument("--alpha", type=float, default=0.8, help="Apply ratio of estimated delta [0..1]")
    parser.add_argument("--max-pos-mm", type=float, default=3.0, help="Use samples with servo_fk_pos_err_mm <= this")
    parser.add_argument("--max-rot-rad", type=float, default=0.03, help="Use samples with servo_fk_rot_err_rad <= this")
    parser.add_argument("--apply", action="store_true", help="Write updated joint_tcp origin to URDF")
    args = parser.parse_args()

    if args.alpha < 0.0 or args.alpha > 1.0:
        raise ValueError("--alpha must be in [0, 1]")

    dt, drv, used = collect_deltas(args.glob, args.max_pos_mm, args.max_rot_rad)
    print(f"matched files: {len(glob.glob(args.glob))}, used samples: {len(used)}")
    if len(used) == 0:
        raise RuntimeError("No usable samples. Relax thresholds or collect more data.")

    for p, epos, erot in used[-10:]:
        print(f"  {p} | pos_err={epos:.4f} mm rot_err={erot:.6f} rad")

    t_est = robust_center(dt)
    rv_est = robust_center(drv)

    print("\nEstimated delta (FK -> measured)")
    print("  trans(mm):", np.array2string(t_est * 1000.0, precision=4))
    print("  rotvec(deg):", np.array2string(np.rad2deg(rv_est), precision=4))

    tree, _, origin_elem, cur_xyz, cur_rpy = load_joint_tcp_origin(args.urdf)
    R_cur = t3d.euler.euler2mat(*cur_rpy, axes="sxyz")
    T_cur = np.eye(4, dtype=np.float64)
    T_cur[:3, :3] = R_cur
    T_cur[:3, 3] = cur_xyz

    T_delta = np.eye(4, dtype=np.float64)
    T_delta[:3, :3] = rv_to_R(rv_est * args.alpha)
    T_delta[:3, 3] = t_est * args.alpha

    T_new = T_cur @ T_delta
    new_xyz, new_rpy = h_to_xyz_rpy(T_new)

    print("\njoint_tcp current")
    print("  xyz:", np.array2string(cur_xyz, precision=8))
    print("  rpy:", np.array2string(cur_rpy, precision=8))
    print("joint_tcp proposed")
    print("  xyz:", np.array2string(new_xyz, precision=8))
    print("  rpy:", np.array2string(new_rpy, precision=8))

    if args.apply:
        backup = args.urdf + ".bak_tcp_calib"
        if not os.path.exists(backup):
            with open(args.urdf, "rb") as fsrc, open(backup, "wb") as fdst:
                fdst.write(fsrc.read())
            print(f"backup saved: {backup}")
        save_joint_tcp_origin(tree, origin_elem, new_xyz, new_rpy, args.urdf)
        print(f"updated: {args.urdf}")
    else:
        print("dry-run only. add --apply to write URDF.")


if __name__ == "__main__":
    main()

