"""
Calibrate `joint_tcp` from compare logs and report before/after error.

This script estimates a constant SE3 delta D from pairs:
  T_fk (URDF FK pose from servo_fk_pose_aa)
  T_measured (xArm pose from after_servo_pose_aa)
using D ~= inv(T_fk) @ T_measured.

Then it reports expected error change if D^alpha is applied:
  T_fk_corrected = T_fk @ D^alpha

Usage:
  # dry-run (recommended first)
  python src/validate/robot/calibrate_tcp_offset_with_report.py \
    --glob "xarm_compare_*.npz" --alpha 0.8

  # write to URDF
  python src/validate/robot/calibrate_tcp_offset_with_report.py \
    --glob "xarm_compare_*.npz" --alpha 0.8 --apply
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
    rpy = np.asarray(t3d.euler.mat2euler(h[:3, :3], axes="sxyz"), dtype=np.float64)
    return xyz, rpy


def rv_to_R(rv):
    rv = np.asarray(rv, dtype=np.float64)
    ang = np.linalg.norm(rv)
    if ang < 1e-12:
        return np.eye(3, dtype=np.float64)
    return t3d.axangles.axangle2mat(rv / ang, ang)


def pose_err(tgt, cur):
    pos_mm = np.linalg.norm((cur[:3, 3] - tgt[:3, 3]) * 1000.0)
    dR = tgt[:3, :3].T @ cur[:3, :3]
    tr = np.clip((np.trace(dR) - 1.0) * 0.5, -1.0, 1.0)
    rot = np.arccos(tr)
    return float(pos_mm), float(rot)


def parse_xyz_rpy(text):
    return np.asarray([float(x) for x in text.strip().split()], dtype=np.float64)


def robust_center(arr):
    if arr.shape[0] == 0:
        raise ValueError("No samples")
    med = np.median(arr, axis=0)
    if arr.shape[0] < 5:
        return med
    dist = np.linalg.norm(arr - med[None, :], axis=1)
    keep_n = max(3, int(np.ceil(arr.shape[0] * 0.8)))
    idx = np.argsort(dist)[:keep_n]
    return np.mean(arr[idx], axis=0)


def collect_samples(npz_glob, max_pos_mm, max_rot_rad):
    paths = sorted(glob.glob(npz_glob))
    samples = []
    for p in paths:
        d = np.load(p)
        keys = set(d.files)
        needed = {"after_servo_pose_aa", "servo_fk_pose_aa", "servo_fk_pos_err_mm", "servo_fk_rot_err_rad"}
        if not needed.issubset(keys):
            continue

        # Support both per-run shape (6,) and aggregate shape (N,6)
        m = np.asarray(d["after_servo_pose_aa"], dtype=np.float64)
        k = np.asarray(d["servo_fk_pose_aa"], dtype=np.float64)
        epos = np.asarray(d["servo_fk_pos_err_mm"], dtype=np.float64)
        erot = np.asarray(d["servo_fk_rot_err_rad"], dtype=np.float64)

        if m.ndim == 1:
            m = m[None, :]
            k = k[None, :]
            epos = np.asarray([float(epos)], dtype=np.float64)
            erot = np.asarray([float(erot)], dtype=np.float64)

        for i in range(m.shape[0]):
            if i >= k.shape[0] or i >= epos.shape[0] or i >= erot.shape[0]:
                break
            if np.any(~np.isfinite(m[i])) or np.any(~np.isfinite(k[i])):
                continue
            if not np.isfinite(epos[i]) or not np.isfinite(erot[i]):
                continue
            if float(epos[i]) > max_pos_mm or float(erot[i]) > max_rot_rad:
                continue
            Tm = aa2h(m[i])
            Tk = aa2h(k[i])
            samples.append((p, i, Tm, Tk, float(epos[i]), float(erot[i])))

    return samples


def load_joint_tcp_origin(urdf_path):
    tree = ET.parse(urdf_path)
    root = tree.getroot()
    for joint in root.findall("joint"):
        if joint.attrib.get("name") == "joint_tcp":
            origin = joint.find("origin")
            if origin is None:
                raise RuntimeError("joint_tcp found but no <origin>")
            xyz = parse_xyz_rpy(origin.attrib["xyz"])
            rpy = parse_xyz_rpy(origin.attrib["rpy"])
            return tree, origin, xyz, rpy
    raise RuntimeError("joint_tcp not found in URDF")


def save_joint_tcp_origin(tree, origin_elem, xyz, rpy, urdf_path):
    origin_elem.attrib["xyz"] = f"{xyz[0]:.8f} {xyz[1]:.8f} {xyz[2]:.8f}"
    origin_elem.attrib["rpy"] = f"{rpy[0]:.8f} {rpy[1]:.8f} {rpy[2]:.8f}"
    tree.write(urdf_path, encoding="utf-8", xml_declaration=False)


def summarize(name, x):
    x = np.asarray(x, dtype=np.float64)
    return (
        f"{name}: mean={np.mean(x):.4f}, median={np.median(x):.4f}, "
        f"p95={np.percentile(x,95):.4f}, max={np.max(x):.4f}"
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--glob", type=str, default="xarm_compare_*.npz")
    parser.add_argument("--urdf", type=str, default="rsc/robot/xarm/xarm.urdf")
    parser.add_argument("--alpha", type=float, default=0.8, help="Apply ratio in [0,1]")
    parser.add_argument("--max-pos-mm", type=float, default=3.0)
    parser.add_argument("--max-rot-rad", type=float, default=0.03)
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()

    if not (0.0 <= args.alpha <= 1.0):
        raise ValueError("--alpha must be in [0, 1]")

    samples = collect_samples(args.glob, args.max_pos_mm, args.max_rot_rad)
    print(f"matched files: {len(glob.glob(args.glob))}, used samples: {len(samples)}")
    if len(samples) == 0:
        raise RuntimeError("No usable samples. Relax thresholds or collect more data.")

    # Build delta set
    deltas_t = []
    deltas_rv = []
    before_pos = []
    before_rot = []
    for p, i, Tm, Tk, _, _ in samples:
        D = np.linalg.inv(Tk) @ Tm
        deltas_t.append(D[:3, 3])
        ax, ang = t3d.axangles.mat2axangle(D[:3, :3])
        deltas_rv.append(np.asarray(ax, dtype=np.float64) * float(ang))
        pe, re = pose_err(Tm, Tk)
        before_pos.append(pe)
        before_rot.append(re)

    deltas_t = np.asarray(deltas_t)
    deltas_rv = np.asarray(deltas_rv)

    t_est = robust_center(deltas_t)
    rv_est = robust_center(deltas_rv)

    # Build D^alpha
    D_alpha = np.eye(4, dtype=np.float64)
    D_alpha[:3, :3] = rv_to_R(rv_est * args.alpha)
    D_alpha[:3, 3] = t_est * args.alpha

    # Expected after errors
    after_pos = []
    after_rot = []
    for _, _, Tm, Tk, _, _ in samples:
        Tk_corr = Tk @ D_alpha
        pe, re = pose_err(Tm, Tk_corr)
        after_pos.append(pe)
        after_rot.append(re)

    print("\nEstimated delta (FK -> measured)")
    print("  trans(mm):", np.array2string(t_est * 1000.0, precision=4))
    print("  rotvec(deg):", np.array2string(np.rad2deg(rv_est), precision=4))

    print("\nError report")
    print("  " + summarize("Before pos_err_mm", before_pos))
    print("  " + summarize("After  pos_err_mm", after_pos))
    print("  " + summarize("Before rot_err_rad", before_rot))
    print("  " + summarize("After  rot_err_rad", after_rot))

    mean_before_pos = float(np.mean(before_pos))
    mean_after_pos = float(np.mean(after_pos))
    mean_before_rot = float(np.mean(before_rot))
    mean_after_rot = float(np.mean(after_rot))
    pos_red = (mean_before_pos - mean_after_pos) / max(mean_before_pos, 1e-12) * 100.0
    rot_red = (mean_before_rot - mean_after_rot) / max(mean_before_rot, 1e-12) * 100.0
    print(f"  Reduction: pos={pos_red:.2f}% rot={rot_red:.2f}%")

    tree, origin_elem, cur_xyz, cur_rpy = load_joint_tcp_origin(args.urdf)
    R_cur = t3d.euler.euler2mat(*cur_rpy, axes="sxyz")
    T_cur = np.eye(4, dtype=np.float64)
    T_cur[:3, :3] = R_cur
    T_cur[:3, 3] = cur_xyz

    T_new = T_cur @ D_alpha
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
