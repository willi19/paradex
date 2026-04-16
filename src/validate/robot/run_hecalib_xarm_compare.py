"""
Run sequential compare over system/current/hecalib/xarm/{i}_aa.npy poses.

Each pose file is a 4x4 homogeneous matrix.
For each index, this script runs run_sequential_compare() and saves one npz.
"""

import argparse
import glob
import os
from datetime import datetime

import numpy as np
import transforms3d as t3d

from src.validate.robot.xarm_ik_pose_cli import run_sequential_compare


def homo2aa(h):
    t = h[:3, 3] * 1000.0
    axis, angle = t3d.axangles.mat2axangle(h[:3, :3], unit_thresh=0.001)
    return np.concatenate([t, axis * angle])


def collect_pose_files(root, start_idx, end_idx):
    files = []
    for i in range(start_idx, end_idx + 1):
        p = os.path.join(root, f"{i}_aa.npy")
        if os.path.exists(p):
            files.append((i, p))
    if not files:
        # fallback: scan all *_aa.npy and parse integer prefix
        for p in sorted(glob.glob(os.path.join(root, "*_aa.npy"))):
            name = os.path.basename(p)
            try:
                idx = int(name.split("_", 1)[0])
            except Exception:
                continue
            files.append((idx, p))
        files.sort(key=lambda x: x[0])
    return files


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pose-root", type=str, default="system/current/hecalib/xarm")
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--end", type=int, default=50)
    parser.add_argument("--hold-sec", type=float, default=1.0)
    parser.add_argument("--ee-frame", type=str, default="link_tcp")
    parser.add_argument("--fps", type=float, default=100.0)
    parser.add_argument("--out-dir", type=str, default=None)
    args = parser.parse_args()

    files = collect_pose_files(args.pose_root, args.start, args.end)
    print(f"found {len(files)} pose files in range [{args.start}, {args.end}]")
    if not files:
        raise RuntimeError("No pose files found")

    if args.out_dir is None:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_dir = f"xarm_hecalib_compare_{ts}"
    else:
        out_dir = args.out_dir
    os.makedirs(out_dir, exist_ok=True)
    print(f"output dir: {out_dir}")

    successes = 0
    failures = []
    pos_errs = []
    rot_errs = []

    for n, (idx, path) in enumerate(files, start=1):
        print(f"\n[{n}/{len(files)}] idx={idx} file={path}")
        try:
            h = np.asarray(np.load(path), dtype=np.float64)
            if h.shape != (4, 4):
                raise ValueError(f"Unexpected pose shape: {h.shape}")

            pose_aa = homo2aa(h)
            pose_deg = pose_aa.copy()
            pose_deg[3:6] = np.rad2deg(pose_deg[3:6])
            print(f"target(mm/deg): {np.array2string(pose_deg, precision=3)}")

            result = run_sequential_compare(
                pose_aa=pose_aa,
                hold_sec=args.hold_sec,
                ee_frame=args.ee_frame,
                fps=args.fps,
            )

            out_path = os.path.join(out_dir, f"{idx:02d}_compare.npz")
            np.savez(
                out_path,
                source_pose_file=path,
                source_index=np.array([idx], dtype=np.int32),
                target_pose_aa=pose_aa,
                initial_qpos=result["initial_qpos"],
                initial_pose_aa=result["initial_pose_aa"],
                ik_action_qpos=result["ik_action_qpos"],
                ik_fk_pose_aa=result["ik_fk_pose_aa"],
                after_angle_qpos=result["after_angle_qpos"],
                after_angle_pose_aa=result["after_angle_pose_aa"],
                after_servo_qpos=result["after_servo_qpos"],
                after_servo_pose_aa=result["after_servo_pose_aa"],
                servo_fk_pose_aa=result["servo_fk_pose_aa"],
                after_return_pose_aa=result["after_return_pose_aa"],
                ik_fk_pos_err_mm=result["ik_fk_pos_err_mm"],
                ik_fk_rot_err_rad=result["ik_fk_rot_err_rad"],
                ik_pos_err_mm=result["ik_pos_err_mm"],
                ik_rot_err_rad=result["ik_rot_err_rad"],
                servo_fk_pos_err_mm=result["servo_fk_pos_err_mm"],
                servo_fk_rot_err_rad=result["servo_fk_rot_err_rad"],
                servo_pos_err_mm=result["servo_pos_err_mm"],
                servo_rot_err_rad=result["servo_rot_err_rad"],
            )
            print(f"saved: {out_path}")

            successes += 1
            pos_errs.append(float(result["servo_fk_pos_err_mm"]))
            rot_errs.append(float(result["servo_fk_rot_err_rad"]))
        except Exception as e:
            failures.append((idx, path, str(e)))
            print(f"[FAIL] idx={idx}: {e}")

    print("\n=== Batch Summary ===")
    print(f"success={successes}, fail={len(failures)}")
    if pos_errs:
        x = np.asarray(pos_errs, dtype=np.float64)
        print(
            f"servo_fk_pos_err_mm: mean={np.mean(x):.4f}, median={np.median(x):.4f}, "
            f"p95={np.percentile(x,95):.4f}, max={np.max(x):.4f}"
        )
    if rot_errs:
        x = np.asarray(rot_errs, dtype=np.float64)
        print(
            f"servo_fk_rot_err_rad: mean={np.mean(x):.6f}, median={np.median(x):.6f}, "
            f"p95={np.percentile(x,95):.6f}, max={np.max(x):.6f}"
        )

    if failures:
        fail_path = os.path.join(out_dir, "failures.txt")
        with open(fail_path, "w", encoding="utf-8") as f:
            for idx, path, msg in failures:
                f.write(f"{idx}\t{path}\t{msg}\n")
        print(f"failure log: {fail_path}")


if __name__ == "__main__":
    main()
