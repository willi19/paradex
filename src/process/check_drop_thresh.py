"""
Inspect frame mean-intensity distribution per cam in a scene, to tune
`--drop-thresh` for trim_and_rename_frames.py.

Scans <scene_dir>/video_extracted/<cam>/*.jpg, reports brightness stats,
and lists frames that would be flagged as drops at the given threshold.
Optionally saves flagged frames to a directory for visual inspection.

Example:
    python src/process/check_drop_thresh.py \
        --scene-dir /path/to/obj/0 --thresh 60
    python src/process/check_drop_thresh.py \
        --scene-dir /path/to/obj/0 --cam 22645026 --thresh 60 --show-all
    python src/process/check_drop_thresh.py \
        --scene-dir /path/to/obj/0 --thresh 60 --save-viz /tmp/flagged
"""

import argparse
import os
import cv2
import numpy as np


def list_files(folder):
    return sorted(f for f in os.listdir(folder) if not f.startswith("."))


def list_dirs(folder):
    return sorted(d for d in os.listdir(folder) if os.path.isdir(os.path.join(folder, d)))


def scan_cam(cam_dir: str, thresh: float):
    files = list_files(cam_dir)
    means = []  # list[(fname, mean or None)]
    flagged = []
    for f in files:
        img = cv2.imread(os.path.join(cam_dir, f))
        if img is None:
            means.append((f, None))
            flagged.append((f, None))
            continue
        m = float(img.mean())
        means.append((f, m))
        if m <= thresh:
            flagged.append((f, m))
    return means, flagged


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene-dir", required=True,
                    help="scene dir containing video_extracted/")
    ap.add_argument("--cam", default=None,
                    help="specific cam id; default scans all cams in the scene")
    ap.add_argument("--thresh", type=float, default=60.0)
    ap.add_argument("--show-all", action="store_true",
                    help="print mean of every frame (not just flagged)")
    ap.add_argument("--save-viz", default=None,
                    help="directory to copy flagged frames into for inspection")
    args = ap.parse_args()

    ve_dir = os.path.join(args.scene_dir, "video_extracted")
    if not os.path.isdir(ve_dir):
        print(f"[error] not a scene dir (no video_extracted): {args.scene_dir}")
        return

    cams = [args.cam] if args.cam else list_dirs(ve_dir)

    total = 0
    total_flagged = 0
    for cam in cams:
        cam_dir = os.path.join(ve_dir, cam)
        if not os.path.isdir(cam_dir):
            print(f"[miss] {cam_dir}")
            continue

        means, flagged = scan_cam(cam_dir, args.thresh)
        arr = np.array([m for _, m in means if m is not None])
        total += len(means)
        total_flagged += len(flagged)

        print(f"\n== {cam} == frames={len(means)} flagged={len(flagged)} (thresh={args.thresh})")
        if len(arr):
            print(f"  brightness: min={arr.min():.1f} max={arr.max():.1f} "
                  f"mean={arr.mean():.1f} median={np.median(arr):.1f}")
            p = np.percentile(arr, [1, 5, 10, 50])
            print(f"  percentiles: p1={p[0]:.1f} p5={p[1]:.1f} p10={p[2]:.1f} p50={p[3]:.1f}")

        if args.show_all:
            for f, m in means:
                mstr = "None" if m is None else f"{m:6.1f}"
                flag = "  <-- FLAG" if (m is None or (m is not None and m <= args.thresh)) else ""
                print(f"  {f}: {mstr}{flag}")
        else:
            for f, m in flagged:
                mstr = "None" if m is None else f"{m:.1f}"
                print(f"  FLAG {f} mean={mstr}")

        if args.save_viz and flagged:
            outd = os.path.join(args.save_viz, cam)
            os.makedirs(outd, exist_ok=True)
            for f, _ in flagged:
                src = os.path.join(cam_dir, f)
                img = cv2.imread(src)
                if img is not None:
                    cv2.imwrite(os.path.join(outd, f), img)
            print(f"  saved {len(flagged)} flagged frame(s) to {outd}")

    print(f"\nTOTAL: frames={total} flagged={total_flagged} (thresh={args.thresh})")


if __name__ == "__main__":
    main()
