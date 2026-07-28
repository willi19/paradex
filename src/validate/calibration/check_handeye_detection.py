"""Per-pose Charuco detection check for a hand-eye capture.

Hand-eye needs the board seen (and triangulable) at every pose. calculate.py silently
skips a pose whose board yields zero 3D corners, so a bad pose only shows up as a
missing `charuco_3d_*.npy` later. This inspects each pose directly and reports, per
camera, how many charuco corners are detected — so you can tell whether the board was
out of frame, blurred, or just seen by too few cameras to triangulate.

Usage:
    # latest capture, all poses:
    python src/validate/calibration/check_handeye_detection.py
    # a specific capture / single pose:
    python src/validate/calibration/check_handeye_detection.py --name 20260721_134538 --pose 1
"""
import argparse
import os

import numpy as np

from paradex.calibration.utils import handeye_calib_path, load_camparam
from paradex.image.image_dict import ImageDict
from paradex.image.aruco import detect_charuco, merge_charuco_detection
from paradex.utils.file_io import find_latest_directory


def check_pose(root_dir, pose, intrinsic, extrinsic):
    pose_dir = os.path.join(root_dir, pose)
    if not os.path.isdir(os.path.join(pose_dir, "images")):
        return

    # Camera params live only in pose 0/cam_param, so inject them into every pose —
    # exactly what calculate.py does via update_path. Without them triangulation
    # yields zero 3D corners even when 2D detection is fine, which is the trap here.
    img_dict = ImageDict.from_path(pose_dir)
    img_dict.set_camparam(intrinsic, extrinsic)

    per_cam = img_dict.apply(detect_charuco, False)   # {serial: detection}
    seen_cams = sum(1 for det in per_cam.values()
                    if len(merge_charuco_detection(det).get("checkerIDs", [])) > 0)

    # The number that actually matters: 3D corners after triangulation. Must go
    # through undistort() first, same as calculate.py — triangulating raw images
    # gives a malformed result.
    undist = img_dict.undistort()
    charuco_3d = merge_charuco_detection(undist.triangulate_charuco())
    n3d = len(charuco_3d.get("checkerCorner", []))

    flag = "" if n3d > 0 else "   <-- 0 3D corners -> calculate.py DROPS this pose"
    print(f"pose {pose:>3}: {seen_cams:2}/{len(per_cam)} cams see board, "
          f"{n3d:3} 3D corners{flag}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default=None, help="capture dir (default: latest)")
    ap.add_argument("--pose", default=None, help="single pose index (default: all)")
    args = ap.parse_args()

    name = args.name or find_latest_directory(handeye_calib_path)
    root_dir = os.path.join(handeye_calib_path, name)
    print(f"Capture: {root_dir}\n")

    intrinsic, extrinsic = load_camparam(os.path.join(root_dir, "0"))

    if args.pose is not None:
        check_pose(root_dir, args.pose, intrinsic, extrinsic)
        return

    poses = sorted((d for d in os.listdir(root_dir)
                    if os.path.isdir(os.path.join(root_dir, d))),
                   key=lambda x: int(x) if x.isdigit() else 1e9)
    for pose in poses:
        check_pose(root_dir, pose, intrinsic, extrinsic)


if __name__ == "__main__":
    main()
