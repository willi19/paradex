"""Snapshot the cameras at the robot's CURRENT pose and report whether the EE board
triangulates — a go/no-go check to run while teaching hand-eye poses.

Hand-eye only uses the board mounted on the robot (the non-floor board). A taught pose
is useless if that board isn't seen by enough cameras, and that only surfaced after a
full capture last time. Run this with the arm held at a candidate pose: PASS -> save it
in the GUI, FAIL -> adjust the pose so the EE board faces the cameras, and re-run.

Prereqs: capture-PC daemons up (server_daemon.py), and a current extrinsic calibration
(load_current_camparam). Does NOT need the franka daemon — it only reads cameras.

Usage:
    python src/validate/calibration/check_board_now.py
    python src/validate/calibration/check_board_now.py --min 8   # stricter
"""
import argparse
import os
import shutil

import numpy as np

from paradex.io.camera_system.remote_camera_controller import remote_camera_controller
from paradex.calibration.utils import load_current_camparam
from paradex.image.image_dict import ImageDict
from paradex.image.aruco import merge_charuco_detection, boardinfo_dict
from paradex.utils.file_io import remove_home
from paradex.utils.path import shared_dir

FLOOR_BOARD = "1"   # static board id excluded from hand-eye (matches calculate.py)


def _floor_range(key):
    offset = 0
    for b_id, cfg in boardinfo_dict.items():
        n = (cfg["numX"] - 1) * (cfg["numY"] - 1)
        if b_id == key:
            return (offset, offset + n)
        offset += n
    return None


def count_board_corners():
    """Capture the cameras once and triangulate. Returns (floor_corners, ee_corners).

    Reusable by the teaching GUI's "Check Board" button. Raises on capture/param
    failure so the caller can surface it.
    """
    tmp = os.path.join(shared_dir, "handeye_check_tmp")
    if os.path.exists(tmp):
        shutil.rmtree(tmp)
    os.makedirs(os.path.join(tmp, "images"), exist_ok=True)

    # Capture one multi-cam image set at the current pose.
    rcc = remote_camera_controller("handeye_check")
    try:
        rcc.start("image", False, remove_home(tmp))
        rcc.stop()
    finally:
        rcc.end()

    # Same path as calculate.py: inject current camera params, undistort, triangulate.
    intrinsic, extrinsic = load_current_camparam()
    img_dict = ImageDict.from_path(tmp)
    img_dict.set_camparam(intrinsic, extrinsic)

    charuco_3d = merge_charuco_detection(img_dict.undistort().triangulate_charuco())
    ids = np.asarray(charuco_3d.get("checkerIDs", np.zeros(0))).ravel()

    fr = _floor_range(FLOOR_BOARD)
    if fr is not None:
        ee = int(((ids < fr[0]) | (ids >= fr[1])).sum())
    else:
        ee = len(ids)
    return len(ids) - ee, ee


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--min", type=int, default=6,
                    help="min EE-board 3D corners for PASS (hand-eye needs >=4)")
    args = ap.parse_args()

    floor, ee = count_board_corners()
    print()
    print(f"  floor board : {floor:3} 3D corners")
    print(f"  EE board    : {ee:3} 3D corners   (need >= {args.min})")
    if ee >= args.min:
        print(f"  ==> PASS — save this pose")
    else:
        print(f"  ==> FAIL — turn the EE board toward the cameras and re-run")


if __name__ == "__main__":
    main()
