"""Calculate xArm hand-eye calibration from controller-reported EEF poses.

This is an alternative to ``calculate.py`` for robots whose controller
kinematics do not match the repository's generic xArm URDF. It consumes the
``eef.npy`` files already written by ``capture.py`` and leaves the original
calculator and its output files untouched.
"""

import argparse
import importlib.util
import os
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from paradex.calibration.Tsai_Lenz import solve_ax_xb
from paradex.calibration.utils import (
    handeye_calib_bimanual_path,
    handeye_calib_path,
)
from paradex.image.aruco import find_common_indices
from paradex.transforms.conversion import SOLVE_XA_B
from paradex.utils.file_io import find_latest_directory


DEFAULT_CHARUCO_WORKERS = min(8, os.cpu_count() or 1)


def load_base_calculator():
    """Load image preprocessing lazily; motion solving itself needs no robot URDF."""
    calculator_path = Path(__file__).with_name("calculate.py")
    spec = importlib.util.spec_from_file_location(
        "xarm_handeye_calculate_base",
        calculator_path,
    )
    calculator = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(calculator)
    return calculator


def get_valid_indices(root_dir):
    valid_indices = []
    for index in os.listdir(root_dir):
        index_path = os.path.join(root_dir, index)
        if not index.isdigit() or not os.path.isdir(index_path):
            continue
        ids_path = os.path.join(index_path, "charuco_3d_ids.npy")
        corners_path = os.path.join(index_path, "charuco_3d_corners.npy")
        if not os.path.isfile(ids_path) or not os.path.isfile(corners_path):
            print(f"Skipping index {index}: missing ChArUco detection files")
            continue
        if len(np.load(ids_path)) == 0:
            print(f"Skipping index {index}: empty ChArUco detection")
            continue
        valid_indices.append(index)
    return sorted(valid_indices, key=int)


def load_robot_eef_pose(root_dir, index):
    pose_path = os.path.join(root_dir, index, "eef.npy")
    if not os.path.isfile(pose_path):
        raise FileNotFoundError(
            f"Missing controller EEF pose for index {index}: {pose_path}"
        )

    pose = np.asarray(np.load(pose_path), dtype=np.float64)
    if pose.shape != (4, 4) or not np.all(np.isfinite(pose)):
        raise ValueError(
            f"Invalid controller EEF pose at {pose_path}: "
            f"expected a finite 4x4 matrix, got shape {pose.shape}"
        )
    return pose


def rotation_error_degrees(left, right):
    relative_rotation = left[:3, :3].T @ right[:3, :3]
    cosine = np.clip(
        (np.trace(relative_rotation) - 1.0) / 2.0,
        -1.0,
        1.0,
    )
    return float(np.degrees(np.arccos(cosine)))


def compute_motion(root_dir):
    motion_wrt_cam = []
    motion_wrt_robot = []
    index_list = get_valid_indices(root_dir)
    if len(index_list) < 2:
        raise ValueError(
            f"Need at least 2 valid indices for motion computation, got {len(index_list)}"
        )

    eef_list = [load_robot_eef_pose(root_dir, index) for index in index_list]
    charuco_id_list = [
        np.load(os.path.join(root_dir, index, "charuco_3d_ids.npy"))
        for index in index_list
    ]
    charuco_corner_list = [
        np.load(os.path.join(root_dir, index, "charuco_3d_corners.npy"))
        for index in index_list
    ]

    for i in range(1, len(index_list)):
        eef = eef_list[i]
        eef_prev = eef_list[i - 1]
        motion_wrt_robot.append(eef_prev @ np.linalg.inv(eef))

        common_idx, common_idx_prev = find_common_indices(
            charuco_id_list[i],
            charuco_id_list[i - 1],
        )
        if len(common_idx) < 3:
            raise ValueError(
                f"Need at least 3 common ChArUco corners for motion "
                f"{index_list[i - 1]}->{index_list[i]}, got {len(common_idx)}"
            )

        cam_corners = charuco_corner_list[i][common_idx]
        cam_corners_prev = charuco_corner_list[i - 1][common_idx_prev]
        cam_motion = SOLVE_XA_B(cam_corners, cam_corners_prev)
        motion_wrt_cam.append(cam_motion)

        fitted = (
            (cam_motion[:3, :3] @ cam_corners.T).T
            + cam_motion[:3, 3]
        )
        fit_error_mm = np.mean(
            np.linalg.norm(cam_corners_prev - fitted, axis=1)
        ) * 1000
        print(
            f"Motion {index_list[i - 1]}->{index_list[i]} camera fit: "
            f"{fit_error_mm:.2f} mm"
        )

    return motion_wrt_cam, motion_wrt_robot, index_list


def print_ax_xb_residuals(motion_wrt_cam, motion_wrt_robot, transform):
    translation_errors = []
    rotation_errors = []
    for i, (cam_motion, robot_motion) in enumerate(
        zip(motion_wrt_cam, motion_wrt_robot)
    ):
        left = cam_motion @ transform
        right = transform @ robot_motion
        trans_error = np.linalg.norm(left[:3, 3] - right[:3, 3]) * 1000
        rot_error = rotation_error_degrees(left, right)
        translation_errors.append(trans_error)
        rotation_errors.append(rot_error)
        print(f"Motion {i}: trans={trans_error:.2f}mm, rot={rot_error:.3f}deg")

    for label, values, unit in (
        ("translation", translation_errors, "mm"),
        ("rotation", rotation_errors, "deg"),
    ):
        values = np.asarray(values)
        print(
            f"AX=XB {label}: mean={values.mean():.3f}{unit}, "
            f"median={np.median(values):.3f}{unit}, "
            f"rms={np.sqrt(np.mean(values ** 2)):.3f}{unit}, "
            f"p95={np.percentile(values, 95):.3f}{unit}, "
            f"max={values.max():.3f}{unit}"
        )


def print_marker_consistency(root_dir, index_list, robot_wrt_cam_world):
    marker_positions = {}
    cam_wrt_robot = np.linalg.inv(robot_wrt_cam_world)

    for index in index_list:
        eef = load_robot_eef_pose(root_dir, index)
        eef_wrt_robot = np.linalg.inv(eef)
        corners = np.load(os.path.join(root_dir, index, "charuco_3d_corners.npy"))
        ids = np.load(os.path.join(root_dir, index, "charuco_3d_ids.npy"))

        corners_h = np.ones((len(corners), 4))
        corners_h[:, :3] = corners
        corners_wrt_eef = (
            eef_wrt_robot @ cam_wrt_robot @ corners_h.T
        ).T[:, :3]
        for marker_id, position in zip(np.asarray(ids).reshape(-1), corners_wrt_eef):
            marker_positions.setdefault(int(marker_id), []).append(position)

    std_norms_mm = np.asarray(
        [
            np.linalg.norm(np.std(np.asarray(positions), axis=0)) * 1000
            for positions in marker_positions.values()
            if len(positions) > 1
        ]
    )
    if len(std_norms_mm):
        print(
            f"Marker consistency: mean={std_norms_mm.mean():.3f}mm, "
            f"median={np.median(std_norms_mm):.3f}mm, "
            f"max={std_norms_mm.max():.3f}mm"
        )


def calculate_sequence(root_path, save_path, charuco_workers=DEFAULT_CHARUCO_WORKERS):
    base_calculator = load_base_calculator()
    base_calculator.validate_capture_directory(root_path)
    base_calculator.undistort_and_detect_charuco(root_path, charuco_workers)
    motion_wrt_cam, motion_wrt_robot, index_list = compute_motion(root_path)
    robot_wrt_cam_world = solve_ax_xb(
        motion_wrt_cam,
        motion_wrt_robot,
        verbose=True,
    )

    print_ax_xb_residuals(
        motion_wrt_cam,
        motion_wrt_robot,
        robot_wrt_cam_world,
    )
    print_marker_consistency(root_path, index_list, robot_wrt_cam_world)
    np.save(save_path, robot_wrt_cam_world)
    print(f"Saved controller-pose C2R to {save_path}")
    return robot_wrt_cam_world


def main():
    parser = argparse.ArgumentParser(
        description="Calculate xArm hand-eye calibration using captured eef.npy poses."
    )
    parser.add_argument(
        "--name",
        type=str,
        default=None,
        help="Name of the calibration directory.",
    )
    parser.add_argument(
        "--bimanual",
        action="store_true",
        help="Calculate right and left transforms from a bimanual session.",
    )
    parser.add_argument(
        "--charuco-workers",
        type=int,
        default=DEFAULT_CHARUCO_WORKERS,
        help="Number of camera images to process in parallel.",
    )
    args = parser.parse_args()

    base_dir = handeye_calib_bimanual_path if args.bimanual else handeye_calib_path
    name = args.name or find_latest_directory(base_dir)
    root_path = os.path.join(base_dir, name)

    if args.bimanual:
        for side, suffix in (("Right", "R"), ("Left", "L")):
            side_path = os.path.join(root_path, side)
            if not os.path.isdir(side_path):
                raise FileNotFoundError(f"Missing {side} capture directory: {side_path}")
            print(f"Calculating {side} xArm from captured controller EEF poses...")
            calculate_sequence(
                side_path,
                os.path.join(root_path, f"C2R_{suffix}_robot_eef.npy"),
                args.charuco_workers,
            )
    else:
        calculate_sequence(
            root_path,
            os.path.join(root_path, "C2R_robot_eef.npy"),
            args.charuco_workers,
        )


if __name__ == "__main__":
    main()
