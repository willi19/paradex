"""Create bimanual xArm pose files by matching a single-hand capture.

For each reference pose this script repeatedly:

1. captures exactly one image from each camera,
2. fits a rigid ChArUco board by robust multi-view reprojection,
3. minimizes a robust 3D keypoint matching loss, and
4. applies one bounded Cartesian correction to the selected xArm.

No prior hand-eye transform is required.  A small central-difference XYZ probe
estimates the camera/robot axis rotation once per arm.  The resulting
``*_qpos.npy`` files are intended to be replayed by the normal calibration
capture path in a later step.

Physical motion is disabled unless ``--execute`` is passed.
"""

import argparse
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from paradex.calibration.handeye_pose_matching import (
    align_keypoints,
    build_servo_target,
    estimate_camera_from_robot_rotation,
    fit_multiview_pose,
    match_keypoints,
)
from paradex.calibration.utils import (
    get_cammtx,
    handeye_calib_path,
    load_camparam,
    load_current_camparam,
    save_current_camparam,
)
from paradex.image.aruco import boardinfo_dict, detect_charuco, get_board_cor
from paradex.image.image_dict import ImageDict
from paradex.io.camera_system.remote_camera_controller import remote_camera_controller
from paradex.utils.file_io import find_latest_directory, remove_home
from paradex.utils.path import shared_dir
from paradex.utils.system import config_dir, get_pc_list, network_info


LEFT_XARM_IP = "192.168.1.196"
DEFAULT_SEED_DIR = Path(config_dir) / "hecalib" / "xarm"
DEFAULT_OUTPUT_ROOT = Path(shared_dir) / "handeye_pose_matching"


@dataclass(frozen=True)
class KeypointObservation:
    ids: np.ndarray
    points: np.ndarray
    path: Path
    reprojection_rmse_px: float = float("nan")
    board_id: str = ""
    image_path: Optional[Path] = None


def _write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as file_obj:
        json.dump(value, file_obj, indent=2, sort_keys=True)


def _read_json(path):
    with Path(path).open("r") as file_obj:
        return json.load(file_obj)


def _latest_resume_dir():
    if not DEFAULT_OUTPUT_ROOT.is_dir():
        raise FileNotFoundError(
            "Hand-eye pose-matching root does not exist: {}".format(
                DEFAULT_OUTPUT_ROOT
            )
        )
    candidates = sorted(
        (
            path
            for path in DEFAULT_OUTPUT_ROOT.iterdir()
            if path.is_dir() and (path / "manifest.json").is_file()
        ),
        key=lambda path: path.name,
        reverse=True,
    )
    if not candidates:
        raise FileNotFoundError("No hand-eye pose-matching sessions can be resumed")
    return candidates[0]


def _completed_pose_prefix(output_dir, side_name, pose_indices):
    """Return the contiguous completed prefix, remaining poses, and warm qpos."""

    completed_reports = {}
    warm_start_qpos = None
    split_index = 0
    for split_index, pose_index in enumerate(pose_indices):
        metrics_path = Path(output_dir) / side_name / "{}_metrics.json".format(
            pose_index
        )
        qpos_path = Path(output_dir) / side_name / "{}_qpos.npy".format(pose_index)
        if not metrics_path.is_file() or not qpos_path.is_file():
            break
        report = _read_json(metrics_path)
        if not report.get("success"):
            break
        qpos = np.asarray(np.load(qpos_path), dtype=np.float64)
        if qpos.shape != (6,) or not np.all(np.isfinite(qpos)):
            break
        completed_reports[str(pose_index)] = report
        warm_start_qpos = qpos
    else:
        split_index = len(pose_indices)
    return (
        completed_reports,
        list(pose_indices[:split_index]),
        list(pose_indices[split_index:]),
        warm_start_qpos,
    )


def save_reference_overlays(reference_path, current_path, output_dir, alpha=0.5):
    """Blend matching reference/current camera images for visual comparison."""

    if not 0 <= alpha <= 1:
        raise ValueError("overlay alpha must be in [0, 1]")
    reference_images = ImageDict.from_path(reference_path).images
    current_images = ImageDict.from_path(current_path).images
    common_serials = sorted(set(reference_images) & set(current_images))
    if not common_serials:
        raise RuntimeError("Reference and current captures have no common cameras")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    for serial in common_serials:
        reference_image = reference_images[serial]
        current_image = current_images[serial]
        if reference_image.shape != current_image.shape:
            raise ValueError(
                "Camera {} image shapes differ: {} != {}".format(
                    serial, reference_image.shape, current_image.shape
                )
            )
        overlay = cv2.addWeighted(
            reference_image,
            1.0 - alpha,
            current_image,
            alpha,
            0.0,
        )
        output_path = output_dir / "{}.png".format(serial)
        if not cv2.imwrite(str(output_path), overlay):
            raise RuntimeError("Could not save reference overlay: {}".format(output_path))
    return output_dir


def _latest_reference_dir():
    name = find_latest_directory(handeye_calib_path)
    if name is None:
        raise FileNotFoundError("No single-hand calibration sessions were found")
    return Path(handeye_calib_path) / name


def _reference_pose_indices(reference_dir):
    indices = []
    for child in Path(reference_dir).iterdir():
        if not child.is_dir() or not child.name.isdigit():
            continue
        if (child / "images").is_dir():
            indices.append(int(child.name))
    return sorted(indices)


def _select_pose_indices(available, start, end, explicit):
    available = set(available)
    if explicit:
        requested = [int(value) for value in explicit.split(",") if value.strip()]
    else:
        requested = sorted(index for index in available if start <= index <= end)
    missing = [index for index in requested if index not in available]
    if missing:
        raise ValueError("Reference keypoints are missing for poses {}".format(missing))
    if not requested:
        raise ValueError("No reference poses were selected")
    return requested


def _load_seed(seed_dir, pose_index):
    path = Path(seed_dir) / "{}_qpos.npy".format(pose_index)
    if not path.is_file():
        raise FileNotFoundError("Missing seed pose: {}".format(path))
    qpos = np.asarray(np.load(path), dtype=np.float64)
    if qpos.shape != (6,):
        raise ValueError("Seed pose {} must have shape (6,), got {}".format(path, qpos.shape))
    return qpos


class SnapshotKeypointDetector:
    def __init__(
        self,
        rcc,
        intrinsic,
        extrinsic,
        observation_root,
        workers,
        checker_length_m,
        fps,
    ):
        self.rcc = rcc
        self.intrinsic = intrinsic
        self.extrinsic = extrinsic
        self.observation_root = Path(observation_root)
        self.workers = int(workers)
        self.checker_length_m = float(checker_length_m)
        self.fps = int(fps)
        if self.workers < 1:
            raise ValueError("workers must be positive")
        if self.fps < 1:
            raise ValueError("fps must be positive")
        self.observation_root.mkdir(parents=True, exist_ok=True)
        self.observation_root.chmod(0o777)

    def capture(self, label):
        capture_dir = self.observation_root / label
        capture_dir.mkdir(parents=True, exist_ok=True)
        capture_dir.chmod(0o777)
        capture_started = False
        try:
            self.rcc.start(
                "image",
                False,
                save_path=remove_home(capture_dir),
                fps=self.fps,
            )
            capture_started = True
        finally:
            if capture_started:
                self.rcc.stop()

        return extract_keypoints(
            image_path=capture_dir,
            intrinsic=self.intrinsic,
            extrinsic=self.extrinsic,
            workers=self.workers,
            save_dir=capture_dir,
            checker_length_m=self.checker_length_m,
        )


def extract_keypoints(
    image_path,
    intrinsic,
    extrinsic,
    workers,
    save_dir=None,
    checker_length_m=0.025,
):
    """Fit a rigid ChArUco pose from one multi-camera snapshot."""

    image_path = Path(image_path)
    image_dict = ImageDict.from_path(image_path)
    image_dict.images = {
        serial: image
        for serial, image in image_dict.images.items()
        if serial in intrinsic and serial in extrinsic
    }
    if not image_dict.images:
        raise RuntimeError("Snapshot contains no cameras with current calibration")
    image_dict.set_camparam(intrinsic, extrinsic)
    undistorted = image_dict.undistort()

    serials = list(undistorted.images)
    with ThreadPoolExecutor(max_workers=int(workers)) as executor:
        detections = dict(
            zip(
                serials,
                executor.map(
                    detect_charuco,
                    (undistorted.images[serial] for serial in serials),
                ),
            )
        )
    board_counts = {}
    for camera_detection in detections.values():
        for board_id, detection in camera_detection.items():
            board_counts[board_id] = board_counts.get(board_id, 0) + len(
                detection["checkerIDs"]
            )
    if not board_counts:
        raise RuntimeError("No triangulated ChArUco corners in {}".format(image_path))
    board_id = max(board_counts, key=board_counts.get)

    triangulated = undistorted.triangulate_charuco(detections=detections)
    if board_id not in triangulated or len(triangulated[board_id]["checkerIDs"]) < 6:
        raise RuntimeError(
            "ChArUco board {} could not be triangulated in {}".format(
                board_id, image_path
            )
        )
    model = get_board_cor()[board_id]
    model_ids = np.asarray(model["checkerIDs"]).reshape(-1)
    model_points = np.asarray(model["checkerCorner"], dtype=np.float64)
    if checker_length_m <= 0:
        raise ValueError("checker_length_m must be positive")
    # get_board_cor() uses 0.05 m per configured unit; the calibration board
    # used by this xArm setup has 25 mm checkers by default.
    model_points = model_points * (checker_length_m / 0.05)
    initial_alignment = align_keypoints(
        triangulated[board_id]["checkerIDs"],
        triangulated[board_id]["checkerCorner"],
        model_ids,
        model_points,
        min_common=6,
    )
    observations = {
        serial: (
            camera_detection[board_id]["checkerIDs"],
            camera_detection[board_id]["checkerCorner"],
        )
        for serial, camera_detection in detections.items()
        if board_id in camera_detection
    }
    pose_fit = fit_multiview_pose(
        object_ids=model_ids,
        object_points=model_points,
        observations=observations,
        projection_matrices=get_cammtx(intrinsic, extrinsic),
        initial_transform=initial_alignment.transform,
    )

    detected_local_ids = np.unique(
        np.concatenate(
            [np.asarray(value[0]).reshape(-1) for value in observations.values()]
        )
    ).astype(np.int64)
    model_lookup = {int(point_id): index for index, point_id in enumerate(model_ids)}
    model_indices = [model_lookup[int(point_id)] for point_id in detected_local_ids]
    points = (
        pose_fit.transform[:3, :3] @ model_points[model_indices].T
    ).T + pose_fit.transform[:3, 3]
    board_offset = 0
    for candidate_id, config in boardinfo_dict.items():
        if candidate_id == board_id:
            break
        board_offset += (config["numX"] - 1) * (config["numY"] - 1)
    ids = detected_local_ids + board_offset

    result_path = Path(save_dir) if save_dir is not None else image_path
    if save_dir is not None:
        result_path.mkdir(parents=True, exist_ok=True)
        np.save(result_path / "charuco_3d_ids.npy", ids)
        np.save(result_path / "charuco_3d_corners.npy", points)
        _write_json(
            result_path / "pose_fit.json",
            {
                "board_id": board_id,
                "camera_count": pose_fit.camera_count,
                "observation_count": pose_fit.observation_count,
                "reprojection_rmse_px": pose_fit.reprojection_rmse_px,
                "checker_length_m": checker_length_m,
                "transform": pose_fit.transform.tolist(),
            },
        )
    return KeypointObservation(
        ids=ids,
        points=points,
        path=result_path,
        reprojection_rmse_px=pose_fit.reprojection_rmse_px,
        board_id=board_id,
        image_path=image_path,
    )


def prepare_reference_keypoints(
    reference_dir,
    pose_indices,
    intrinsic,
    extrinsic,
    output_dir,
    workers,
    checker_length_m,
    reuse_existing=False,
):
    """Re-triangulate old reference images with the current camera parameters."""

    prepared = {}
    reference_output = Path(output_dir) / "reference_keypoints"
    for pose_index in pose_indices:
        cached_path = reference_output / str(pose_index)
        ids_path = cached_path / "charuco_3d_ids.npy"
        corners_path = cached_path / "charuco_3d_corners.npy"
        fit_path = cached_path / "pose_fit.json"
        if (
            reuse_existing
            and ids_path.is_file()
            and corners_path.is_file()
            and fit_path.is_file()
        ):
            fit = _read_json(fit_path)
            prepared[pose_index] = KeypointObservation(
                ids=np.load(ids_path),
                points=np.load(corners_path),
                path=cached_path,
                reprojection_rmse_px=float(fit["reprojection_rmse_px"]),
                board_id=fit.get("board_id", ""),
                image_path=Path(reference_dir) / str(pose_index),
            )
            print("Reusing reference keypoints for pose {}".format(pose_index))
            continue
        print("Preparing reference keypoints for pose {}...".format(pose_index))
        prepared[pose_index] = extract_keypoints(
            image_path=Path(reference_dir) / str(pose_index),
            intrinsic=intrinsic,
            extrinsic=extrinsic,
            workers=workers,
            save_dir=cached_path,
            checker_length_m=checker_length_m,
        )
        print(
            "Reference pose {} rigid reprojection RMSE: {:.2f} px".format(
                pose_index, prepared[pose_index].reprojection_rmse_px
            )
        )
    return prepared


def _create_controller(ip):
    from paradex.io.robot_controller.xarm_controller import XArmController

    return XArmController(ip=ip)


def _move_and_settle(controller, target, settle_sec):
    controller.move(np.asarray(target, dtype=np.float64), is_servo=False)
    time.sleep(settle_sec)
    if controller.is_error():
        raise RuntimeError("xArm reported an error after a motion command")


def estimate_axis_rotation(
    controller,
    detector,
    side_name,
    probe_m,
    settle_sec,
    min_common,
):
    """Estimate camera-from-robot rotation using bounded +/- XYZ probes."""

    origin = controller.get_data()["position"]
    robot_deltas = []
    camera_deltas = []

    try:
        for axis_index, axis_name in enumerate("xyz"):
            axis = np.zeros(3, dtype=np.float64)
            axis[axis_index] = 1.0

            plus_target = origin.copy()
            plus_target[:3, 3] += axis * probe_m
            _move_and_settle(controller, plus_target, settle_sec)
            plus = detector.capture(
                "{}/axis_probe/{}_plus".format(side_name, axis_name)
            )

            minus_target = origin.copy()
            minus_target[:3, 3] -= axis * probe_m
            _move_and_settle(controller, minus_target, settle_sec)
            minus = detector.capture(
                "{}/axis_probe/{}_minus".format(side_name, axis_name)
            )

            _, plus_points, minus_points = match_keypoints(
                plus.ids,
                plus.points,
                minus.ids,
                minus.points,
                min_common=min_common,
            )
            robot_deltas.append(axis * probe_m)
            camera_deltas.append(np.mean(plus_points - minus_points, axis=0) * 0.5)
            _move_and_settle(controller, origin, settle_sec)
    finally:
        _move_and_settle(controller, origin, settle_sec)

    robot_deltas = np.asarray(robot_deltas)
    camera_deltas = np.asarray(camera_deltas)
    camera_from_robot = estimate_camera_from_robot_rotation(robot_deltas, camera_deltas)
    predicted = (camera_from_robot @ robot_deltas.T).T
    relative_error = float(
        np.linalg.norm(predicted - camera_deltas) / np.linalg.norm(robot_deltas)
    )
    observed_scales = np.linalg.norm(camera_deltas, axis=1) / probe_m
    if np.any(observed_scales < 0.5) or np.any(observed_scales > 1.5):
        raise RuntimeError(
            "Axis probe scale is inconsistent with rigid motion: {}".format(observed_scales)
        )
    if relative_error > 0.35:
        raise RuntimeError(
            "Axis probe rotation fit is too noisy: relative error {:.3f}".format(
                relative_error
            )
        )
    return camera_from_robot, {
        "camera_from_robot_rotation": camera_from_robot.tolist(),
        "observed_scales": observed_scales.tolist(),
        "relative_fit_error": relative_error,
    }


def _alignment_metrics(iteration, alignment, current, reference):
    return {
        "iteration": int(iteration),
        "common_keypoints": int(len(alignment.common_ids)),
        "raw_rmse_mm": alignment.raw_rmse_m * 1000.0,
        "registered_rmse_mm": alignment.registered_rmse_m * 1000.0,
        "robust_loss_m2": alignment.robust_loss_m2,
        "centroid_error_mm": alignment.translation_error_m * 1000.0,
        "rotation_error_deg": float(np.rad2deg(alignment.rotation_error_rad)),
        "live_reprojection_rmse_px": current.reprojection_rmse_px,
        "reference_reprojection_rmse_px": reference.reprojection_rmse_px,
    }


def optimize_pose(
    controller,
    detector,
    side_name,
    pose_index,
    reference,
    seed_qpos,
    camera_from_robot_rotation,
    output_dir,
    args,
):
    _move_and_settle(controller, seed_qpos, args.settle_sec)
    history = []
    success = False

    for iteration in range(args.max_moves + 1):
        current = detector.capture(
            "{}/pose_{:03d}/iteration_{:02d}".format(
                side_name, pose_index, iteration
            )
        )
        overlay_dir = None
        if reference.image_path is not None and current.image_path is not None:
            overlay_dir = save_reference_overlays(
                reference_path=reference.image_path,
                current_path=current.image_path,
                output_dir=current.path / "reference_overlay",
                alpha=0.5,
            )
        alignment = align_keypoints(
            reference.ids,
            reference.points,
            current.ids,
            current.points,
            min_common=args.min_common,
            huber_delta_m=args.huber_delta_mm / 1000.0,
        )
        metrics = _alignment_metrics(iteration, alignment, current, reference)
        metrics["phase"] = "initial" if iteration == 0 else "refinement"
        metrics["reference_overlay_dir"] = (
            str(overlay_dir) if overlay_dir is not None else None
        )
        history.append(metrics)
        print(
            "[{} pose {:03d} iter {:02d}] loss={:.3e}, keypoint={:.2f} mm, "
            "centroid={:.2f} mm, rotation={:.2f} deg, fit={:.2f} mm".format(
                side_name,
                pose_index,
                iteration,
                metrics["robust_loss_m2"],
                metrics["raw_rmse_mm"],
                metrics["centroid_error_mm"],
                metrics["rotation_error_deg"],
                metrics["registered_rmse_mm"],
            )
        )
        if metrics["registered_rmse_mm"] > args.max_registered_rmse_mm:
            raise RuntimeError(
                "Rigid keypoint fit is inconsistent ({:.2f} mm)".format(
                    metrics["registered_rmse_mm"]
                )
            )
        if metrics["live_reprojection_rmse_px"] > args.max_live_reprojection_rmse_px:
            raise RuntimeError(
                "Live rigid-board reprojection is too noisy ({:.2f} px)".format(
                    metrics["live_reprojection_rmse_px"]
                )
            )
        if (
            metrics["raw_rmse_mm"] <= args.keypoint_tolerance_mm
            and metrics["rotation_error_deg"] <= args.rotation_tolerance_deg
        ):
            success = True
            break
        if iteration == args.max_moves:
            break

        current_eef = controller.get_data()["position"]
        one_shot = iteration == 0
        servo_target = build_servo_target(
            current_eef=current_eef,
            camera_from_robot_rotation=camera_from_robot_rotation,
            centroid_error_camera=alignment.centroid_error_camera,
            rotation_error_camera=alignment.transform[:3, :3],
            translation_gain=1.0 if one_shot else args.translation_gain,
            rotation_gain=1.0 if one_shot else args.rotation_gain,
            max_translation_step_m=(
                args.one_shot_max_translation_mm
                if one_shot
                else args.max_translation_step_mm
            )
            / 1000.0,
            max_rotation_step_rad=np.deg2rad(
                args.one_shot_max_rotation_deg
                if one_shot
                else args.max_rotation_step_deg
            ),
        )
        translation_step_mm = float(
            np.linalg.norm(servo_target[:3, 3] - current_eef[:3, 3]) * 1000.0
        )
        rotation_step = servo_target[:3, :3] @ current_eef[:3, :3].T
        rotation_step_deg = float(
            np.rad2deg(
                np.arccos(np.clip((np.trace(rotation_step) - 1.0) * 0.5, -1.0, 1.0))
            )
        )
        metrics["next_move_mode"] = "one_shot" if one_shot else "refinement"
        metrics["command_translation_mm"] = translation_step_mm
        metrics["command_rotation_deg"] = rotation_step_deg
        print(
            "  -> {} move: {:.1f} mm, {:.1f} deg".format(
                metrics["next_move_mode"],
                translation_step_mm,
                rotation_step_deg,
            )
        )
        _move_and_settle(controller, servo_target, args.settle_sec)

    result = controller.get_data()
    pose_output = Path(output_dir) / side_name
    pose_output.mkdir(parents=True, exist_ok=True)
    report = {
        "pose_index": int(pose_index),
        "reference_path": str(reference.path),
        "success": success,
        "history": history,
        "final_qpos": result["qpos"].tolist(),
        "final_eef": result["position"].tolist(),
    }
    _write_json(pose_output / "{}_metrics.json".format(pose_index), report)
    if success:
        np.save(pose_output / "{}_qpos.npy".format(pose_index), result["qpos"])
        np.save(pose_output / "{}_aa.npy".format(pose_index), result["position"])
    return report


def _side_specs(args):
    specs = {
        "right": ("Right", args.right_ip, Path(args.right_seed_dir)),
        "left": ("Left", args.left_ip, Path(args.left_seed_dir)),
    }
    if args.side == "both":
        return [specs["left"], specs["right"]]
    return [specs[args.side]]


def _confirm_side_ready(side_name):
    phrase = "{} READY".format(side_name.upper())
    answer = input(
        "Attach the ChArUco marker rigidly to {}, park the other arm, then type '{}': ".format(
            side_name, phrase
        )
    )
    if answer.strip() != phrase:
        raise RuntimeError("Readiness phrase did not match; no robot motion was started")


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-dir", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--side", choices=("right", "left", "both"), default=None)
    parser.add_argument("--right-ip", default=network_info["xarm"]["param"]["ip"])
    parser.add_argument("--left-ip", default=LEFT_XARM_IP)
    parser.add_argument("--right-seed-dir", default=str(DEFAULT_SEED_DIR))
    parser.add_argument("--left-seed-dir", default=str(DEFAULT_SEED_DIR))
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--end", type=int, default=50)
    parser.add_argument("--pose-indices", default=None, help="Comma-separated pose indices")
    parser.add_argument("--max-moves", type=int, default=50)
    parser.add_argument("--keypoint-tolerance-mm", type=float, default=3.0)
    parser.add_argument("--rotation-tolerance-deg", type=float, default=2.0)
    parser.add_argument("--max-registered-rmse-mm", type=float, default=4.0)
    parser.add_argument("--max-live-reprojection-rmse-px", type=float, default=5.0)
    parser.add_argument("--huber-delta-mm", type=float, default=4.0)
    parser.add_argument("--min-common", type=int, default=8)
    parser.add_argument("--translation-gain", type=float, default=0.9)
    parser.add_argument("--rotation-gain", type=float, default=0.9)
    parser.add_argument("--max-translation-step-mm", type=float, default=35.0)
    parser.add_argument("--max-rotation-step-deg", type=float, default=15.0)
    parser.add_argument("--one-shot-max-translation-mm", type=float, default=150.0)
    parser.add_argument("--one-shot-max-rotation-deg", type=float, default=30.0)
    parser.add_argument("--axis-probe-mm", type=float, default=4.0)
    parser.add_argument("--settle-sec", type=float, default=0.6)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--charuco-workers", type=int, default=min(8, os.cpu_count() or 1))
    parser.add_argument("--checker-length-mm", type=float, default=25.0)
    parser.add_argument(
        "--continue-on-failure",
        action="store_true",
        help="Continue to later poses when a pose does not converge",
    )
    parser.add_argument(
        "--resume-latest",
        action="store_true",
        help="Resume the newest incomplete pose-matching session",
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Enable physical robot motion; without this flag only validate inputs",
    )
    return parser


def _validate_args(args):
    positive_values = {
        "max_moves": args.max_moves,
        "keypoint_tolerance_mm": args.keypoint_tolerance_mm,
        "rotation_tolerance_deg": args.rotation_tolerance_deg,
        "max_registered_rmse_mm": args.max_registered_rmse_mm,
        "max_live_reprojection_rmse_px": args.max_live_reprojection_rmse_px,
        "huber_delta_mm": args.huber_delta_mm,
        "min_common": args.min_common,
        "max_translation_step_mm": args.max_translation_step_mm,
        "max_rotation_step_deg": args.max_rotation_step_deg,
        "one_shot_max_translation_mm": args.one_shot_max_translation_mm,
        "one_shot_max_rotation_deg": args.one_shot_max_rotation_deg,
        "axis_probe_mm": args.axis_probe_mm,
        "settle_sec": args.settle_sec,
        "fps": args.fps,
        "charuco_workers": args.charuco_workers,
        "checker_length_mm": args.checker_length_mm,
    }
    invalid = [name for name, value in positive_values.items() if value <= 0]
    if invalid:
        raise ValueError("Arguments must be positive: {}".format(invalid))
    if not 0 < args.translation_gain <= 1 or not 0 < args.rotation_gain <= 1:
        raise ValueError("Servo gains must be in (0, 1]")


def main():
    args = build_parser().parse_args()
    _validate_args(args)
    resume = bool(args.resume_latest)
    if resume and args.output_dir is not None:
        raise ValueError("--output-dir cannot be combined with --resume-latest")

    if resume:
        output_dir = _latest_resume_dir()
        manifest = _read_json(output_dir / "manifest.json")
        reference_dir = Path(manifest["reference_dir"])
        pose_indices = [int(value) for value in manifest["pose_indices"]]
        if args.side is None:
            args.side = "both"
    else:
        if args.side is None:
            args.side = "both"
        reference_dir = (
            Path(args.reference_dir) if args.reference_dir else _latest_reference_dir()
        )
        available = _reference_pose_indices(reference_dir)
        pose_indices = _select_pose_indices(
            available, args.start, args.end, args.pose_indices
        )
        output_dir = None
        manifest = None

    side_specs = _side_specs(args)
    for side_name, _, seed_dir in side_specs:
        completed_reports = {}
        warm_start_qpos = None
        pending_indices = pose_indices
        if resume:
            (
                completed_reports,
                completed_indices,
                pending_indices,
                warm_start_qpos,
            ) = _completed_pose_prefix(output_dir, side_name, pose_indices)
            print(
                "Resume {}: {} completed, {} pending{}".format(
                    side_name,
                    len(completed_indices),
                    len(pending_indices),
                    " (next pose {})".format(pending_indices[0])
                    if pending_indices
                    else "",
                )
            )
        if pending_indices and warm_start_qpos is None:
            _load_seed(seed_dir, pending_indices[0])

    print("Reference single-hand session: {}".format(reference_dir))
    print("Selected poses: {}".format(pose_indices))
    print("Sides: {}".format([spec[0] for spec in side_specs]))
    if resume:
        print("Resuming pose-matching session: {}".format(output_dir))
    if not args.execute:
        print("Validation complete. Re-run with --execute to enable physical motion.")
        return

    if resume:
        intrinsic, extrinsic = load_camparam(output_dir)
        manifest["status"] = "running"
        manifest.pop("error", None)
    else:
        output_dir = (
            Path(args.output_dir)
            if args.output_dir
            else DEFAULT_OUTPUT_ROOT / datetime.now().strftime("%Y%m%d_%H%M%S")
        )
        try:
            output_dir.relative_to(Path.home())
        except ValueError as exc:
            raise ValueError(
                "output-dir must be under {} for remote snapshots".format(Path.home())
            ) from exc
        output_dir.mkdir(parents=True, exist_ok=False)
        output_dir.chmod(0o777)
        intrinsic, extrinsic = load_current_camparam()
        save_current_camparam(output_dir)
        manifest = {
            "reference_dir": str(reference_dir),
            "pose_indices": pose_indices,
            "side_order": [spec[0] for spec in side_specs],
            "status": "running",
            "sides": {},
        }

    existing_side_order = manifest.get("side_order") or list(manifest["sides"])
    for side_name, _, _ in side_specs:
        if side_name not in existing_side_order:
            existing_side_order.append(side_name)
    manifest["side_order"] = existing_side_order
    _write_json(output_dir / "manifest.json", manifest)

    references = prepare_reference_keypoints(
        reference_dir=reference_dir,
        pose_indices=pose_indices,
        intrinsic=intrinsic,
        extrinsic=extrinsic,
        output_dir=output_dir,
        workers=args.charuco_workers,
        checker_length_m=args.checker_length_mm / 1000.0,
        reuse_existing=resume,
    )

    selected_progress = {}
    for side_name, _, _ in side_specs:
        selected_progress[side_name] = _completed_pose_prefix(
            output_dir, side_name, pose_indices
        )
    if all(not progress[2] for progress in selected_progress.values()):
        session_complete = all(
            not _completed_pose_prefix(output_dir, side_name, pose_indices)[2]
            for side_name in manifest["side_order"]
        )
        manifest["status"] = "complete" if session_complete else "partial"
        _write_json(output_dir / "manifest.json", manifest)
        print("All selected sides are already complete in {}".format(output_dir))
        return

    pc_list = get_pc_list()
    rcc = remote_camera_controller("handeye_pose_matching", pc_list=pc_list)
    try:
        rcc.wait_until_ready()
        for side_name, ip, seed_dir in side_specs:
            (
                completed_reports,
                completed_indices,
                pending_indices,
                warm_start_qpos,
            ) = _completed_pose_prefix(output_dir, side_name, pose_indices)
            side_report = manifest["sides"].setdefault(
                side_name, {"axis_probe": None, "poses": {}}
            )
            side_report.setdefault("poses", {}).update(completed_reports)
            if not pending_indices:
                print("{} is already complete; skipping".format(side_name))
                continue

            _confirm_side_ready(side_name)
            controller = None
            try:
                controller = _create_controller(ip)
                detector = SnapshotKeypointDetector(
                    rcc=rcc,
                    intrinsic=intrinsic,
                    extrinsic=extrinsic,
                    observation_root=output_dir / "observations",
                    workers=args.charuco_workers,
                    checker_length_m=args.checker_length_mm / 1000.0,
                    fps=args.fps,
                )
                probe_report = side_report.get("axis_probe")
                if probe_report and "camera_from_robot_rotation" in probe_report:
                    camera_from_robot = np.asarray(
                        probe_report["camera_from_robot_rotation"], dtype=np.float64
                    )
                    if camera_from_robot.shape != (3, 3):
                        raise ValueError("Saved axis-probe rotation must be 3x3")
                    print("Reusing saved {} axis probe".format(side_name))
                else:
                    probe_seed = (
                        warm_start_qpos
                        if warm_start_qpos is not None
                        else _load_seed(seed_dir, pending_indices[0])
                    )
                    _move_and_settle(controller, probe_seed, args.settle_sec)
                    camera_from_robot, probe_report = estimate_axis_rotation(
                        controller=controller,
                        detector=detector,
                        side_name=side_name,
                        probe_m=args.axis_probe_mm / 1000.0,
                        settle_sec=args.settle_sec,
                        min_common=args.min_common,
                    )
                    side_report["axis_probe"] = probe_report
                _write_json(output_dir / "manifest.json", manifest)

                for pose_index in pending_indices:
                    reference = references[pose_index]
                    seed_qpos = (
                        warm_start_qpos
                        if warm_start_qpos is not None
                        else _load_seed(seed_dir, pose_index)
                    )
                    if warm_start_qpos is not None:
                        print(
                            "Warm-starting {} pose {} from the previous optimized pose".format(
                                side_name, pose_index
                            )
                        )
                    report = optimize_pose(
                        controller=controller,
                        detector=detector,
                        side_name=side_name,
                        pose_index=pose_index,
                        reference=reference,
                        seed_qpos=seed_qpos,
                        camera_from_robot_rotation=camera_from_robot,
                        output_dir=output_dir,
                        args=args,
                    )
                    side_report["poses"][str(pose_index)] = report
                    _write_json(output_dir / "manifest.json", manifest)
                    if report["success"]:
                        warm_start_qpos = np.asarray(
                            report["final_qpos"], dtype=np.float64
                        )
                    if not report["success"] and not args.continue_on_failure:
                        raise RuntimeError(
                            "{} pose {} did not converge".format(side_name, pose_index)
                        )
            finally:
                if controller is not None:
                    controller.end(set_break=False)

        session_complete = True
        for side_name in manifest["side_order"]:
            if _completed_pose_prefix(output_dir, side_name, pose_indices)[2]:
                session_complete = False
                break
        manifest["status"] = "complete" if session_complete else "partial"
        _write_json(output_dir / "manifest.json", manifest)
        print("Saved optimized poses to {} ({})".format(output_dir, manifest["status"]))
    except Exception as exc:
        manifest["status"] = "failed"
        manifest["error"] = str(exc)
        _write_json(output_dir / "manifest.json", manifest)
        raise
    finally:
        rcc.end()


if __name__ == "__main__":
    main()
