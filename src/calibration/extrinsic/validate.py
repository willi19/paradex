"""Quick live validation of the current multi-camera extrinsics.

This uses the existing capture-PC ``save`` command without requiring a client
update. Full-resolution images and ChArUco observations are written to a
temporary extrinsic workspace and removed when the program exits.
"""

import argparse
import os
import shutil
import tempfile
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

from paradex.calibration.extrinsic_validation import (
    aggregate_reprojection_errors,
    calculate_reprojection_errors,
    summarize_reprojection_errors,
)
from paradex.calibration.utils import (
    cam_param_dir,
    extrinsic_dir,
    load_current_camparam,
)
from paradex.image.aruco import draw_charuco
from paradex.image.merge import merge_image
from paradex.io.camera_system.remote_camera_controller import (
    remote_camera_controller,
)
from paradex.io.capture_pc.command_sender import CommandSender
from paradex.io.capture_pc.data_sender import DataCollector
from paradex.io.capture_pc.ssh import run_script
from paradex.utils.file_io import find_latest_directory
from paradex.utils.system import get_pc_list


EXCLUDED_PCS = set()
TEMP_ROOT_MARKER = ".extrinsic_validation_tmp"


def _decode_preview_corners(item):
    data = item.get("data", b"")
    values = np.frombuffer(data, dtype=np.float32)
    if values.size % 2 != 0:
        return None

    return values.reshape(-1, 2).copy()


def _create_validation_root():
    prefix = datetime.now().strftime("%Y%m%d_%H%M%S_validation_")
    root = Path(tempfile.mkdtemp(prefix=prefix, dir=extrinsic_dir))
    root.chmod(0o777)
    (root / TEMP_ROOT_MARKER).write_text("temporary live extrinsic validation\n")
    return root


def _create_capture_directory(validation_root):
    capture_name = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    capture_path = validation_root / capture_name
    (capture_path / "markers_2d").mkdir(parents=True)
    (capture_path / "images").mkdir()
    capture_path.chmod(0o777)
    (capture_path / "markers_2d").chmod(0o777)
    (capture_path / "images").chmod(0o777)
    return capture_path


def _assert_owned_temp_path(path, validation_root):
    path = Path(path).resolve()
    validation_root = Path(validation_root).resolve()
    if not (validation_root / TEMP_ROOT_MARKER).is_file():
        raise RuntimeError(f"Temporary validation marker missing: {validation_root}")
    if path != validation_root and path.parent != validation_root:
        raise RuntimeError(f"Refusing to remove non-validation path: {path}")
    return path


def _remove_temp_path(path, validation_root):
    path = _assert_owned_temp_path(path, validation_root)
    if path.exists():
        shutil.rmtree(path)


def _wait_for_save(collector, baseline_save_ids, timeout):
    pending = set(baseline_save_ids)
    deadline = time.monotonic() + timeout
    while pending and time.monotonic() < deadline:
        for item_name, item in collector.get_data().items():
            if item.get("type") != "image" or item_name not in pending:
                continue
            if int(item.get("save_id", 0)) > baseline_save_ids[item_name]:
                pending.remove(item_name)
        if pending:
            time.sleep(0.01)
    return pending


def _load_saved_observations(capture_path, intrinsics, extrinsics):
    markers_path = Path(capture_path) / "markers_2d"
    observations = {}
    for corner_path in markers_path.glob("*_corner.npy"):
        serial = corner_path.name.split("_")[0]
        id_path = markers_path / f"{serial}_id.npy"
        if serial not in intrinsics or serial not in extrinsics or not id_path.exists():
            continue
        observations[serial] = {
            "corners": np.load(corner_path),
            "ids": np.load(id_path),
        }
    return observations


def _observation_diagnostics(observations):
    views_by_id = {}
    nonempty_cameras = 0
    for serial, observation in observations.items():
        ids = np.asarray(observation["ids"]).reshape(-1)
        if len(ids) > 0:
            nonempty_cameras += 1
        for corner_id in ids:
            views_by_id.setdefault(int(corner_id), set()).add(serial)
    max_shared_views = max((len(serials) for serials in views_by_id.values()), default=0)
    return nonempty_cameras, len(views_by_id), max_shared_views


def _print_summary(errors, title, warn_threshold):
    summary = summarize_reprojection_errors(errors)
    print(f"\n{title}")
    if not summary:
        print("  No corners had enough source views for reprojection.")
        return

    print(
        f"{'camera':>10}  {'points':>6}  {'mean':>8}  {'median':>8}"
        f"  {'p95':>8}  {'max':>8}  status"
    )
    ordered = sorted(
        summary.items(),
        key=lambda item: item[1]["median"],
        reverse=True,
    )
    for serial, stats in ordered:
        status = "WARN" if stats["median"] > warn_threshold else "OK"
        print(
            f"{serial:>10}  {stats['count']:6d}  {stats['mean']:8.3f}"
            f"  {stats['median']:8.3f}  {stats['p95']:8.3f}"
            f"  {stats['max']:8.3f}  {status}"
        )


def _load_parameters(name):
    calibration_name = name or find_latest_directory(cam_param_dir)
    if calibration_name is None:
        raise FileNotFoundError(f"No camera parameters found under {cam_param_dir}")

    calibration_path = os.path.join(cam_param_dir, calibration_name)
    intrinsics, extrinsics = load_current_camparam(calibration_name)
    return calibration_name, calibration_path, intrinsics, extrinsics


def run(args):
    calibration_name, calibration_path, intrinsics, extrinsics = _load_parameters(
        args.cam_param
    )
    latest_extrinsic = find_latest_directory(extrinsic_dir)
    pc_list = [
        pc
        for pc in get_pc_list()
        if pc not in EXCLUDED_PCS and pc not in set(args.exclude_pc)
    ]

    print(f"Camera parameters: {calibration_path}")
    if latest_extrinsic is not None:
        print(f"Latest extrinsic workspace: {os.path.join(extrinsic_dir, latest_extrinsic)}")
    print(f"Calibrated cameras: {len(set(intrinsics) & set(extrinsics))}")
    print("Hold the ChArUco boards still, press 'c' to capture, or 'q' to finish.")

    rcc = None
    collector = None
    command_sender = None
    validation_root = _create_validation_root()
    captured_errors = []
    images = {}
    image_frames = {}
    image_received_at = {}
    save_ids = {}
    preview_corners = {}

    try:
        print(f"Temporary capture workspace: {validation_root}")
        rcc = remote_camera_controller("extrinsic_calibration", pc_list=pc_list)
        rcc.start("stream", False, fps=args.fps)
        run_script(
            "python src/calibration/extrinsic/client.py",
            pc_list=pc_list,
            log=True,
        )

        collector = DataCollector(pc_list=pc_list)
        collector.start()
        command_sender = CommandSender(
            pc_list=pc_list,
            timeout=args.command_timeout_ms,
        )

        while True:
            all_data = collector.get_data()
            for item_name, item in all_data.items():
                item_type = item.get("type")
                frame_id = int(item.get("frame_id", 0))

                if item_type == "image":
                    save_ids[item_name] = int(item.get("save_id", 0))
                    if image_frames.get(item_name) == frame_id:
                        continue
                    image_bytes = item.get("data")
                    if not image_bytes:
                        continue
                    image = cv2.imdecode(
                        np.frombuffer(image_bytes, np.uint8),
                        cv2.IMREAD_COLOR,
                    )
                    if image is not None:
                        images[item_name] = image
                        image_frames[item_name] = frame_id
                        image_received_at[item_name] = time.monotonic()

                elif item_type == "charuco_detection":
                    suffix = "_corners"
                    serial = (
                        item_name[: -len(suffix)]
                        if item_name.endswith(suffix)
                        else item_name
                    )
                    corners = _decode_preview_corners(item)
                    if corners is not None:
                        preview_corners[serial] = {
                            "corners": corners,
                            "frame_id": frame_id,
                        }

            if images:
                display_images = {
                    serial: image.copy() for serial, image in images.items()
                }
                for serial, detection in preview_corners.items():
                    if serial not in display_images or (
                        detection["frame_id"] != image_frames.get(serial)
                    ):
                        continue
                    draw_charuco(
                        display_images[serial],
                        detection["corners"],
                        color=(0, 255, 0),
                        radius=1,
                        thickness=-1,
                    )

                frame_text = {
                    serial: str(image_frames[serial]) for serial in display_images
                }
                merged_image = merge_image(display_images, frame_text)
                target_text = (
                    "c: capture "
                    f"{len(captured_errors)}/{args.captures or 'unlimited'}"
                    "    q: finish"
                )
                cv2.putText(
                    merged_image,
                    target_text,
                    (20, 35),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 0, 255),
                    2,
                    cv2.LINE_AA,
                )
            else:
                merged_image = np.full((600, 800, 3), 255, dtype=np.uint8)
                cv2.putText(
                    merged_image,
                    "Waiting for camera streams...",
                    (50, 300),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 0, 0),
                    2,
                )

            cv2.imshow("Extrinsic validation", merged_image)
            key = cv2.waitKey(1) & 0xFF

            if key == ord("q"):
                break

            if key == ord("c"):
                capture_time = time.monotonic()
                baseline_save_ids = {
                    serial: save_ids[serial]
                    for serial in images
                    if serial in save_ids
                    and capture_time - image_received_at.get(serial, 0.0)
                    <= args.max_stream_age
                }
                if not baseline_save_ids:
                    print("Capture skipped: no live camera streams were found.")
                    continue

                capture_path = _create_capture_directory(validation_root)
                command_sender.send_command("save", True)
                pending = _wait_for_save(
                    collector,
                    baseline_save_ids,
                    args.save_timeout,
                )
                if pending:
                    print(
                        "Capture skipped: timed out waiting for "
                        f"{len(pending)} cameras: {', '.join(sorted(pending))}"
                    )
                    break

                current_observations = _load_saved_observations(
                    capture_path,
                    intrinsics,
                    extrinsics,
                )
                frame_errors = calculate_reprojection_errors(
                    current_observations,
                    intrinsics,
                    extrinsics,
                    min_source_views=args.min_source_views,
                    source_inlier_threshold=args.source_inlier_threshold,
                )
                if not any(len(values) > 0 for values in frame_errors.values()):
                    nonempty, unique_corners, max_shared = _observation_diagnostics(
                        current_observations
                    )
                    print(
                        "Capture skipped: "
                        f"{nonempty} cameras detected {unique_corners} unique corners; "
                        f"the best corner was shared by {max_shared} cameras "
                        f"(need {args.min_source_views + 1})."
                    )
                    continue

                captured_errors.append(frame_errors)
                _print_summary(
                    frame_errors,
                    f"Capture {len(captured_errors)} reprojection error (pixels)",
                    args.warn_threshold,
                )
                if args.captures and len(captured_errors) >= args.captures:
                    break

    except KeyboardInterrupt:
        print("\nInterrupted; reporting captures collected so far.")
    finally:
        cv2.destroyAllWindows()
        try:
            if command_sender is not None:
                command_sender.end()
        finally:
            try:
                if collector is not None:
                    collector.end()
            finally:
                try:
                    if rcc is not None:
                        try:
                            rcc.stop()
                        finally:
                            rcc.end()
                finally:
                    _remove_temp_path(validation_root, validation_root)

    if not captured_errors:
        print("No valid validation captures were collected.")
        return

    aggregate = aggregate_reprojection_errors(captured_errors)
    _print_summary(
        aggregate,
        (
            f"Aggregate reprojection error from {len(captured_errors)} captures "
            f"(pixels, parameters {calibration_name})"
        ),
        args.warn_threshold,
    )


def parse_args():
    parser = argparse.ArgumentParser(
        description="Validate current extrinsics without retaining captured images."
    )
    parser.add_argument(
        "--cam-param",
        help="Timestamp under ~/shared_data/cam_param (default: latest).",
    )
    parser.add_argument(
        "--captures",
        type=int,
        default=5,
        help="Number of valid 'c' captures before exit; 0 waits for 'q'.",
    )
    parser.add_argument(
        "--min-source-views",
        type=int,
        default=2,
        help="Other cameras required to triangulate each tested corner.",
    )
    parser.add_argument(
        "--warn-threshold",
        type=float,
        default=2.0,
        help="Mark cameras whose median error exceeds this pixel value.",
    )
    parser.add_argument(
        "--source-inlier-threshold",
        type=float,
        default=2.0,
        help="Pixel threshold for rejecting inconsistent triangulation views.",
    )
    parser.add_argument(
        "--save-timeout",
        type=float,
        default=10.0,
        help="Seconds to wait for existing clients to finish a temporary save.",
    )
    parser.add_argument(
        "--max-stream-age",
        type=float,
        default=2.0,
        help="Only wait for cameras whose preview updated within this many seconds.",
    )
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument(
        "--exclude-pc",
        action="append",
        default=[],
        help="Capture PC to omit; may be repeated.",
    )
    parser.add_argument(
        "--command-timeout-ms",
        type=int,
        default=5000,
        help="Remote shutdown command timeout.",
    )
    args = parser.parse_args()
    if args.captures < 0:
        parser.error("--captures must be non-negative")
    if args.min_source_views < 2:
        parser.error("--min-source-views must be at least 2")
    if args.warn_threshold <= 0:
        parser.error("--warn-threshold must be positive")
    if args.source_inlier_threshold <= 0:
        parser.error("--source-inlier-threshold must be positive")
    if args.save_timeout <= 0:
        parser.error("--save-timeout must be positive")
    if args.max_stream_age <= 0:
        parser.error("--max-stream-age must be positive")
    if args.fps <= 0:
        parser.error("--fps must be positive")
    if args.command_timeout_ms <= 0:
        parser.error("--command-timeout-ms must be positive")
    return args


if __name__ == "__main__":
    run(parse_args())
