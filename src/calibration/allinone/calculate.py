"""Run camera extrinsic calibration and then xArm hand-eye calibration."""

import argparse
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import cv2
import numpy as np
import tqdm


REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from paradex.calibration.utils import (  # noqa: E402
    cam_param_dir,
    extrinsic_dir,
    get_cammtx,
    handeye_calib_path,
    load_current_intrinsic,
)
from paradex.image.aruco import (  # noqa: E402
    boardinfo_dict,
    detect_charuco,
    merge_charuco_detection,
)
from paradex.image.image_dict import ImageDict  # noqa: E402
from paradex.image.undistort import (  # noqa: E402
    precomute_undistort_map,
    undistort_points,
)
from paradex.utils.file_io import find_latest_directory  # noqa: E402


# The all-in-one capture follows the xArm hand-eye setup, whose ChArUco
# checker spacing is 25 mm rather than the 50 mm extrinsic-calibration board.
HAND_EYE_SQUARE_LENGTH_M = 0.025
CHARUCO_DETECTION_WORKERS = max(1, os.cpu_count() or 1)
EXCLUDED_SERIALS = set()
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tiff"}
UNDISTORT_EXCLUDED_SERIALS = {"22684253"}


def numeric_step_names(root_dir):
    return sorted(
        (
            name
            for name in os.listdir(root_dir)
            if name.isdigit() and os.path.isdir(os.path.join(root_dir, name))
        ),
        key=int,
    )


def validate_allinone_capture(root_dir):
    steps = numeric_step_names(root_dir)
    if len(steps) < 2:
        raise ValueError(
            f"Need at least 2 captured steps in {root_dir}, got {len(steps)}"
        )

    for step in steps:
        step_dir = os.path.join(root_dir, step)
        missing = [
            name
            for name in ("images", "qpos.npy", "eef.npy")
            if not os.path.exists(os.path.join(step_dir, name))
        ]
        if missing:
            raise ValueError(f"Incomplete capture at {step_dir}: missing {missing}")
    return steps


def detect_charuco_file(image_path):
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Failed to read camera image: {image_path}")
    return merge_charuco_detection(detect_charuco(image))


def image_paths_for_step(step_dir, allowed_serials):
    image_dir = Path(step_dir) / "images"
    return [
        path
        for path in sorted(image_dir.iterdir())
        if path.is_file()
        and path.suffix.lower() in IMAGE_EXTENSIONS
        and path.stem in allowed_serials
    ]


def save_charuco_markers(root_dir, steps):
    intrinsics = load_current_intrinsic()
    if not intrinsics:
        raise ValueError("No current camera intrinsics are available.")

    tasks = []
    for step in steps:
        step_dir = os.path.join(root_dir, step)
        marker_dir = os.path.join(step_dir, "markers_2d")
        os.makedirs(marker_dir, exist_ok=True)
        allowed_serials = set(intrinsics) - EXCLUDED_SERIALS
        image_paths = image_paths_for_step(step_dir, allowed_serials)
        if not image_paths:
            raise ValueError(f"No usable camera images at {step_dir}")
        tasks.extend((step, marker_dir, image_path) for image_path in image_paths)

    total_by_step = {step: 0 for step in steps}
    completed_by_step = {step: 0 for step in steps}
    detected_by_step = {step: 0 for step in steps}
    for step, _, _ in tasks:
        total_by_step[step] += 1

    worker_count = min(CHARUCO_DETECTION_WORKERS, len(tasks))
    print(
        f"Detecting charuco in {len(tasks)} images with "
        f"{worker_count} workers..."
    )
    previous_opencv_threads = cv2.getNumThreads()
    cv2.setNumThreads(1)
    try:
        with ThreadPoolExecutor(max_workers=worker_count) as executor:
            futures = {
                executor.submit(detect_charuco_file, image_path): (
                    step,
                    marker_dir,
                    image_path.stem,
                )
                for step, marker_dir, image_path in tasks
            }
            progress = tqdm.tqdm(
                as_completed(futures),
                total=len(futures),
                desc="Detecting charuco",
            )
            for future in progress:
                step, marker_dir, serial = futures[future]
                progress.set_postfix(step=step, camera=serial)
                merged = future.result()
                corners = np.asarray(merged["checkerCorner"], dtype=np.float32)
                ids = np.asarray(merged["checkerIDs"])
                np.save(os.path.join(marker_dir, f"{serial}_corner.npy"), corners)
                np.save(os.path.join(marker_dir, f"{serial}_id.npy"), ids)
                completed_by_step[step] += 1
                if len(ids) > 0:
                    detected_by_step[step] += 1
                if completed_by_step[step] == total_by_step[step]:
                    tqdm.tqdm.write(
                        f"Detected charuco for index {step}: "
                        f"{detected_by_step[step]}/{total_by_step[step]} cameras"
                    )
    finally:
        cv2.setNumThreads(previous_opencv_threads)


def undistort_image_file(image_path, output_path, map_x, map_y):
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Failed to read camera image: {image_path}")
    undistorted = cv2.remap(image, map_x, map_y, cv2.INTER_LINEAR)
    if not cv2.imwrite(
        str(output_path),
        undistorted,
        [cv2.IMWRITE_PNG_COMPRESSION, 1],
    ):
        raise IOError(f"Failed to save undistorted image: {output_path}")


def undistort_images_parallel(root_dir, steps, intrinsics, extrinsics):
    allowed_serials = (
        (set(intrinsics) & set(extrinsics))
        - EXCLUDED_SERIALS
        - UNDISTORT_EXCLUDED_SERIALS
    )
    undistort_maps = {
        serial: precomute_undistort_map(intrinsics[serial])[1:]
        for serial in allowed_serials
    }
    tasks = []
    for step in steps:
        step_dir = os.path.join(root_dir, step)
        output_dir = Path(step_dir) / "undistort" / "images"
        output_dir.mkdir(parents=True, exist_ok=True)
        for image_path in image_paths_for_step(step_dir, allowed_serials):
            output_path = output_dir / f"{image_path.stem}.png"
            map_x, map_y = undistort_maps[image_path.stem]
            tasks.append((step, image_path.stem, image_path, output_path, map_x, map_y))

    if not tasks:
        raise ValueError("No camera images are available for undistortion")

    worker_count = min(CHARUCO_DETECTION_WORKERS, len(tasks))
    previous_opencv_threads = cv2.getNumThreads()
    cv2.setNumThreads(1)
    try:
        with ThreadPoolExecutor(max_workers=worker_count) as executor:
            futures = {
                executor.submit(
                    undistort_image_file,
                    image_path,
                    output_path,
                    map_x,
                    map_y,
                ): (step, serial)
                for step, serial, image_path, output_path, map_x, map_y in tasks
            }
            progress = tqdm.tqdm(
                as_completed(futures),
                total=len(futures),
                desc="Undistorting images",
            )
            for future in progress:
                step, serial = futures[future]
                progress.set_postfix(step=step, camera=serial)
                future.result()
    finally:
        cv2.setNumThreads(previous_opencv_threads)


def load_saved_charuco_detections(step_dir, intrinsics):
    """Load the one-time raw detections and map them to undistorted pixels."""
    marker_dir = Path(step_dir) / "markers_2d"
    detections = {}

    board_ranges = []
    offset = 0
    for board_id, config in boardinfo_dict.items():
        corner_count = (config["numX"] - 1) * (config["numY"] - 1)
        board_ranges.append((board_id, offset, offset + corner_count))
        offset += corner_count

    for corner_path in sorted(marker_dir.glob("*_corner.npy")):
        serial = corner_path.name[: -len("_corner.npy")]
        if serial in EXCLUDED_SERIALS or serial not in intrinsics:
            continue

        id_path = marker_dir / f"{serial}_id.npy"
        if not id_path.is_file():
            raise FileNotFoundError(f"Missing ChArUco IDs: {id_path}")

        corners = np.asarray(np.load(corner_path), dtype=np.float64).reshape(-1, 2)
        ids = np.asarray(np.load(id_path), dtype=np.int64).reshape(-1)
        if len(corners) != len(ids):
            raise ValueError(
                f"ChArUco corner/ID count mismatch for {serial} in {step_dir}: "
                f"{len(corners)} != {len(ids)}"
            )

        undistorted = (
            np.asarray(undistort_points(corners, intrinsics[serial])).reshape(-1, 2)
            if len(corners)
            else corners
        )
        camera_detection = {}
        assigned = np.zeros(len(ids), dtype=bool)
        for board_id, start, end in board_ranges:
            mask = (ids >= start) & (ids < end)
            if not np.any(mask):
                continue
            assigned |= mask
            camera_detection[board_id] = {
                "checkerCorner": undistorted[mask],
                "checkerIDs": ids[mask] - start,
            }
        if np.any(~assigned):
            unknown = np.unique(ids[~assigned]).tolist()
            raise ValueError(f"Unknown merged ChArUco IDs for {serial}: {unknown}")
        detections[serial] = camera_detection

    return detections


def load_all_saved_charuco_detections(root_dir, steps, intrinsics):
    worker_count = min(CHARUCO_DETECTION_WORKERS, len(steps))
    with ThreadPoolExecutor(max_workers=worker_count) as executor:
        futures = {
            executor.submit(
                load_saved_charuco_detections,
                os.path.join(root_dir, step),
                intrinsics,
            ): step
            for step in steps
        }
        detections_by_step = {}
        progress = tqdm.tqdm(
            as_completed(futures),
            total=len(futures),
            desc="Loading cached charuco",
        )
        for future in progress:
            step = futures[future]
            progress.set_postfix(step=step)
            detections_by_step[step] = future.result()
    return detections_by_step


def triangulate_saved_charuco_step(
    root_dir,
    step,
    intrinsics,
    extrinsics,
    id_file_name,
    corner_file_name,
    detections=None,
):
    step_dir = os.path.join(root_dir, step)
    id_path = os.path.join(step_dir, id_file_name)
    corner_path = os.path.join(step_dir, corner_file_name)
    for stale_path in (id_path, corner_path):
        if os.path.isfile(stale_path):
            os.remove(stale_path)

    if detections is None:
        detections = load_saved_charuco_detections(step_dir, intrinsics)
    detections = {
        serial: detection
        for serial, detection in detections.items()
        if serial not in EXCLUDED_SERIALS
        and serial in intrinsics
        and serial in extrinsics
    }
    if not detections:
        raise ValueError(f"No saved ChArUco detections for index {step}")
    triangulator = ImageDict({}, intrinsics, extrinsics)
    charuco_3d = merge_charuco_detection(
        triangulator.triangulate_charuco(detections=detections)
    )
    ids = np.asarray(charuco_3d["checkerIDs"]).reshape(-1)
    corners = np.asarray(charuco_3d["checkerCorner"], dtype=np.float64)
    if len(ids) == 0:
        np.save(id_path, ids)
        corners = np.empty((0, 3), dtype=np.float64)
        np.save(corner_path, corners)
        return ids, corners
    corners = corners.reshape(-1, 3)
    np.save(id_path, ids)
    np.save(corner_path, corners)
    return ids, corners


def triangulate_saved_charuco(
    root_dir,
    steps,
    intrinsics,
    extrinsics,
    id_file_name,
    corner_file_name,
    detections_by_step=None,
):
    """Triangulate cached 2D detections once, in parallel across poses."""
    worker_count = min(CHARUCO_DETECTION_WORKERS, len(steps))
    with ThreadPoolExecutor(max_workers=worker_count) as executor:
        futures = {
            executor.submit(
                triangulate_saved_charuco_step,
                root_dir,
                step,
                intrinsics,
                extrinsics,
                id_file_name,
                corner_file_name,
                None if detections_by_step is None else detections_by_step[step],
            ): step
            for step in steps
        }
        progress = tqdm.tqdm(
            as_completed(futures),
            total=len(futures),
            desc="Triangulating 3D keypoints",
        )
        triangulated_by_step = {}
        for future in progress:
            step = futures[future]
            progress.set_postfix(step=step)
            ids, corners = future.result()
            triangulated_by_step[step] = {"ids": ids, "corners": corners}
            if len(ids) == 0:
                tqdm.tqdm.write(
                    f"No charuco corners triangulated for index {step}."
                )
    return triangulated_by_step


def scale_triangulated_points(points, scale):
    return np.asarray(points, dtype=np.float64) * float(scale)


def scale_triangulated_step(root_dir, step, scale, triangulated=None):
    step_dir = os.path.join(root_dir, step)
    if triangulated is None:
        ids = np.asarray(
            np.load(os.path.join(step_dir, "kypt_3d_id.npy"))
        ).reshape(-1)
        corners = np.asarray(
            np.load(os.path.join(step_dir, "kypt_3d_cor.npy")),
            dtype=np.float64,
        ).reshape(-1, 3)
    else:
        ids = triangulated["ids"]
        corners = triangulated["corners"]
    np.save(os.path.join(step_dir, "charuco_3d_ids.npy"), ids)
    np.save(
        os.path.join(step_dir, "charuco_3d_corners.npy"),
        scale_triangulated_points(corners, scale),
    )


def scale_triangulated_charuco(
    root_dir, steps, scale, triangulated_by_step=None
):
    worker_count = min(CHARUCO_DETECTION_WORKERS, len(steps))
    with ThreadPoolExecutor(max_workers=worker_count) as executor:
        futures = {
            executor.submit(
                scale_triangulated_step,
                root_dir,
                step,
                scale,
                None if triangulated_by_step is None else triangulated_by_step[step],
            ): step
            for step in steps
        }
        progress = tqdm.tqdm(
            as_completed(futures),
            total=len(futures),
            desc="Scaling hand-eye 3D points",
        )
        for future in progress:
            progress.set_postfix(step=futures[future])
            future.result()


def project_points(points, projection_matrix):
    homogeneous = np.column_stack((points, np.ones(len(points))))
    projected = (projection_matrix @ homogeneous.T).T
    return projected[:, :2] / projected[:, 2:3]


def calculate_step_reprojection(
    root_dir,
    step,
    intrinsics,
    projection_matrices,
    detections=None,
    triangulated=None,
):
    step_dir = os.path.join(root_dir, step)
    if triangulated is None:
        ids = np.asarray(
            np.load(os.path.join(step_dir, "kypt_3d_id.npy"))
        ).reshape(-1)
        corners = np.asarray(
            np.load(os.path.join(step_dir, "kypt_3d_cor.npy")), dtype=np.float64
        ).reshape(-1, 3)
    else:
        ids = triangulated["ids"]
        corners = triangulated["corners"]
    if len(ids) == 0:
        return {}, []

    if detections is None:
        detections = load_saved_charuco_detections(step_dir, intrinsics)
    step_errors = {}
    debug_tasks = []
    for serial, detection in detections.items():
        if serial not in projection_matrices:
            continue
        projected = project_points(corners, projection_matrices[serial])
        merged = merge_charuco_detection(detection)
        observed_ids = np.asarray(merged["checkerIDs"]).reshape(-1)
        observed_corners = np.asarray(merged["checkerCorner"]).reshape(-1, 2)
        common_ids, observed_indices, point_indices = np.intersect1d(
            observed_ids,
            ids,
            return_indices=True,
        )
        if len(common_ids) == 0:
            continue
        step_errors[serial] = np.linalg.norm(
            observed_corners[observed_indices] - projected[point_indices],
            axis=1,
        ).tolist()
        debug_tasks.append((step, serial, observed_corners, projected))
    return step_errors, debug_tasks


def save_reprojection_debug_image(image_path, output_path, observed, projected):
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Failed to read undistorted image: {image_path}")
    for point in observed:
        cv2.circle(image, tuple(np.rint(point).astype(int)), 3, (0, 255, 0), -1)
    for point in projected:
        cv2.circle(image, tuple(np.rint(point).astype(int)), 3, (255, 0, 0), -1)
    if not cv2.imwrite(
        str(output_path),
        image,
        [cv2.IMWRITE_PNG_COMPRESSION, 1],
    ):
        raise IOError(f"Failed to save reprojection debug image: {output_path}")


def calculate_reprojection_errors(
    root_dir,
    steps,
    intrinsics,
    extrinsics,
    detections_by_step=None,
    triangulated_by_step=None,
):
    error_dict = {}
    projection_matrices = get_cammtx(intrinsics, extrinsics)
    debug_tasks = []
    worker_count = min(CHARUCO_DETECTION_WORKERS, len(steps))
    with ThreadPoolExecutor(max_workers=worker_count) as executor:
        futures = {
            executor.submit(
                calculate_step_reprojection,
                root_dir,
                step,
                intrinsics,
                projection_matrices,
                None if detections_by_step is None else detections_by_step[step],
                None if triangulated_by_step is None else triangulated_by_step[step],
            ): step
            for step in steps
        }
        progress = tqdm.tqdm(
            as_completed(futures),
            total=len(futures),
            desc="Calculating reprojection errors",
        )
        for future in progress:
            progress.set_postfix(step=futures[future])
            step_errors, step_debug_tasks = future.result()
            for serial, errors in step_errors.items():
                error_dict.setdefault(serial, []).extend(errors)
            debug_tasks.extend(step_debug_tasks)

    if debug_tasks:
        previous_opencv_threads = cv2.getNumThreads()
        cv2.setNumThreads(1)
        try:
            with ThreadPoolExecutor(
                max_workers=min(CHARUCO_DETECTION_WORKERS, len(debug_tasks))
            ) as executor:
                futures = {}
                for step, serial, observed, projected in debug_tasks:
                    output_dir = Path(root_dir) / step / "reproj_debug" / "images"
                    output_dir.mkdir(parents=True, exist_ok=True)
                    image_path = (
                        Path(root_dir)
                        / step
                        / "undistort"
                        / "images"
                        / f"{serial}.png"
                    )
                    output_path = output_dir / f"{serial}.png"
                    future = executor.submit(
                        save_reprojection_debug_image,
                        image_path,
                        output_path,
                        observed,
                        projected,
                    )
                    futures[future] = (step, serial)
                progress = tqdm.tqdm(
                    as_completed(futures),
                    total=len(futures),
                    desc="Saving reprojection debug images",
                )
                for future in progress:
                    step, serial = futures[future]
                    progress.set_postfix(step=step, camera=serial)
                    future.result()
        finally:
            cv2.setNumThreads(previous_opencv_threads)

    return error_dict


def json_ready(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, dict):
        return {key: json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    return value


def save_camera_parameters(root_dir, session_name, intrinsics, extrinsics):
    session_cam_param_dir = os.path.join(cam_param_dir, session_name)
    capture_cam_param_dir = os.path.join(root_dir, "0", "cam_param")

    for output_dir in (session_cam_param_dir, capture_cam_param_dir):
        os.makedirs(output_dir, exist_ok=True)
        with open(os.path.join(output_dir, "intrinsics.json"), "w") as stream:
            json.dump(json_ready(intrinsics), stream, indent=4)
        with open(os.path.join(output_dir, "extrinsics.json"), "w") as stream:
            json.dump(json_ready(extrinsics), stream, indent=4)

    print(f"Saved camera parameters to {session_cam_param_dir}")
    print(f"Copied camera parameters into {capture_cam_param_dir}")


def save_reprojection_errors(root_dir, first_step, error_dict):
    output_path = os.path.join(root_dir, first_step, "reproj_error.txt")
    with open(output_path, "w") as stream:
        for serial in sorted(error_dict):
            errors = np.asarray(error_dict[serial], dtype=np.float64)
            if errors.size == 0:
                continue
            line = (
                f"{serial} : mean {np.mean(errors):.4f}, "
                f"max {np.max(errors):.4f}"
            )
            stream.write(f"{line}\n")
            print(line)
    print(f"Saved reprojection errors to {output_path}")
    return output_path


def get_reconstructed_length(triangulated_by_step, adjacent_ids):
    lengths = []
    for triangulated in triangulated_by_step.values():
        points_by_id = dict(zip(triangulated["ids"], triangulated["corners"]))
        for marker_id, point in points_by_id.items():
            for adjacent_id in adjacent_ids[marker_id]:
                if adjacent_id in points_by_id:
                    lengths.append(np.linalg.norm(point - points_by_id[adjacent_id]))
    if not lengths:
        raise ValueError("No adjacent ChArUco corners were triangulated")
    print("Length mean:", np.mean(lengths))
    print("Length std:", np.std(lengths))
    print("Max length:", np.max(lengths))
    print("Min length:", np.min(lengths))
    return np.mean(lengths)


def calculate_camera_stage(name, root_dir, steps):
    from src.calibration.extrinsic import calculate as camera_calibration

    print("Calculating camera extrinsic calibration...")
    save_charuco_markers(root_dir, steps)
    camera_calibration.run_calibration(name)

    colmap_dir = os.path.join(root_dir, steps[0], "colmap")
    intrinsics, extrinsics = camera_calibration.load_colmap_camparam(colmap_dir)
    undistort_images_parallel(root_dir, steps, intrinsics, extrinsics)
    detections_by_step = load_all_saved_charuco_detections(
        root_dir, steps, intrinsics
    )
    triangulated_by_step = triangulate_saved_charuco(
        root_dir,
        steps,
        intrinsics,
        extrinsics,
        "kypt_3d_id.npy",
        "kypt_3d_cor.npy",
        detections_by_step,
    )

    reconstructed_length = get_reconstructed_length(
        triangulated_by_step,
        camera_calibration.get_adjecent_ids(),
    )
    if not np.isfinite(reconstructed_length) or reconstructed_length <= 0:
        raise ValueError(
            f"Invalid reconstructed ChArUco spacing: {reconstructed_length}"
        )
    reprojection_errors = calculate_reprojection_errors(
        root_dir,
        steps,
        intrinsics,
        extrinsics,
        detections_by_step,
        triangulated_by_step,
    )
    save_reprojection_errors(root_dir, steps[0], reprojection_errors)
    scale = HAND_EYE_SQUARE_LENGTH_M / reconstructed_length

    scaled_extrinsics = {}
    for serial, extrinsic in extrinsics.items():
        scaled = np.asarray(extrinsic, dtype=np.float64).copy()
        scaled[:3, 3] *= scale
        scaled_extrinsics[serial] = scaled

    save_camera_parameters(
        root_dir,
        os.path.basename(root_dir),
        intrinsics,
        scaled_extrinsics,
    )
    print(
        "ChArUco square length: "
        f"{HAND_EYE_SQUARE_LENGTH_M * 1000:.1f}mm (xArm hand-eye board)"
    )
    print(f"Camera scale factor: {scale:.8f}")
    return scale, triangulated_by_step


def calculate_handeye_stage(
    root_dir,
    arm,
    steps,
    scale,
    triangulated_by_step=None,
):
    from src.calibration.handeye.xarm import calculate as handeye_calibration

    print("Calculating xArm calibration...")
    scale_triangulated_charuco(
        root_dir, steps, scale, triangulated_by_step
    )
    session_name = os.path.basename(os.path.normpath(root_dir))
    output_dir = os.path.join(handeye_calib_path, session_name, steps[0])
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "C2R.npy")
    handeye_calibration.calculate_sequence(
        root_dir,
        arm,
        output_path,
        precomputed_charuco=True,
    )
    return output_path


def format_elapsed(seconds):
    minutes, remaining_seconds = divmod(float(seconds), 60.0)
    hours, minutes = divmod(int(minutes), 60)
    if hours:
        return f"{hours:d}h {minutes:d}m {remaining_seconds:.2f}s"
    if minutes:
        return f"{minutes:d}m {remaining_seconds:.2f}s"
    return f"{remaining_seconds:.2f}s"


def print_timing_summary(extrinsic_seconds, handeye_seconds, total_seconds):
    print("Calibration timing:")
    print(f"  Extrinsic: {format_elapsed(extrinsic_seconds)}")
    print(f"  Hand-eye:  {format_elapsed(handeye_seconds)}")
    print(f"  Total:     {format_elapsed(total_seconds)}")


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Calculate camera extrinsics first, then xArm hand-eye calibration "
            "from one all-in-one capture session."
        )
    )
    parser.add_argument("--name", default=None, help="Capture session timestamp.")
    parser.add_argument("--arm", default="xarm")
    args = parser.parse_args()

    name = args.name or find_latest_directory(extrinsic_dir)
    root_dir = os.path.join(extrinsic_dir, name)
    if not os.path.isdir(root_dir):
        raise FileNotFoundError(f"Calibration session does not exist: {root_dir}")

    total_started = time.perf_counter()
    steps = validate_allinone_capture(root_dir)
    extrinsic_started = time.perf_counter()
    scale, triangulated_by_step = calculate_camera_stage(name, root_dir, steps)
    extrinsic_seconds = time.perf_counter() - extrinsic_started
    handeye_started = time.perf_counter()
    c2r_path = calculate_handeye_stage(
        root_dir,
        args.arm,
        steps,
        scale,
        triangulated_by_step,
    )
    handeye_seconds = time.perf_counter() - handeye_started
    total_seconds = time.perf_counter() - total_started
    print("All-in-one calibration complete.")
    print(
        "Camera parameters: "
        f"{os.path.join(cam_param_dir, os.path.basename(root_dir))}"
    )
    print(f"Hand-eye C2R: {c2r_path}")
    print_timing_summary(extrinsic_seconds, handeye_seconds, total_seconds)


if __name__ == "__main__":
    main()
