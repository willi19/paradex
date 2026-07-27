"""Cross-validated reprojection error for a calibrated camera system."""

from itertools import combinations
from typing import Dict, List, Mapping, Optional

import cv2
import numpy as np


def _projection_matrix(intrinsic: Mapping, extrinsic: np.ndarray) -> np.ndarray:
    camera_matrix = np.asarray(intrinsic["intrinsics_undistort"], dtype=np.float64)
    world_to_camera = np.asarray(extrinsic, dtype=np.float64).reshape(3, 4)
    return camera_matrix @ world_to_camera


def _undistort_corners(corners: np.ndarray, intrinsic: Mapping) -> np.ndarray:
    corners = np.asarray(corners, dtype=np.float64).reshape(-1, 2)
    if len(corners) == 0:
        return corners

    return cv2.undistortPoints(
        corners.reshape(-1, 1, 2),
        np.asarray(intrinsic["original_intrinsics"], dtype=np.float64),
        np.asarray(intrinsic["dist_params"], dtype=np.float64),
        P=np.asarray(intrinsic["intrinsics_undistort"], dtype=np.float64),
    ).reshape(-1, 2)


def _triangulate_linear(
    image_points: np.ndarray,
    projection_matrices: np.ndarray,
) -> Optional[np.ndarray]:
    if len(image_points) < 2:
        return None

    rows = []
    for (x, y), projection in zip(image_points, projection_matrices):
        rows.append(x * projection[2] - projection[0])
        rows.append(y * projection[2] - projection[1])

    _, _, vh = np.linalg.svd(np.asarray(rows))
    homogeneous_point = vh[-1]
    if abs(homogeneous_point[3]) < 1e-12:
        return None

    point = homogeneous_point[:3] / homogeneous_point[3]
    return point if np.all(np.isfinite(point)) else None


def _triangulate_point(
    image_points: np.ndarray,
    projection_matrices: np.ndarray,
    inlier_threshold: float,
) -> Optional[np.ndarray]:
    """Triangulate from the largest geometrically consistent source-view set."""
    if len(image_points) <= 2:
        return _triangulate_linear(image_points, projection_matrices)

    best_inliers = None
    best_score = None
    for first, second in combinations(range(len(image_points)), 2):
        pair = np.asarray([first, second])
        candidate = _triangulate_linear(
            image_points[pair],
            projection_matrices[pair],
        )
        if candidate is None:
            continue

        homogeneous_point = np.append(candidate, 1.0)
        projected = projection_matrices @ homogeneous_point
        valid = np.abs(projected[:, 2]) > 1e-12
        projected[valid, :2] /= projected[valid, 2:3]
        errors = np.full(len(image_points), np.inf)
        errors[valid] = np.linalg.norm(
            projected[valid, :2] - image_points[valid],
            axis=1,
        )
        inliers = errors <= inlier_threshold
        inlier_count = int(np.sum(inliers))
        if inlier_count < 2:
            continue

        score = (
            inlier_count,
            -float(np.mean(errors[inliers])),
            -float(np.median(errors[np.isfinite(errors)])),
        )
        if best_score is None or score > best_score:
            best_score = score
            best_inliers = inliers

    if best_inliers is None:
        return None
    return _triangulate_linear(
        image_points[best_inliers],
        projection_matrices[best_inliers],
    )


def calculate_reprojection_errors(
    observations: Mapping[str, Mapping[str, np.ndarray]],
    intrinsics: Mapping[str, Mapping],
    extrinsics: Mapping[str, np.ndarray],
    min_source_views: int = 2,
    source_inlier_threshold: float = 2.0,
) -> Dict[str, np.ndarray]:
    """Compute leave-one-camera-out ChArUco reprojection errors.

    Each point observed by a target camera is triangulated only from the other
    cameras. The resulting 3D point is then projected into the target camera,
    so a moved target camera cannot hide its error by contributing to its own
    triangulation.
    """
    if min_source_views < 2:
        raise ValueError("min_source_views must be at least 2")
    if source_inlier_threshold <= 0:
        raise ValueError("source_inlier_threshold must be positive")

    valid_serials = sorted(set(observations) & set(intrinsics) & set(extrinsics))
    projections = {
        serial: _projection_matrix(intrinsics[serial], extrinsics[serial])
        for serial in valid_serials
    }
    world_to_camera = {
        serial: np.asarray(extrinsics[serial], dtype=np.float64).reshape(3, 4)
        for serial in valid_serials
    }

    points_by_id = {}
    for serial in valid_serials:
        ids = np.asarray(observations[serial]["ids"]).reshape(-1)
        corners = np.asarray(observations[serial]["corners"]).reshape(-1, 2)
        if len(ids) != len(corners):
            raise ValueError(f"Mismatched corner and ID counts for camera {serial}")

        undistorted = _undistort_corners(corners, intrinsics[serial])
        for corner_id, corner in zip(ids, undistorted):
            if not np.all(np.isfinite(corner)):
                continue
            points_by_id.setdefault(int(corner_id), {})[serial] = corner

    errors = {serial: [] for serial in valid_serials}
    for camera_observations in points_by_id.values():
        for target_serial, target_corner in camera_observations.items():
            source_serials = [
                serial for serial in camera_observations if serial != target_serial
            ]
            if len(source_serials) < min_source_views:
                continue

            point_3d = _triangulate_point(
                np.asarray([camera_observations[serial] for serial in source_serials]),
                np.asarray([projections[serial] for serial in source_serials]),
                source_inlier_threshold,
            )
            if point_3d is None:
                continue

            homogeneous_point = np.append(point_3d, 1.0)
            source_depths = np.asarray(
                [
                    world_to_camera[serial][2] @ homogeneous_point
                    for serial in source_serials
                ]
            )
            target_depth = world_to_camera[target_serial][2] @ homogeneous_point
            if target_depth <= 1e-8 or (
                np.sum(source_depths > 1e-8) < min_source_views
            ):
                continue

            projected = projections[target_serial] @ homogeneous_point
            projected = projected[:2] / projected[2]
            errors[target_serial].append(np.linalg.norm(projected - target_corner))

    return {
        serial: np.asarray(camera_errors, dtype=np.float64)
        for serial, camera_errors in errors.items()
    }


def aggregate_reprojection_errors(
    error_sets: List[Mapping[str, np.ndarray]],
) -> Dict[str, np.ndarray]:
    """Concatenate per-camera errors from multiple captured frames."""
    serials = sorted({serial for errors in error_sets for serial in errors})
    return {
        serial: np.concatenate(
            [
                np.asarray(errors[serial], dtype=np.float64)
                for errors in error_sets
                if serial in errors and len(errors[serial]) > 0
            ]
        )
        for serial in serials
        if any(serial in errors and len(errors[serial]) > 0 for errors in error_sets)
    }


def summarize_reprojection_errors(
    errors: Mapping[str, np.ndarray],
) -> Dict[str, Dict[str, float]]:
    """Return count and pixel-error statistics for each camera."""
    summary = {}
    for serial, values in errors.items():
        values = np.asarray(values, dtype=np.float64)
        values = values[np.isfinite(values)]
        if len(values) == 0:
            continue

        summary[serial] = {
            "count": int(len(values)),
            "mean": float(np.mean(values)),
            "median": float(np.median(values)),
            "p95": float(np.percentile(values, 95)),
            "max": float(np.max(values)),
        }
    return summary
