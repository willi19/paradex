"""Geometry helpers for closed-loop hand-eye pose matching.

The live optimizer aligns triangulated ChArUco corners to a reference capture.
This module contains only deterministic geometry so it can be tested without a
camera or a physical robot.
"""

from dataclasses import dataclass

import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation


@dataclass(frozen=True)
class KeypointAlignment:
    """Rigid alignment and loss diagnostics for one reference/current pair."""

    common_ids: np.ndarray
    current_points: np.ndarray
    target_points: np.ndarray
    transform: np.ndarray
    centroid_error_camera: np.ndarray
    raw_rmse_m: float
    registered_rmse_m: float
    robust_loss_m2: float
    translation_error_m: float
    rotation_error_rad: float


@dataclass(frozen=True)
class MultiviewPoseFit:
    """Rigid board pose fitted directly against multi-camera image keypoints."""

    transform: np.ndarray
    reprojection_rmse_px: float
    observation_count: int
    camera_count: int


def _validate_keypoints(ids, points, name):
    ids = np.asarray(ids).reshape(-1)
    points = np.asarray(points, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("{} points must have shape (N, 3), got {}".format(name, points.shape))
    if len(ids) != len(points):
        raise ValueError(
            "{} ids/points length mismatch: {} != {}".format(name, len(ids), len(points))
        )
    if len(np.unique(ids)) != len(ids):
        raise ValueError("{} keypoint ids must be unique".format(name))
    if not np.all(np.isfinite(points)):
        raise ValueError("{} points contain non-finite values".format(name))
    return ids.astype(np.int64), points


def match_keypoints(target_ids, target_points, current_ids, current_points, min_common=6):
    """Return target/current ChArUco points ordered by their shared global IDs."""

    target_ids, target_points = _validate_keypoints(target_ids, target_points, "target")
    current_ids, current_points = _validate_keypoints(current_ids, current_points, "current")

    common_ids = np.intersect1d(target_ids, current_ids)
    if len(common_ids) < int(min_common):
        raise ValueError(
            "Need at least {} common keypoints, got {}".format(min_common, len(common_ids))
        )

    target_lookup = {int(keypoint_id): index for index, keypoint_id in enumerate(target_ids)}
    current_lookup = {int(keypoint_id): index for index, keypoint_id in enumerate(current_ids)}
    target_indices = [target_lookup[int(keypoint_id)] for keypoint_id in common_ids]
    current_indices = [current_lookup[int(keypoint_id)] for keypoint_id in common_ids]
    return common_ids, target_points[target_indices], current_points[current_indices]


def project_to_rotation(matrix):
    """Return the closest proper rotation matrix in Frobenius norm."""

    matrix = np.asarray(matrix, dtype=np.float64)
    if matrix.shape != (3, 3):
        raise ValueError("Rotation candidate must have shape (3, 3)")
    u, _, vt = np.linalg.svd(matrix)
    rotation = u @ vt
    if np.linalg.det(rotation) < 0:
        u[:, -1] *= -1
        rotation = u @ vt
    return rotation


def _weighted_rigid_transform(source, target, weights):
    source = np.asarray(source, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64).reshape(-1)
    weight_sum = float(np.sum(weights))
    if weight_sum <= 0:
        raise ValueError("Rigid alignment weights must have positive sum")
    weights = weights / weight_sum

    source_centroid = np.sum(source * weights[:, None], axis=0)
    target_centroid = np.sum(target * weights[:, None], axis=0)
    source_centered = source - source_centroid
    target_centered = target - target_centroid
    covariance = (source_centered * weights[:, None]).T @ target_centered
    if np.linalg.matrix_rank(covariance, tol=1e-10) < 2:
        raise ValueError("Common keypoints are geometrically degenerate")

    u, _, vt = np.linalg.svd(covariance)
    rotation = vt.T @ u.T
    if np.linalg.det(rotation) < 0:
        vt[-1] *= -1
        rotation = vt.T @ u.T
    translation = target_centroid - rotation @ source_centroid

    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = rotation
    transform[:3, 3] = translation
    return transform


def _huber_loss(residual_norms, delta):
    residual_norms = np.asarray(residual_norms, dtype=np.float64)
    quadratic = residual_norms <= delta
    return np.where(
        quadratic,
        0.5 * residual_norms**2,
        delta * (residual_norms - 0.5 * delta),
    )


def align_keypoints(
    target_ids,
    target_points,
    current_ids,
    current_points,
    min_common=6,
    huber_delta_m=0.004,
    irls_iterations=4,
):
    """Align current 3D keypoints to a reference and evaluate matching loss.

    The robust loss is evaluated before applying the fitted transform; this is
    the visual-servo objective. ``registered_rmse_m`` measures whether the two
    detections are mutually consistent after the best rigid alignment.
    """

    common_ids, target, current = match_keypoints(
        target_ids,
        target_points,
        current_ids,
        current_points,
        min_common=min_common,
    )
    if huber_delta_m <= 0:
        raise ValueError("huber_delta_m must be positive")

    weights = np.ones(len(common_ids), dtype=np.float64)
    transform = np.eye(4, dtype=np.float64)
    for _ in range(max(1, int(irls_iterations))):
        transform = _weighted_rigid_transform(current, target, weights)
        registered = (transform[:3, :3] @ current.T).T + transform[:3, 3]
        residual_norms = np.linalg.norm(target - registered, axis=1)
        weights = np.ones_like(residual_norms)
        outliers = residual_norms > huber_delta_m
        weights[outliers] = huber_delta_m / residual_norms[outliers]

    registered = (transform[:3, :3] @ current.T).T + transform[:3, 3]
    raw_residual_norms = np.linalg.norm(target - current, axis=1)
    registered_residual_norms = np.linalg.norm(target - registered, axis=1)
    centroid_error = np.mean(target, axis=0) - np.mean(current, axis=0)
    rotation_error = Rotation.from_matrix(transform[:3, :3]).magnitude()

    return KeypointAlignment(
        common_ids=common_ids,
        current_points=current,
        target_points=target,
        transform=transform,
        centroid_error_camera=centroid_error,
        raw_rmse_m=float(np.sqrt(np.mean(raw_residual_norms**2))),
        registered_rmse_m=float(np.sqrt(np.mean(registered_residual_norms**2))),
        robust_loss_m2=float(np.mean(_huber_loss(raw_residual_norms, huber_delta_m))),
        translation_error_m=float(np.linalg.norm(centroid_error)),
        rotation_error_rad=float(rotation_error),
    )


def estimate_camera_from_robot_rotation(robot_deltas, camera_deltas):
    """Estimate ``R_camera_robot`` from small Cartesian translation probes."""

    robot_deltas = np.asarray(robot_deltas, dtype=np.float64)
    camera_deltas = np.asarray(camera_deltas, dtype=np.float64)
    if robot_deltas.shape != camera_deltas.shape or robot_deltas.ndim != 2:
        raise ValueError("Probe deltas must have equal shape (N, 3)")
    if robot_deltas.shape[1] != 3 or len(robot_deltas) < 3:
        raise ValueError("At least three 3D probe deltas are required")
    if np.linalg.matrix_rank(robot_deltas) < 3:
        raise ValueError("Robot probe directions must span 3D")

    # Rows satisfy camera_delta = robot_delta @ R_camera_robot.T.
    rotation_transpose, _, _, _ = np.linalg.lstsq(robot_deltas, camera_deltas, rcond=None)
    return project_to_rotation(rotation_transpose.T)


def _clamp_vector(vector, maximum_norm):
    vector = np.asarray(vector, dtype=np.float64)
    norm = float(np.linalg.norm(vector))
    if norm <= maximum_norm or norm == 0:
        return vector
    return vector * (maximum_norm / norm)


def build_servo_target(
    current_eef,
    camera_from_robot_rotation,
    centroid_error_camera,
    rotation_error_camera,
    translation_gain=0.7,
    rotation_gain=0.7,
    max_translation_step_m=0.010,
    max_rotation_step_rad=np.deg2rad(5.0),
):
    """Build one bounded Cartesian visual-servo target in robot-base space."""

    current_eef = np.asarray(current_eef, dtype=np.float64)
    camera_from_robot_rotation = np.asarray(camera_from_robot_rotation, dtype=np.float64)
    rotation_error_camera = np.asarray(rotation_error_camera, dtype=np.float64)
    if current_eef.shape != (4, 4):
        raise ValueError("current_eef must have shape (4, 4)")
    if camera_from_robot_rotation.shape != (3, 3):
        raise ValueError("camera_from_robot_rotation must have shape (3, 3)")
    if rotation_error_camera.shape != (3, 3):
        raise ValueError("rotation_error_camera must have shape (3, 3)")
    if not 0 < translation_gain <= 1 or not 0 < rotation_gain <= 1:
        raise ValueError("Servo gains must be in (0, 1]")
    if max_translation_step_m <= 0 or max_rotation_step_rad <= 0:
        raise ValueError("Servo step limits must be positive")

    robot_from_camera_rotation = camera_from_robot_rotation.T
    translation_step_robot = robot_from_camera_rotation @ np.asarray(
        centroid_error_camera, dtype=np.float64
    )
    translation_step_robot = _clamp_vector(
        translation_step_robot * translation_gain,
        max_translation_step_m,
    )

    rotation_error_robot = (
        robot_from_camera_rotation
        @ rotation_error_camera
        @ camera_from_robot_rotation
    )
    rotation_vector_robot = Rotation.from_matrix(rotation_error_robot).as_rotvec()
    rotation_vector_robot = _clamp_vector(
        rotation_vector_robot * rotation_gain,
        max_rotation_step_rad,
    )
    rotation_step_robot = Rotation.from_rotvec(rotation_vector_robot).as_matrix()

    target = current_eef.copy()
    target[:3, 3] += translation_step_robot
    target[:3, :3] = rotation_step_robot @ current_eef[:3, :3]
    return target


def fit_multiview_pose(
    object_ids,
    object_points,
    observations,
    projection_matrices,
    initial_transform,
    f_scale_px=2.0,
    max_nfev=200,
):
    """Fit one rigid object pose by robust multi-camera reprojection loss.

    Args:
        object_ids: IDs corresponding to ``object_points``.
        object_points: Rigid marker points in marker-local coordinates.
        observations: ``serial -> (ids, Nx2 pixels)``.
        projection_matrices: ``serial -> 3x4`` current camera matrices.
        initial_transform: Initial marker-to-camera-world transform.
    """

    object_ids, object_points = _validate_keypoints(object_ids, object_points, "object")
    initial_transform = np.asarray(initial_transform, dtype=np.float64)
    if initial_transform.shape != (4, 4):
        raise ValueError("initial_transform must have shape (4, 4)")
    if f_scale_px <= 0 or max_nfev < 1:
        raise ValueError("Pose-fit controls must be positive")

    object_lookup = {int(point_id): index for index, point_id in enumerate(object_ids)}
    observation_rows = []
    camera_names = set()
    for serial, (ids, pixels) in observations.items():
        if serial not in projection_matrices:
            continue
        ids = np.asarray(ids).reshape(-1)
        pixels = np.asarray(pixels, dtype=np.float64)
        if pixels.shape != (len(ids), 2):
            raise ValueError("Observation {} has inconsistent shape".format(serial))
        projection = np.asarray(projection_matrices[serial], dtype=np.float64)
        if projection.shape != (3, 4):
            raise ValueError("Projection matrix {} must be 3x4".format(serial))
        for point_id, pixel in zip(ids, pixels):
            point_index = object_lookup.get(int(point_id))
            if point_index is None:
                continue
            observation_rows.append((point_index, pixel, projection))
            camera_names.add(serial)

    if len(camera_names) < 2:
        raise ValueError("Multiview pose fitting needs observations from at least two cameras")
    if len(observation_rows) < 6:
        raise ValueError("Multiview pose fitting needs at least six image keypoints")

    initial_parameters = np.concatenate(
        [
            Rotation.from_matrix(initial_transform[:3, :3]).as_rotvec(),
            initial_transform[:3, 3],
        ]
    )

    def residual(parameters):
        rotation = Rotation.from_rotvec(parameters[:3]).as_matrix()
        translation = parameters[3:]
        world_points = (rotation @ object_points.T).T + translation
        errors = []
        for point_index, pixel, projection in observation_rows:
            world_point = np.append(world_points[point_index], 1.0)
            projected_h = projection @ world_point
            if projected_h[2] <= 1e-8:
                errors.extend((1e4, 1e4))
                continue
            projected = projected_h[:2] / projected_h[2]
            errors.extend(projected - pixel)
        return np.asarray(errors, dtype=np.float64)

    result = least_squares(
        residual,
        initial_parameters,
        loss="soft_l1",
        f_scale=float(f_scale_px),
        max_nfev=int(max_nfev),
    )
    if not result.success:
        raise RuntimeError("Multiview marker pose fit failed: {}".format(result.message))

    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = Rotation.from_rotvec(result.x[:3]).as_matrix()
    transform[:3, 3] = result.x[3:]
    residual_values = residual(result.x).reshape(-1, 2)
    rmse = float(np.sqrt(np.mean(np.sum(residual_values**2, axis=1))))
    return MultiviewPoseFit(
        transform=transform,
        reprojection_rmse_px=rmse,
        observation_count=len(observation_rows),
        camera_count=len(camera_names),
    )
