import numpy as np

from paradex.calibration.extrinsic_validation import (
    aggregate_reprojection_errors,
    calculate_reprojection_errors,
    summarize_reprojection_errors,
)
from src.calibration.extrinsic import validate as validate_module


def _camera_parameters(camera_centers):
    intrinsics = {}
    extrinsics = {}
    camera_matrix = np.array(
        [
            [1000.0, 0.0, 640.0],
            [0.0, 1000.0, 480.0],
            [0.0, 0.0, 1.0],
        ]
    )
    for index, center_x in enumerate(camera_centers):
        serial = f"cam{index}"
        intrinsics[serial] = {
            "original_intrinsics": camera_matrix,
            "intrinsics_undistort": camera_matrix,
            "dist_params": np.zeros(5),
        }
        extrinsics[serial] = np.column_stack(
            (np.eye(3), np.array([-center_x, 0.0, 0.0]))
        )
    return intrinsics, extrinsics


def _observations(points, intrinsics, actual_extrinsics):
    observations = {}
    homogeneous_points = np.column_stack((points, np.ones(len(points))))
    for serial, extrinsic in actual_extrinsics.items():
        projection = intrinsics[serial]["intrinsics_undistort"] @ extrinsic
        projected = (projection @ homogeneous_points.T).T
        observations[serial] = {
            "ids": np.arange(len(points)),
            "corners": projected[:, :2] / projected[:, 2:3],
        }
    return observations


def test_leave_one_out_reprojection_is_zero_for_consistent_calibration():
    intrinsics, extrinsics = _camera_parameters([-1.0, -0.5, 0.0, 0.5, 1.0])
    points = np.array(
        [
            [-0.4, -0.2, 4.0],
            [0.2, -0.1, 4.5],
            [-0.1, 0.3, 5.0],
            [0.5, 0.2, 5.5],
        ]
    )

    errors = calculate_reprojection_errors(
        _observations(points, intrinsics, extrinsics),
        intrinsics,
        extrinsics,
    )

    assert all(len(values) == len(points) for values in errors.values())
    assert max(np.max(values) for values in errors.values()) < 1e-8


def test_moved_camera_has_largest_reprojection_error():
    centers = [-1.0, -0.5, 0.0, 0.5, 1.0]
    intrinsics, calibrated_extrinsics = _camera_parameters(centers)
    _, actual_extrinsics = _camera_parameters(centers[:-1] + [1.2])
    points = np.array(
        [
            [-0.5, -0.3, 4.0],
            [0.2, -0.2, 4.5],
            [-0.2, 0.3, 5.0],
            [0.4, 0.1, 5.5],
            [0.0, 0.0, 6.0],
        ]
    )

    errors = calculate_reprojection_errors(
        _observations(points, intrinsics, actual_extrinsics),
        intrinsics,
        calibrated_extrinsics,
    )
    medians = {serial: np.median(values) for serial, values in errors.items()}

    assert max(medians, key=medians.get) == "cam4"
    assert medians["cam4"] > 20.0
    assert medians["cam4"] > 3.0 * max(
        value for serial, value in medians.items() if serial != "cam4"
    )


def test_aggregate_and_summary_ignore_empty_camera_results():
    aggregate = aggregate_reprojection_errors(
        [
            {"cam0": np.array([1.0, 2.0]), "cam1": np.array([])},
            {"cam0": np.array([3.0]), "cam1": np.array([4.0])},
        ]
    )
    summary = summarize_reprojection_errors(aggregate)

    np.testing.assert_array_equal(aggregate["cam0"], [1.0, 2.0, 3.0])
    np.testing.assert_array_equal(aggregate["cam1"], [4.0])
    assert summary["cam0"]["count"] == 3
    assert summary["cam0"]["median"] == 2.0


def test_temporary_capture_uses_existing_saved_observation_format(
    tmp_path,
    monkeypatch,
):
    monkeypatch.setattr(validate_module, "extrinsic_dir", str(tmp_path))
    validation_root = validate_module._create_validation_root()
    capture_path = validate_module._create_capture_directory(validation_root)
    markers_path = capture_path / "markers_2d"
    np.save(markers_path / "cam0_corner.npy", np.array([[10.0, 20.0]]))
    np.save(markers_path / "cam0_id.npy", np.array([3]))

    intrinsics, extrinsics = _camera_parameters([0.0])
    observations = validate_module._load_saved_observations(
        capture_path,
        intrinsics,
        extrinsics,
    )

    np.testing.assert_array_equal(observations["cam0"]["corners"], [[10.0, 20.0]])
    np.testing.assert_array_equal(observations["cam0"]["ids"], [3])
    assert validation_root.stat().st_mode & 0o777 == 0o777

    validate_module._remove_temp_path(validation_root, validation_root)
    assert not validation_root.exists()
