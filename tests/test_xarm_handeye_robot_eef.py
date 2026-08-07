import importlib.util
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]


def load_calculator():
    script_path = (
        REPO_ROOT
        / "src/calibration/handeye/xarm/calculate_robot_eef.py"
    )
    spec = importlib.util.spec_from_file_location(
        "xarm_handeye_robot_eef_test",
        script_path,
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_compute_motion_uses_controller_eef_poses(tmp_path):
    calculator = load_calculator()
    corners = np.array(
        [[0.0, 0.0, 0.0], [0.1, 0.0, 0.0], [0.0, 0.1, 0.0]]
    )
    for index in ("0", "1"):
        (tmp_path / index).mkdir()
        np.save(tmp_path / index / "charuco_3d_ids.npy", np.arange(3))
        np.save(tmp_path / index / "charuco_3d_corners.npy", corners)

    eef_0 = np.eye(4)
    eef_1 = np.eye(4)
    eef_1[0, 3] = 0.2
    np.save(tmp_path / "0" / "eef.npy", eef_0)
    np.save(tmp_path / "1" / "eef.npy", eef_1)

    _, robot_motion, indices = calculator.compute_motion(str(tmp_path))

    assert indices == ["0", "1"]
    np.testing.assert_allclose(robot_motion[0][0, 3], -0.2)


def test_rotation_error_degrees_reports_actual_residual():
    calculator = load_calculator()
    left = np.eye(4)
    right = np.eye(4)
    right[:3, :3] = np.array(
        [
            [0.0, -1.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0],
        ]
    )

    np.testing.assert_allclose(calculator.rotation_error_degrees(left, right), 90.0)
