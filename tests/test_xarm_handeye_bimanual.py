import importlib.util
import sys
import threading
from pathlib import Path

import numpy as np
import torch

from paradex.calibration.Tsai_Lenz import _handeye_device, handeye_residual_loss
from paradex.image.image_dict import ImageDict


REPO_ROOT = Path(__file__).resolve().parents[1]


def test_batched_handeye_loss_matches_original_pose_loop():
    rng = np.random.default_rng(7)
    A = torch.as_tensor(rng.normal(size=(8, 4, 4)), dtype=torch.float64)
    B = torch.as_tensor(rng.normal(size=(8, 4, 4)), dtype=torch.float64)
    X = torch.as_tensor(rng.normal(size=(4, 4)), dtype=torch.float64)

    expected = torch.zeros((), dtype=torch.float64)
    for pose_a, pose_b in zip(A, B):
        residual = pose_a @ X - X @ pose_b
        expected += torch.norm(residual, "fro") ** 2
        expected += torch.norm(residual[:3, 3])

    torch.testing.assert_close(handeye_residual_loss(A, B, X), expected)


def test_handeye_prefers_gpu_zero_when_cuda_is_available(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)

    assert _handeye_device() == torch.device("cuda:0")


def load_script(name, relative_path):
    spec = importlib.util.spec_from_file_location(name, REPO_ROOT / relative_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_capture_bimanual_uses_right_then_left(monkeypatch, tmp_path):
    capture = load_script(
        "xarm_handeye_capture_test",
        "src/calibration/handeye/xarm/capture.py",
    )
    calls = []
    rcc = None

    class FakeNow:
        @staticmethod
        def strftime(_format):
            return "20260731_120000"

    class FakeDatetime:
        @staticmethod
        def now():
            return FakeNow()

    class FakeRcc:
        def start(self, mode, sync_mode, fps):
            calls.append(("rcc_start", mode, sync_mode, fps))

        def stop(self):
            calls.append(("rcc_stop",))

        def end(self):
            calls.append(("rcc_end",))

    monkeypatch.setattr(capture, "datetime", FakeDatetime)
    monkeypatch.setattr(capture, "handeye_calib_bimanual_path", str(tmp_path))
    monkeypatch.setattr(
        capture,
        "BIMANUAL_POSE_DIR",
        str(tmp_path / "handeye_pose_matching/20260803_125942"),
    )
    rcc = FakeRcc()
    monkeypatch.setattr(capture, "remote_camera_controller", lambda *_args, **_kwargs: rcc)
    monkeypatch.setattr(
        capture,
        "capture_sequence",
        lambda root, arm, ip, actual_rcc, trajectory_dir=None: calls.append(
            (Path(root).name, arm, ip, actual_rcc, Path(trajectory_dir))
        ),
    )
    monkeypatch.setattr(
        capture,
        "wait_for_left_arm",
        lambda: calls.append(("wait",)),
    )
    monkeypatch.setattr(sys, "argv", ["capture.py", "--bimanual"])

    capture.main()

    assert calls == [
        ("rcc_start", "stream", False, 30),
        (
            "Right",
            "xarm",
            capture.network_info["xarm"]["param"]["ip"],
            rcc,
            tmp_path / "handeye_pose_matching/20260803_125942/Right",
        ),
        ("rcc_stop",),
        ("wait",),
        ("rcc_start", "stream", False, 30),
        (
            "Left",
            "xarm",
            capture.LEFT_XARM_IP,
            rcc,
            tmp_path / "handeye_pose_matching/20260803_125942/Left",
        ),
        ("rcc_stop",),
        ("rcc_end",),
    ]
    session = tmp_path / "20260731_120000"
    assert (session / "Right").is_dir()
    assert (session / "Left").is_dir()


def test_wait_for_left_arm_requires_y(monkeypatch):
    capture = load_script(
        "xarm_handeye_capture_wait_test",
        "src/calibration/handeye/xarm/capture.py",
    )
    answers = iter(["", "n", "Y"])
    monkeypatch.setattr("builtins.input", lambda _prompt: next(answers))

    capture.wait_for_left_arm()


def test_capture_single_arm_keeps_default_trajectory(monkeypatch, tmp_path):
    capture = load_script(
        "xarm_handeye_capture_single_test",
        "src/calibration/handeye/xarm/capture.py",
    )
    trajectory_args = []

    class FakeNow:
        @staticmethod
        def strftime(_format):
            return "20260803_140000"

    class FakeDatetime:
        @staticmethod
        def now():
            return FakeNow()

    class FakeRcc:
        def start(self, _mode, _sync_mode, fps):
            assert fps == 30

        def stop(self):
            pass

        def end(self):
            pass

    monkeypatch.setattr(capture, "datetime", FakeDatetime)
    monkeypatch.setattr(capture, "handeye_calib_path", str(tmp_path))
    monkeypatch.setattr(
        capture,
        "remote_camera_controller",
        lambda *_args, **_kwargs: FakeRcc(),
    )
    monkeypatch.setattr(
        capture,
        "capture_sequence",
        lambda _root, _arm, _ip, _rcc, trajectory_dir=None: trajectory_args.append(
            trajectory_dir
        ),
    )
    monkeypatch.setattr(sys, "argv", ["capture.py"])

    capture.main()

    assert trajectory_args == [None]


def test_capture_sequence_sends_remote_snapshot_path(monkeypatch, tmp_path):
    capture = load_script(
        "xarm_handeye_capture_snapshot_test",
        "src/calibration/handeye/xarm/capture.py",
    )
    trajectory_dir = tmp_path / "trajectory"
    trajectory_dir.mkdir()
    np.save(trajectory_dir / "0_qpos.npy", np.zeros(6))
    calls = []

    class FakeController:
        def move(self, action, is_servo):
            calls.append(("move", action.shape, is_servo))

        def get_data(self):
            return {
                "position": np.eye(4),
                "qpos": np.zeros(6),
            }

        def end(self, set_break):
            calls.append(("end", set_break))

    class FakeRcc:
        def snapshot(self, save_path):
            calls.append(("snapshot", save_path))
            (tmp_path / "capture" / "0").mkdir(parents=True, exist_ok=True)

    monkeypatch.setattr(capture, "create_controller", lambda *_args: FakeController())
    monkeypatch.setattr(capture, "get_handeye_calib_traj", lambda _arm: str(trajectory_dir))
    monkeypatch.setattr(capture, "save_current_camparam", lambda _path: None)
    monkeypatch.setattr(capture.time, "sleep", lambda _seconds: None)
    monkeypatch.setattr(capture, "remove_home", lambda path: f"remote/{Path(path).name}")

    capture.capture_sequence(
        str(tmp_path / "capture"),
        "xarm",
        "192.0.2.1",
        FakeRcc(),
    )

    assert calls == [
        ("move", (6,), False),
        ("snapshot", "remote/0"),
        ("end", False),
    ]


def test_calculate_bimanual_saves_named_transforms(monkeypatch, tmp_path):
    calculate = load_script(
        "xarm_handeye_calculate_test",
        "src/calibration/handeye/xarm/calculate.py",
    )
    session = tmp_path / "20260731_120000"
    (session / "Right").mkdir(parents=True)
    (session / "Left").mkdir()
    calls = []

    monkeypatch.setattr(calculate, "handeye_calib_bimanual_path", str(tmp_path))
    monkeypatch.setattr(
        calculate,
        "validate_capture_directory",
        lambda _root: None,
    )
    monkeypatch.setattr(
        calculate,
        "calculate_sequence",
        lambda root, arm, save, workers: calls.append(
            (Path(root), arm, Path(save), workers)
        ),
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "calculate.py",
            "--bimanual",
            "--name",
            session.name,
            "--charuco-workers",
            "3",
        ],
    )

    calculate.main()

    assert calls == [
        (session / "Right", "xarm", session / "C2R_R.npy", 3),
        (session / "Left", "xarm_left", session / "C2R_L.npy", 3),
    ]


def test_compute_fk_replaces_cached_pose_with_selected_urdf(monkeypatch, tmp_path):
    calculate = load_script(
        "xarm_handeye_calculate_fk_urdf_test",
        "src/calibration/handeye/xarm/calculate.py",
    )
    pose_dir = tmp_path / "0"
    pose_dir.mkdir()
    np.save(pose_dir / "qpos.npy", np.zeros(6))
    np.save(pose_dir / "eef_fk.npy", np.eye(4))

    expected_pose = np.eye(4)
    expected_pose[0, 3] = 0.123
    selected_paths = []

    class FakeRobotWrapper:
        def __init__(self, urdf_path):
            selected_paths.append(urdf_path)

        def compute_forward_kinematics(self, qpos, link_list):
            np.testing.assert_array_equal(qpos, np.zeros(6))
            assert link_list == ["link6"]
            return {"link6": expected_pose}

    monkeypatch.setattr(calculate, "RobotWrapper", FakeRobotWrapper)
    monkeypatch.setattr(
        calculate,
        "get_robot_urdf_path",
        lambda arm_name: f"/robots/{arm_name}_new.urdf",
    )

    calculate.compute_fk(str(tmp_path), "xarm_left")

    assert selected_paths == ["/robots/xarm_left_new.urdf"]
    np.testing.assert_array_equal(np.load(pose_dir / "eef_fk.npy"), expected_pose)


def test_calculate_sequence_keeps_single_arm_output_compatible(
    monkeypatch,
    tmp_path,
):
    calculate = load_script(
        "xarm_handeye_calculate_sequence_test",
        "src/calibration/handeye/xarm/calculate.py",
    )
    transform = np.eye(4)
    (tmp_path / "0").mkdir()

    monkeypatch.setattr(
        calculate,
        "undistort_and_detect_charuco",
        lambda _root, _workers: None,
    )
    monkeypatch.setattr(calculate, "compute_fk", lambda _root, _arm: None)
    monkeypatch.setattr(
        calculate,
        "compute_motion",
        lambda _root: ([np.eye(4)], [np.eye(4)]),
    )
    monkeypatch.setattr(calculate, "solve_ax_xb", lambda *_args, **_kwargs: transform)
    monkeypatch.setattr(calculate, "get_valid_indices", lambda _root: ["0", "1"])
    monkeypatch.setattr(calculate, "debug", lambda *_args: None)
    monkeypatch.setattr(
        calculate,
        "validate_capture_directory",
        lambda _root: None,
    )

    calculate.calculate_sequence(str(tmp_path), "xarm", None)

    np.testing.assert_array_equal(np.load(tmp_path / "0" / "C2R.npy"), transform)


def test_calculate_sequence_uses_precomputed_charuco(monkeypatch, tmp_path):
    calculate = load_script(
        "xarm_handeye_calculate_precomputed_test",
        "src/calibration/handeye/xarm/calculate.py",
    )
    transform = np.eye(4)
    (tmp_path / "0").mkdir()

    def fail_if_detected(*_args, **_kwargs):
        raise AssertionError("ChArUco detection must not run for precomputed data")

    monkeypatch.setattr(calculate, "undistort_and_detect_charuco", fail_if_detected)
    monkeypatch.setattr(calculate, "compute_fk", lambda _root, _arm: None)
    monkeypatch.setattr(
        calculate,
        "compute_motion",
        lambda _root: ([np.eye(4)], [np.eye(4)]),
    )
    monkeypatch.setattr(calculate, "solve_ax_xb", lambda *_args, **_kwargs: transform)
    monkeypatch.setattr(calculate, "get_valid_indices", lambda _root: ["0", "1"])
    monkeypatch.setattr(calculate, "debug", lambda *_args: None)
    monkeypatch.setattr(calculate, "validate_capture_directory", lambda _root: None)

    calculate.calculate_sequence(
        str(tmp_path),
        "xarm",
        None,
        precomputed_charuco=True,
    )

    np.testing.assert_array_equal(np.load(tmp_path / "0" / "C2R.npy"), transform)


def test_charuco_detection_runs_camera_images_in_parallel(monkeypatch):
    calculate = load_script(
        "xarm_handeye_calculate_parallel_test",
        "src/calibration/handeye/xarm/calculate.py",
    )
    barrier = threading.Barrier(2)
    thread_ids = set()

    def fake_detect(image):
        thread_ids.add(threading.get_ident())
        barrier.wait(timeout=1.0)
        return {"value": int(image[0])}

    class FakeImageDict:
        images = {
            "cam-a": np.array([1]),
            "cam-b": np.array([2]),
        }

    monkeypatch.setattr(calculate, "detect_charuco", fake_detect)

    with calculate.ThreadPoolExecutor(max_workers=2) as executor:
        result = calculate._detect_charuco_images(FakeImageDict(), executor)

    assert result == {
        "cam-a": {"value": 1},
        "cam-b": {"value": 2},
    }
    assert len(thread_ids) == 2


def test_robot_debug_image_postprocessing_runs_in_parallel(monkeypatch, tmp_path):
    calculate = load_script(
        "xarm_handeye_debug_parallel_test",
        "src/calibration/handeye/xarm/calculate.py",
    )
    raw_image_dir = tmp_path / "0" / "images"
    image_dir = tmp_path / "0" / "undistort" / "images"
    raw_image_dir.mkdir(parents=True)
    image_dir.mkdir(parents=True)
    for serial in ("cam-a", "cam-b"):
        (raw_image_dir / f"{serial}.png").touch()
        (image_dir / f"{serial}.png").touch()
    np.save(tmp_path / "0" / "qpos.npy", np.zeros(6))
    np.save(tmp_path / "0" / "eef_fk.npy", np.eye(4))
    np.save(
        tmp_path / "0" / "charuco_3d_corners.npy",
        np.array([[0.0, 0.0, 1.0]]),
    )

    class FakeMesh:
        def apply_transform(self, _transform):
            pass

    class FakeRobotModule:
        def update_cfg(self, _qpos):
            pass

        def get_robot_mesh(self):
            return FakeMesh()

    class FakeRenderer:
        def __init__(self, _intrinsic, _extrinsic):
            pass

        def render_mask(self, _mesh):
            mask = np.zeros((2, 2), dtype=bool)
            return {"cam-a": mask, "cam-b": mask}

    import paradex.image.projection as projection

    monkeypatch.setattr(projection, "BatchRenderer", FakeRenderer)
    barrier = threading.Barrier(2)
    thread_ids = set()

    def fake_save(*_args):
        thread_ids.add(threading.get_ident())
        barrier.wait(timeout=1.0)

    monkeypatch.setattr(calculate, "save_robot_debug_image", fake_save)
    camera_matrix = np.eye(3)
    camera_matrix[2, 2] = 1.0
    intrinsic = {
        serial: {"intrinsics_undistort": camera_matrix}
        for serial in ("cam-a", "cam-b")
    }
    extrinsic = {
        serial: np.column_stack((np.eye(3), np.zeros(3)))
        for serial in ("cam-a", "cam-b")
    }

    calculate.render_robot_debug_parallel(
        str(tmp_path),
        ["0"],
        FakeRobotModule(),
        intrinsic,
        extrinsic,
        np.eye(4),
        np.array([[0.0, 0.0, 1.0]]),
        debug_workers=2,
    )

    assert len(thread_ids) == 2


def test_robot_debug_skips_camera_missing_from_later_step(monkeypatch, tmp_path):
    calculate = load_script(
        "xarm_handeye_debug_missing_camera_test",
        "src/calibration/handeye/xarm/calculate.py",
    )
    for index in ("0", "1"):
        raw_dir = tmp_path / index / "images"
        undistort_dir = tmp_path / index / "undistort" / "images"
        raw_dir.mkdir(parents=True)
        undistort_dir.mkdir(parents=True)
        np.save(tmp_path / index / "qpos.npy", np.zeros(6))
        np.save(tmp_path / index / "eef_fk.npy", np.eye(4))
        np.save(
            tmp_path / index / "charuco_3d_corners.npy",
            np.array([[0.0, 0.0, 1.0]]),
        )
    for index, serial in (("0", "cam-a"), ("0", "cam-b"), ("1", "cam-a")):
        (tmp_path / index / "images" / f"{serial}.png").touch()
        (tmp_path / index / "undistort" / "images" / f"{serial}.png").touch()
    # A stale undistorted image without a corresponding raw source must be ignored.
    (tmp_path / "1" / "undistort" / "images" / "cam-b.png").touch()

    class FakeMesh:
        def apply_transform(self, _transform):
            pass

    class FakeRobotModule:
        def update_cfg(self, _qpos):
            pass

        def get_robot_mesh(self):
            return FakeMesh()

    class FakeRenderer:
        def __init__(self, _intrinsic, _extrinsic):
            pass

        def render_mask(self, _mesh):
            mask = np.zeros((2, 2), dtype=bool)
            return {"cam-a": mask, "cam-b": mask}

    import paradex.image.projection as projection

    monkeypatch.setattr(projection, "BatchRenderer", FakeRenderer)
    saved_paths = []
    monkeypatch.setattr(
        calculate,
        "save_robot_debug_image",
        lambda _image, output, *_args: saved_paths.append(Path(output)),
    )
    camera_matrix = np.eye(3)
    intrinsic = {
        serial: {"intrinsics_undistort": camera_matrix}
        for serial in ("cam-a", "cam-b")
    }
    extrinsic = {
        serial: np.column_stack((np.eye(3), np.zeros(3)))
        for serial in ("cam-a", "cam-b")
    }

    calculate.render_robot_debug_parallel(
        str(tmp_path),
        ["0", "1"],
        FakeRobotModule(),
        intrinsic,
        extrinsic,
        np.eye(4),
        np.array([[0.0, 0.0, 1.0]]),
        debug_workers=2,
    )

    relative_paths = {
        path.relative_to(tmp_path).as_posix() for path in saved_paths
    }
    assert relative_paths == {
        "0/debug/images/cam-a.png",
        "0/debug/images/cam-b.png",
        "1/debug/images/cam-a.png",
    }


def test_charuco_triangulation_reuses_precomputed_detections(monkeypatch):
    image_dict = ImageDict({"cam-a": np.zeros((2, 2, 3), dtype=np.uint8)})
    image_dict._cache["proj_mtx"] = {}
    monkeypatch.setattr(
        "paradex.image.image_dict.detect_charuco",
        lambda _image: (_ for _ in ()).throw(
            AssertionError("Charuco must not be detected twice")
        ),
    )

    result = image_dict.triangulate_charuco(
        detections={"cam-a": {}},
    )

    assert result == {}


def test_incomplete_capture_fails_before_calculation(tmp_path):
    calculate = load_script(
        "xarm_handeye_calculate_validation_test",
        "src/calibration/handeye/xarm/calculate.py",
    )
    (tmp_path / "0").mkdir()

    try:
        calculate.validate_capture_directory(str(tmp_path))
    except ValueError as error:
        assert "missing camera parameters" in str(error)
    else:
        raise AssertionError("Incomplete capture should not pass validation")
