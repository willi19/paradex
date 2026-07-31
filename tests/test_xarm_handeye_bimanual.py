import importlib.util
import sys
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]


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
    snapshotter = object()

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
        "remote_camera_controller",
        lambda *_args, **_kwargs: FakeRcc(),
    )
    monkeypatch.setattr(
        capture,
        "capture_sequence",
        lambda root, arm, ip, actual_snapshotter: calls.append(
            (Path(root).name, arm, ip, actual_snapshotter)
        ),
    )
    monkeypatch.setattr(
        capture,
        "MultiCameraDaemonReader",
        lambda _pc_list: object(),
    )
    monkeypatch.setattr(
        capture,
        "CameraSnapshotter",
        lambda _reader: snapshotter,
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
            snapshotter,
        ),
        ("wait",),
        ("Left", "xarm", "192.168.1.196", snapshotter),
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


def test_snapshotter_saves_only_fresh_daemon_frames(tmp_path):
    capture = load_script(
        "xarm_handeye_capture_snapshot_test",
        "src/calibration/handeye/xarm/capture.py",
    )

    class FakeReader:
        def __init__(self):
            self.previous = []

        def wait_for_new_frames(self, last_frame_ids, timeout):
            self.previous.append((dict(last_frame_ids), timeout))
            frame_id = len(self.previous)
            return {
                "cam-a": (
                    np.full((4, 6, 3), frame_id, dtype=np.uint8),
                    frame_id,
                )
            }

    reader = FakeReader()
    snapshotter = capture.CameraSnapshotter(reader)

    snapshotter.capture(str(tmp_path / "0"))
    snapshotter.capture(str(tmp_path / "1"))

    assert reader.previous == [
        ({}, capture.FRAME_TIMEOUT_SECONDS),
        ({"cam-a": 1}, capture.FRAME_TIMEOUT_SECONDS),
    ]
    assert (tmp_path / "0" / "images" / "cam-a.png").is_file()
    assert (tmp_path / "1" / "images" / "cam-a.png").is_file()


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
        lambda root, arm, save: calls.append((Path(root), arm, Path(save))),
    )
    monkeypatch.setattr(
        sys,
        "argv",
        ["calculate.py", "--bimanual", "--name", session.name],
    )

    calculate.main()

    assert calls == [
        (session / "Right", "xarm", session / "C2R_R.npy"),
        (session / "Left", "xarm", session / "C2R_L.npy"),
    ]


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
        lambda _root: None,
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
