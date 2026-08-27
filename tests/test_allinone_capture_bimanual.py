import importlib.util
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


def load_capture_module(name):
    spec = importlib.util.spec_from_file_location(
        name,
        REPO_ROOT / "src/calibration/allinone/capture.py",
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_bimanual_main_replays_measured_right_then_left_trajectories(
    monkeypatch,
    tmp_path,
    capsys,
):
    capture = load_capture_module("allinone_capture_bimanual_main_test")
    calls = []

    class FakeNow:
        @staticmethod
        def strftime(_format):
            return "20260826_120000"

    class FakeDatetime:
        @staticmethod
        def now():
            return FakeNow()

    class FakeCameraController:
        def start(self, mode, sync_mode, fps):
            calls.append(("start", mode, sync_mode, fps))

        def stop(self):
            calls.append(("stop",))

        def end(self):
            calls.append(("end",))

    camera_controller = FakeCameraController()
    monkeypatch.setattr(capture, "datetime", FakeDatetime)
    monkeypatch.setattr(capture, "extrinsic_dir", str(tmp_path))
    monkeypatch.setattr(
        capture,
        "remote_camera_controller",
        lambda *_args, **_kwargs: camera_controller,
    )
    monkeypatch.setattr(
        capture,
        "get_handeye_calib_traj",
        lambda _arm: str(tmp_path / "trajectory"),
    )
    monkeypatch.setattr(
        capture,
        "capture_sequence",
        lambda root, arm, ip, actual_camera, trajectory_dir: calls.append(
            (
                Path(root).name,
                arm,
                ip,
                actual_camera,
                Path(trajectory_dir),
            )
        ),
    )
    monkeypatch.setattr(
        capture,
        "wait_for_left_arm",
        lambda: calls.append(("wait",)),
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "capture.py",
            "--bimanual",
            "--left-ip",
            "192.0.2.221",
            "--bimanual-trajectory-dir",
            str(tmp_path / "measured"),
        ],
    )

    capture.main()

    assert calls == [
        ("start", "stream", False, 30),
        (
            "Right",
            "xarm",
            None,
            camera_controller,
            tmp_path / "measured" / "Right",
        ),
        ("stop",),
        ("wait",),
        ("start", "stream", False, 30),
        (
            "Left",
            "xarm",
            "192.0.2.221",
            camera_controller,
            tmp_path / "measured" / "Left",
        ),
        ("stop",),
        ("end",),
    ]
    assert (tmp_path / "20260826_120000" / "Right").is_dir()
    assert (tmp_path / "20260826_120000" / "Left").is_dir()
    assert (
        "python src/calibration/allinone/calculate.py "
        "--name 20260826_120000 --arm xarm --bimanual"
    ) in capsys.readouterr().out


def test_single_arm_main_preserves_default_capture_call(monkeypatch, tmp_path):
    capture = load_capture_module("allinone_capture_single_main_test")
    calls = []

    class FakeNow:
        @staticmethod
        def strftime(_format):
            return "20260826_130000"

    class FakeDatetime:
        @staticmethod
        def now():
            return FakeNow()

    class FakeCameraController:
        def start(self, _mode, _sync_mode, fps):
            assert fps == 30

        def stop(self):
            pass

        def end(self):
            pass

    monkeypatch.setattr(capture, "datetime", FakeDatetime)
    monkeypatch.setattr(capture, "extrinsic_dir", str(tmp_path))
    monkeypatch.setattr(
        capture,
        "remote_camera_controller",
        lambda *_args, **_kwargs: FakeCameraController(),
    )
    monkeypatch.setattr(
        capture,
        "get_handeye_calib_traj",
        lambda _arm: str(tmp_path / "trajectory"),
    )
    monkeypatch.setattr(
        capture,
        "capture_sequence",
        lambda *args, **kwargs: calls.append((args, kwargs)),
    )
    monkeypatch.setattr(sys, "argv", ["capture.py"])

    capture.main()

    assert len(calls) == 1
    args, kwargs = calls[0]
    assert args[0] == str(tmp_path / "20260826_130000")
    assert args[1:4] == ("xarm", None, args[3])
    assert args[4] == str(tmp_path / "trajectory")
    assert kwargs == {}
