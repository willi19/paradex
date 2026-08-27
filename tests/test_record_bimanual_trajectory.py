import importlib.util
from pathlib import Path

import numpy as np
import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]


def load_module(name):
    spec = importlib.util.spec_from_file_location(
        name,
        REPO_ROOT / "src/calibration/allinone/record_bimanual_trajectory.py",
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_cartesian_to_homo_converts_xarm_mm_to_meters():
    script = load_module("record_bimanual_cartesian_test")

    pose = script.cartesian_to_homo(np.array([100.0, 200.0, 300.0, 0.0, 0.0, 0.0]))

    np.testing.assert_allclose(pose[:3, 3], [0.1, 0.2, 0.3])
    np.testing.assert_allclose(pose[:3, :3], np.eye(3))


def test_target_pose_with_y_offset_uses_fk_and_not_ik():
    script = load_module("record_bimanual_target_test")

    class FakeArm:
        def get_forward_kinematics(self, qpos, input_is_radian, return_is_radian):
            assert qpos == [1, 2, 3, 4, 5, 6]
            assert input_is_radian is True
            assert return_is_radian is True
            return 0, [10.0, 20.0, 30.0, 0.0, 0.0, 0.0]

        def get_inverse_kinematics(self, _cartesian):
            raise AssertionError("recording utility must not call explicit IK")

    class FakeController:
        arm = FakeArm()

    pose = script.target_pose_with_y_offset(
        FakeController(),
        np.array([1, 2, 3, 4, 5, 6], dtype=np.float64),
        0.32,
    )

    np.testing.assert_allclose(pose[:3, 3], [0.01, 0.34, 0.03])


def test_selected_pose_files_resume_skips_existing_output(tmp_path):
    script = load_module("record_bimanual_selection_test")
    source_dir = tmp_path / "src"
    output_dir = tmp_path / "out"
    source_dir.mkdir()
    (output_dir / "Right").mkdir(parents=True)
    for index in range(4):
        np.save(source_dir / f"{index}_qpos.npy", np.zeros(6))
    np.save(output_dir / "Right" / "1_qpos.npy", np.ones(6))

    selected = script.selected_pose_files(
        source_dir,
        output_dir,
        "Right",
        start=1,
        end=4,
        resume=True,
    )

    assert [path.name for path in selected] == ["2_qpos.npy", "3_qpos.npy"]


def test_record_side_moves_cartesian_and_saves_actual_qpos(tmp_path):
    script = load_module("record_bimanual_record_side_test")
    source_dir = tmp_path / "src"
    output_dir = tmp_path / "out"
    source_dir.mkdir()
    pose_path = source_dir / "0_qpos.npy"
    np.save(pose_path, np.arange(6, dtype=np.float64))
    calls = []

    class FakeArm:
        def get_forward_kinematics(self, qpos, input_is_radian, return_is_radian):
            calls.append(("fk", qpos, input_is_radian, return_is_radian))
            return 0, [100.0, 200.0, 300.0, 0.0, 0.0, 0.0]

        def get_inverse_kinematics(self, _cartesian):
            raise AssertionError("recording utility must not call explicit IK")

    class FakeController:
        arm = FakeArm()
        target = None

        def move(self, action, is_servo):
            self.target = np.asarray(action).copy()
            calls.append(("move", np.asarray(action).copy(), is_servo))

        def get_data(self):
            calls.append(("get_data",))
            return {
                "qpos": np.ones(6, dtype=np.float64),
                "position": self.target,
                "time": 123.0,
            }

        def end(self, set_break):
            calls.append(("end", set_break))

    side = script.SideConfig("Right", "192.0.2.10", 0.32)

    count = script.record_side(
        side,
        source_dir,
        output_dir,
        [pose_path],
        settle_seconds=0.0,
        max_translation_error_m=0.005,
        max_rotation_error_rad=np.deg2rad(3.0),
        overwrite=False,
        controller_factory=lambda ip: FakeController(),
        prompt_fn=lambda _prompt: "MOVE RIGHT",
        sleep_fn=lambda _seconds: None,
    )

    assert count == 1
    assert calls[0] == ("fk", [0, 1, 2, 3, 4, 5], True, True)
    assert calls[1][0] == "move"
    np.testing.assert_allclose(calls[1][1][:3, 3], [0.1, 0.52, 0.3])
    assert calls[1][2] is False
    assert calls[-1] == ("end", False)
    np.testing.assert_allclose(np.load(output_dir / "Right" / "0_qpos.npy"), np.ones(6))
    np.testing.assert_allclose(
        np.load(output_dir / "Right" / "0_eef.npy"),
        calls[1][1],
    )


def test_record_side_rejects_wrong_readiness_phrase_before_controller(tmp_path):
    script = load_module("record_bimanual_readiness_test")
    source_dir = tmp_path / "src"
    source_dir.mkdir()
    pose_path = source_dir / "0_qpos.npy"
    np.save(pose_path, np.zeros(6))

    def fail_controller_factory(_ip):
        raise AssertionError("controller must not be constructed before readiness")

    with pytest.raises(RuntimeError, match="readiness phrase"):
        script.record_side(
            script.SideConfig("Left", "192.0.2.221", -0.32),
            source_dir,
            tmp_path / "out",
            [pose_path],
            settle_seconds=0.0,
            max_translation_error_m=0.005,
            max_rotation_error_rad=np.deg2rad(3.0),
            overwrite=False,
            controller_factory=fail_controller_factory,
            prompt_fn=lambda _prompt: "Y",
            sleep_fn=lambda _seconds: None,
        )


def test_readiness_phrase_accepts_underscore_alias():
    script = load_module("record_bimanual_readiness_alias_test")

    script.require_ready("Right", lambda _prompt: "MOVE_RIGHT")
    script.require_ready("Left", lambda _prompt: "move_left")


def test_main_without_execute_is_dry_run(monkeypatch, tmp_path, capsys):
    script = load_module("record_bimanual_dry_run_test")
    source_dir = tmp_path / "src"
    source_dir.mkdir()
    np.save(source_dir / "0_qpos.npy", np.zeros(6))

    monkeypatch.setattr(
        script,
        "record_side",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("dry run must not record")
        ),
    )

    assert script.main(["--source-dir", str(source_dir), "--output-dir", str(tmp_path / "out")]) == 0

    assert "Dry run only" in capsys.readouterr().out
