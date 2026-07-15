import numpy as np
import pytest

from src.validate.teleop.quest3_xarm import QuestXArmTeleopController


def _data(position):
    wrist = np.eye(4)
    wrist[:3, 3] = np.asarray(position, dtype=float)
    return {"Right": {"wrist": wrist}, "Left": None}


def test_pause_resume_rebases_without_command_jump():
    controller = QuestXArmTeleopController(
        np.eye(4), max_offset_m=0.0, max_step_m=0.0
    )

    target, _ = controller.update(_data([0.2, 0.3, 0.4]), control_state=0)
    assert target == pytest.approx(np.eye(4))

    target, event = controller.update(_data([0.5, -0.2, 0.8]), control_state=1)
    assert target is None
    assert event == "paused"

    target, event = controller.update(_data([-0.4, 0.7, 0.2]), control_state=0)
    assert event == "resumed"
    assert target == pytest.approx(np.eye(4))

    target, _ = controller.update(_data([-0.35, 0.68, 0.23]), control_state=0)
    assert target[:3, 3] == pytest.approx([0.05, -0.02, 0.03])


def test_tracking_reacquisition_rebases_without_command_jump():
    controller = QuestXArmTeleopController(np.eye(4))
    controller.update(_data([0.1, 0.2, 0.3]), control_state=0)

    target, event = controller.update({"Right": None}, control_state=0)
    assert target is None
    assert event == "tracking_lost"

    target, event = controller.update(_data([1.0, -1.0, 0.5]), control_state=0)
    assert event == "tracking_acquired"
    assert target == pytest.approx(np.eye(4))


def test_translation_limits_apply_to_offset_and_single_step():
    controller = QuestXArmTeleopController(
        np.eye(4), max_offset_m=0.20, max_step_m=0.05
    )
    controller.update(_data([0.0, 0.0, 0.0]), control_state=0)

    target, _ = controller.update(_data([1.0, 0.0, 0.0]), control_state=0)
    assert np.linalg.norm(target[:3, 3]) == pytest.approx(0.05)

    for _ in range(10):
        target, _ = controller.update(_data([1.0, 0.0, 0.0]), control_state=0)
    assert np.linalg.norm(target[:3, 3]) == pytest.approx(0.20)
