import importlib

import numpy as np
from scipy.spatial.transform import Rotation


quest_teleop = importlib.import_module("paradex.retargetor.2pose.quest_teleop")


def test_quest_delta_rebases_without_an_enable_jump():
    controller = np.eye(4)
    xarm = np.eye(4)
    xarm[:3, 3] = [0.4, -0.2, 0.3]

    target = quest_teleop.quest_delta_to_xarm_target(
        controller,
        controller,
        xarm,
    )

    np.testing.assert_allclose(target, xarm)


def test_quest_translation_uses_configured_controller_to_xarm_axes():
    initial_controller = np.eye(4)
    controller = np.eye(4)
    controller[:3, 3] = [0.1, 0.2, -0.3]
    xarm = np.eye(4)

    target = quest_teleop.quest_delta_to_xarm_target(
        initial_controller,
        controller,
        xarm,
        translation_scale=2.0,
    )

    expected = 2.0 * (
        quest_teleop.CONTROLLER_TO_XARM_AXIS @ np.array([0.1, 0.2, -0.3])
    )
    np.testing.assert_allclose(target[:3, 3], expected)


def test_quest_rotation_delta_is_expressed_in_xarm_basis():
    initial_controller = np.eye(4)
    controller = np.eye(4)
    controller[:3, :3] = Rotation.from_euler("y", 30.0, degrees=True).as_matrix()
    xarm = np.eye(4)
    xarm[:3, :3] = Rotation.from_euler("z", -20.0, degrees=True).as_matrix()

    target = quest_teleop.quest_delta_to_xarm_target(
        initial_controller,
        controller,
        xarm,
    )

    basis = quest_teleop.OPENXR_TO_XARM_ROTATION
    expected = xarm[:3, :3] @ basis @ controller[:3, :3] @ basis.T
    np.testing.assert_allclose(target[:3, :3], expected, atol=1e-12)


def test_tracking_loss_requires_release_before_grip_can_reactivate():
    gate = quest_teleop.QuestTeleopGate()

    assert gate.update(pose_fresh=True, grip_fresh=True, grip_held=True) == "start"
    assert (
        gate.update(pose_fresh=False, grip_fresh=True, grip_held=True) == "pose_stale"
    )
    assert (
        gate.update(pose_fresh=True, grip_fresh=True, grip_held=True)
        == "release_to_rearm"
    )
    assert (
        gate.update(pose_fresh=True, grip_fresh=True, grip_held=False)
        == "grip_released"
    )
    assert gate.update(pose_fresh=True, grip_fresh=True, grip_held=True) == "start"


def test_stale_grip_value_cannot_reactivate_teleop():
    gate = quest_teleop.QuestTeleopGate()

    assert (
        gate.update(pose_fresh=True, grip_fresh=False, grip_held=True) == "grip_stale"
    )
    assert (
        gate.update(pose_fresh=True, grip_fresh=True, grip_held=True)
        == "release_to_rearm"
    )
    assert (
        gate.update(pose_fresh=True, grip_fresh=True, grip_held=False)
        == "grip_released"
    )
    assert gate.update(pose_fresh=True, grip_fresh=True, grip_held=True) == "start"


def test_vive_deadman_accepts_only_fresh_grip_above_threshold():
    pressed = quest_teleop.quest_grip_deadman_pressed

    assert pressed(
        grip=0.8,
        updated_at=9.9,
        now=10.0,
        threshold=0.5,
        max_age=0.25,
    )
    assert not pressed(
        grip=0.4,
        updated_at=9.9,
        now=10.0,
        threshold=0.5,
        max_age=0.25,
    )
    assert not pressed(
        grip=0.8,
        updated_at=9.0,
        now=10.0,
        threshold=0.5,
        max_age=0.25,
    )
    assert not pressed(
        grip=0.8,
        updated_at=None,
        now=10.0,
        threshold=0.5,
        max_age=0.25,
    )


def test_quest_grip_adapter_overrides_state_and_delegates_device_calls():
    class FakeDevice:
        def get_state(self):
            return None

        def get_data(self):
            return {"Right": "vive-data"}

    states = iter((1, 0))
    adapter = quest_teleop.QuestGripTeleopStateAdapter(
        FakeDevice(), lambda: next(states)
    )

    assert adapter.get_state() == 1
    assert adapter.get_state() == 0
    assert adapter.get_data() == {"Right": "vive-data"}


def test_quest_grip_adapter_rejects_non_teleop_state():
    adapter = quest_teleop.QuestGripTeleopStateAdapter(object(), lambda: 2)

    with np.testing.assert_raises_regex(ValueError, "invalid state"):
        adapter.get_state()
