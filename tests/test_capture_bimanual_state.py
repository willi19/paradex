from threading import Event

import numpy as np
import pytest

from paradex.dataset_acqusition import capture as capture_module
from paradex.io.teleop.vive import receiver as vive_receiver_module
from paradex.retargetor.unimanual import Retargetor
from paradex.transforms.coordinate import DEVICE2WRIST


class FakeArm:
    def __init__(self):
        self.moves = []

    def get_data(self):
        return {"position": np.eye(4)}

    def move(self, pose):
        self.moves.append(pose)


class FakeHand:
    def __init__(self):
        self.moves = []

    def move(self, action):
        self.moves.append(action)


class FakeRetargetor:
    def __init__(self):
        self.starts = []
        self.stop_count = 0
        self.action_count = 0

    def start(self, home_pose):
        self.starts.append(home_pose)

    def stop(self):
        self.stop_count += 1

    def get_action(self, _data):
        self.action_count += 1
        return np.eye(4), np.eye(4), np.array([1.0]), np.array([2.0])


class FakeTeleopDevice:
    def get_data(self):
        return {"Left": object(), "Right": object()}


def make_session():
    session = capture_module.CaptureSession.__new__(capture_module.CaptureSession)
    session.teleop_device = FakeTeleopDevice()
    session.hand_side = "Bimanual"
    session.arm_left = FakeArm()
    session.arm_right = FakeArm()
    session.hand_left = FakeHand()
    session.hand_right = FakeHand()
    session.retargetor = FakeRetargetor()
    session.teleop_name = "vive"
    session.save_path = None
    return session


def make_events():
    return {"save": Event(), "stop": Event(), "exit": Event()}


def test_bimanual_state_zero_moves_both_arms_and_hands(monkeypatch):
    monkeypatch.setattr(capture_module.chime, "warning", lambda **_kwargs: None)
    monkeypatch.setattr(capture_module.chime, "success", lambda **_kwargs: None)
    session = make_session()
    session.save_path = "episode"
    session.state_hist = []
    session.state_time = []
    events = make_events()

    def state_provider():
        events["exit"].set()
        return 0

    result = session.teleop(
        session_events=events,
        state_policy="keyboard_control",
        bimanual_state_provider=state_provider,
    )

    assert result == "exit"
    assert session.retargetor.action_count == 1
    assert session.retargetor.stop_count == 0
    assert len(session.arm_left.moves) == 1
    assert len(session.arm_right.moves) == 1
    assert len(session.hand_left.moves) == 1
    assert len(session.hand_right.moves) == 1
    assert session.state_hist == [0]
    assert len(session.state_time) == 1


def test_bimanual_state_one_stops_retargeting_without_moving(monkeypatch):
    monkeypatch.setattr(capture_module.chime, "warning", lambda **_kwargs: None)
    monkeypatch.setattr(capture_module.chime, "success", lambda **_kwargs: None)
    session = make_session()
    session.save_path = "episode"
    session.state_hist = []
    session.state_time = []
    events = make_events()

    def state_provider():
        events["exit"].set()
        return 1

    result = session.teleop(
        session_events=events,
        state_policy="keyboard_control",
        bimanual_state_provider=state_provider,
    )

    assert result == "exit"
    assert session.retargetor.action_count == 0
    assert session.retargetor.stop_count == 1
    assert not session.arm_left.moves
    assert not session.arm_right.moves
    assert not session.hand_left.moves
    assert not session.hand_right.moves
    assert session.state_hist == [1]
    assert len(session.state_time) == 1


def test_capture_session_routes_vive_bimanual_mode_to_receiver(monkeypatch):
    receiver_kwargs = {}

    class FakeViveReceiver:
        def __init__(self, **kwargs):
            receiver_kwargs.update(kwargs)

    monkeypatch.setattr(capture_module, "get_arm", lambda *_args, **_kwargs: FakeArm())
    monkeypatch.setattr(
        vive_receiver_module,
        "ViveManusROSReceiver",
        FakeViveReceiver,
    )

    session = capture_module.CaptureSession(
        camera=False,
        arm="xarm",
        hand=None,
        teleop="vive",
        hand_side="bimanual",
    )

    assert receiver_kwargs == {"hand_side": "bimanual"}
    assert session.arm_left is not None
    assert session.arm_right is not None


def test_capture_session_rejects_vive_left_only_mode():
    with pytest.raises(ValueError, match="right or bimanual"):
        capture_module.CaptureSession(
            camera=False,
            arm="xarm",
            hand=None,
            teleop="vive",
            hand_side="left",
        )


def test_pose_command_limiter_rejects_translation_jump_without_updating_pose():
    limiter = capture_module._PoseCommandLimiter(np.eye(4), timestamp=0.0)
    jump = np.eye(4)
    jump[0, 3] = 0.02

    filtered, translation_delta, _ = limiter.filter(jump, timestamp=0.01)

    assert filtered is None
    assert translation_delta == pytest.approx(0.02)
    assert np.allclose(limiter.last_sent_pose, np.eye(4))
    assert np.allclose(limiter.last_raw_pose, jump)

    recovered = np.eye(4)
    recovered[0, 3] = 0.023
    filtered, _, _ = limiter.filter(recovered, timestamp=0.02)

    assert filtered is not None
    assert filtered[0, 3] == pytest.approx(0.003)
    assert limiter.last_sent_pose[0, 3] == pytest.approx(0.003)
    assert limiter.last_raw_pose[0, 3] == pytest.approx(0.023)


def test_pose_command_limiter_rejects_rotation_jump():
    limiter = capture_module._PoseCommandLimiter(np.eye(4), timestamp=0.0)
    jump = np.eye(4)
    angle = np.deg2rad(5.0)
    jump[:3, :3] = np.array(
        [
            [np.cos(angle), -np.sin(angle), 0.0],
            [np.sin(angle), np.cos(angle), 0.0],
            [0.0, 0.0, 1.0],
        ]
    )

    filtered, _, rotation_delta_deg = limiter.filter(jump, timestamp=0.01)

    assert filtered is None
    assert rotation_delta_deg == pytest.approx(5.0)
    assert np.allclose(limiter.last_sent_pose, np.eye(4))


def test_vive_bimanual_uses_right_xarm_wrist_rotation_for_both_sides():
    retargetor = Retargetor(
        arm_name="xarm",
        hand_side="Bimanual",
        teleop_name="vive",
    )

    np.testing.assert_array_equal(
        retargetor.device2wrist["Left"],
        DEVICE2WRIST["xarm_vive_Left"],
    )
    np.testing.assert_array_equal(
        retargetor.device2wrist["Right"],
        DEVICE2WRIST["xarm_vive_Right"],
    )
    np.testing.assert_array_equal(
        DEVICE2WRIST["xarm_vive_Left"],
        DEVICE2WRIST["xarm_Right"],
    )
    np.testing.assert_array_equal(
        DEVICE2WRIST["xarm_vive_Right"],
        DEVICE2WRIST["xarm_Right"],
    )
