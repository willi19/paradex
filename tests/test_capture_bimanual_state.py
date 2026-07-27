from threading import Event

import numpy as np

from paradex.dataset_acqusition import capture as capture_module


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

