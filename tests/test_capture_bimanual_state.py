from threading import Event

import numpy as np
import pytest

from paradex.dataset_acqusition import capture as capture_module
from paradex.io.teleop.vive import receiver as vive_receiver_module
from paradex.retargetor.hand_regargetor import (
    inspire,
    inspire_from_manus_ergonomics,
)
from paradex.retargetor.unimanual import Retargetor, _resolve_hand
from paradex.transforms.coordinate import DEVICE2WRIST


class FakeArm:
    def __init__(self):
        self.moves = []

    def get_data(self):
        return {"position": np.eye(4)}

    def move(self, pose):
        self.moves.append(pose)


def test_hand_command_rate_limiter_matches_alignment_ui_sample_and_hold():
    limiter = capture_module._HandCommandRateLimiter(30.0)

    assert limiter.is_due(10.0)
    assert not limiter.is_due(10.01)
    assert not limiter.is_due(10.0 + (1.0 / 30.0) - 1.0e-6)
    assert limiter.is_due(10.0 + (1.0 / 30.0))


def test_hand_command_rate_limiter_can_be_explicitly_disabled():
    limiter = capture_module._HandCommandRateLimiter(None)

    assert limiter.is_due(10.0)
    assert limiter.is_due(10.0001)


def test_allegro_teleop_diagnostic_keeps_manus_command_and_feedback_aligned(tmp_path):
    path = tmp_path / "allegro_diagnostic.npz"
    logger = capture_module._AllegroTeleopDiagnosticLogger(path)
    frame = {
        "wrist": np.eye(4),
        "index_distal": np.diag([1.0, 1.0, 1.0, 1.0]),
    }
    frame["index_distal"][:3, 3] = [0.01, 0.02, 0.03]
    action = np.arange(16, dtype=float) / 10.0
    logger.record(
        teleop_data={
            "Right": frame,
            "ergonomics": {"Right": {"IndexMCPStretch": 42.0}},
            "time": 123.0,
        },
        state=0,
        hand_name="allegro_v5",
        retargeter_action=action,
        controller_target=action + 1.0,
        feedback={
            "qpos": action + 2.0,
            "action": action + 3.0,
            "joint_names": [f"joint_{index}_0" for index in range(16)],
            "is_connected": True,
            "tactile": [7, 9],
        },
    )
    logger.flush()

    with np.load(path, allow_pickle=False) as saved:
        assert saved["manus_transforms"].shape == (1, 2, 4, 4)
        assert saved["manus_joint_names"].tolist() == ["index_distal", "wrist"]
        np.testing.assert_array_equal(saved["retargeter_action"][0], action)
        np.testing.assert_array_equal(saved["controller_target"][0], action + 1.0)
        np.testing.assert_array_equal(saved["feedback_qpos"][0], action + 2.0)
        assert saved["feedback_joint_names"].tolist() == [
            f"joint_{index}_0" for index in range(16)
        ]
        np.testing.assert_array_equal(saved["tactile"][0], [7.0, 9.0])


def test_allegro_v5_feedback_hold_target_uses_driver_names_not_feedback_order():
    names = list(reversed(capture_module.ALLEGRO_V5_DRIVER_JOINT_NAMES))
    values = np.arange(16, dtype=float)
    feedback = {
        "is_connected": True,
        "qpos": values,
        "joint_names": names,
    }

    target = capture_module._allegro_v5_feedback_hold_target(feedback)

    expected_by_name = dict(zip(names, values))
    np.testing.assert_array_equal(
        target,
        [expected_by_name[name] for name in capture_module.ALLEGRO_V5_DRIVER_JOINT_NAMES],
    )


def test_allegro_v5_feedback_hold_target_requires_live_feedback():
    assert capture_module._allegro_v5_feedback_hold_target({}) is None


def test_ui_aligned_allegro_pause_holds_current_feedback_not_prior_manus_target(
    monkeypatch,
):
    monkeypatch.setattr(capture_module.chime, "warning", lambda **_kwargs: None)
    monkeypatch.setattr(capture_module.chime, "success", lambda **_kwargs: None)
    events = make_events()
    feedback_target = np.arange(16, dtype=float) / 10.0

    class PausedDevice:
        def get_data(self):
            return {"Left": object(), "Right": {"wrist": np.eye(4)}}

        def get_state(self):
            events["exit"].set()
            return 1

    class FeedbackHand:
        def __init__(self):
            self.moves = []

        def get_data(self):
            return {
                "is_connected": True,
                "qpos": feedback_target,
                "joint_names": capture_module.ALLEGRO_V5_DRIVER_JOINT_NAMES,
            }

        def move(self, action):
            self.moves.append(action)

    class Retargetor:
        def start(self, _home_pose):
            pass

        def stop(self):
            pass

    session = capture_module.CaptureSession.__new__(capture_module.CaptureSession)
    session.teleop_device = PausedDevice()
    session.hand_side = "Right"
    session.arm = None
    session.hand = FeedbackHand()
    session.hand_name = "allegro_v5"
    session.retargetor = Retargetor()
    session.teleop_name = "vive"
    session.save_path = None

    assert session.teleop(session_events=events, state_policy="keyboard_control") == "exit"
    assert len(session.hand.moves) == 1
    np.testing.assert_array_equal(session.hand.moves[0], feedback_target)


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

    assert receiver_kwargs == {
        "hand_side": "bimanual",
        "require_left_control": True,
        "use_vive": True,
    }
    assert session.arm_left is not None
    assert session.arm_right is not None


def test_capture_session_routes_manus_only_hand_teleop_without_an_arm(monkeypatch):
    receiver_kwargs = {}

    class FakeViveReceiver:
        def __init__(self, **kwargs):
            receiver_kwargs.update(kwargs)

    monkeypatch.setattr(capture_module, "get_hand", lambda **_kwargs: FakeHand())
    monkeypatch.setattr(
        vive_receiver_module,
        "ViveManusROSReceiver",
        FakeViveReceiver,
    )

    session = capture_module.CaptureSession(
        camera=False,
        arm=None,
        hand="allegro_v5",
        teleop="vive",
        hand_side="right",
        use_vive=False,
    )

    assert session.arm is None
    assert session.hand is not None
    assert receiver_kwargs == {
        "hand_side": "right",
        "require_left_control": False,
        "use_vive": False,
    }


def test_keyboard_hand_only_teleop_does_not_wait_for_optional_left_glove(monkeypatch):
    monkeypatch.setattr(capture_module.chime, "warning", lambda **_kwargs: None)
    monkeypatch.setattr(capture_module.chime, "success", lambda **_kwargs: None)

    events = make_events()

    class ManusOnlyDevice:
        def get_data(self):
            events["exit"].set()
            return {"Left": None, "Right": object()}

        def get_state(self):
            return None

    class HandOnlyRetargetor:
        def start(self, _home_pose):
            pass

        def get_action(self, _data):
            return np.eye(4), np.array([0.1, 0.2])

    session = capture_module.CaptureSession.__new__(capture_module.CaptureSession)
    session.teleop_device = ManusOnlyDevice()
    session.hand_side = "Right"
    session.arm = None
    session.hand = FakeHand()
    session.retargetor = HandOnlyRetargetor()
    session.teleop_name = "vive"
    session.save_path = None

    assert session.teleop(session_events=events, state_policy="keyboard_control") == "exit"
    assert len(session.hand.moves) == 1
    np.testing.assert_allclose(session.hand.moves[0], [0.1, 0.2])


def test_teleop_loop_callback_runs_while_device_data_is_unavailable(monkeypatch):
    monkeypatch.setattr(capture_module.chime, "warning", lambda **_kwargs: None)
    monkeypatch.setattr(capture_module.chime, "success", lambda **_kwargs: None)
    monkeypatch.setattr(capture_module.time, "sleep", lambda _seconds: None)
    events = make_events()

    class NoDataDevice:
        def get_data(self):
            return {"Left": None, "Right": None}

    class IdleRetargetor:
        def start(self, _home_pose):
            pass

    session = capture_module.CaptureSession.__new__(capture_module.CaptureSession)
    session.teleop_device = NoDataDevice()
    session.hand_side = "Right"
    session.arm = None
    session.hand = None
    session.retargetor = IdleRetargetor()
    session.teleop_name = "vive"
    session.save_path = None
    callback_sessions = []

    def stop_from_callback(active_session):
        callback_sessions.append(active_session)
        events["exit"].set()

    assert session.teleop(
        session_events=events,
        state_policy="keyboard_control",
        loop_callback=stop_from_callback,
    ) == "exit"
    assert callback_sessions == [session]


def test_recording_stop_event_returns_before_reading_another_teleop_frame(monkeypatch):
    monkeypatch.setattr(capture_module.chime, "warning", lambda **_kwargs: None)
    monkeypatch.setattr(capture_module.chime, "info", lambda **_kwargs: None)
    events = make_events()
    events["stop"].set()

    class DeviceThatMustNotBeRead:
        def get_data(self):
            raise AssertionError("stop must be handled before reading teleop data")

    class IdleRetargetor:
        def start(self, _home_pose):
            pass

    session = capture_module.CaptureSession.__new__(capture_module.CaptureSession)
    session.teleop_device = DeviceThatMustNotBeRead()
    session.hand_side = "Right"
    session.arm = None
    session.retargetor = IdleRetargetor()
    session.teleop_name = "vive"
    session.save_path = "capture/in-progress"

    assert session.teleop(
        session_events=events,
        state_policy="keyboard_control",
    ) == "stop"


def test_capture_session_replaces_manus_hand_action_with_external_provider(monkeypatch):
    monkeypatch.setattr(capture_module.chime, "warning", lambda **_kwargs: None)
    monkeypatch.setattr(capture_module.chime, "success", lambda **_kwargs: None)
    events = make_events()

    class ViveOnlyDevice:
        def get_data(self):
            events["exit"].set()
            return {"Left": None, "Right": {"wrist": np.eye(4)}}

        def get_state(self):
            return None

    class ArmOnlyRetargetor:
        def start(self, _home_pose):
            pass

        def get_action(self, _data):
            return np.eye(4), None

    session = capture_module.CaptureSession.__new__(capture_module.CaptureSession)
    session.teleop_device = ViveOnlyDevice()
    session.hand_side = "Right"
    session.arm = None
    session.hand = FakeHand()
    session.retargetor = ArmOnlyRetargetor()
    session.teleop_name = "vive"
    session.save_path = None
    session.hand_action_provider = lambda: np.array([0.3, 0.4])

    assert session.teleop(session_events=events, state_policy="keyboard_control") == "exit"
    np.testing.assert_allclose(session.hand.moves[0], [0.3, 0.4])


def test_unimanual_arm_deadman_holds_arm_but_keeps_external_hand_active(monkeypatch):
    monkeypatch.setattr(capture_module.chime, "warning", lambda **_kwargs: None)
    monkeypatch.setattr(capture_module.chime, "success", lambda **_kwargs: None)
    events = make_events()

    class ViveOnlyDevice:
        def __init__(self):
            self.calls = 0

        def get_data(self):
            self.calls += 1
            if self.calls == 2:
                events["exit"].set()
            return {"Left": None, "Right": {"wrist": np.eye(4)}}

        def get_state(self):
            return None

    class ArmRetargetor:
        def __init__(self):
            self.starts = []
            self.actions = 0

        def start(self, home_pose):
            self.starts.append(home_pose.copy())

        def get_action(self, _data):
            self.actions += 1
            return np.eye(4), None

    enabled = iter((False, True))
    session = capture_module.CaptureSession.__new__(capture_module.CaptureSession)
    session.teleop_device = ViveOnlyDevice()
    session.hand_side = "Right"
    session.arm = FakeArm()
    session.hand = FakeHand()
    session.retargetor = ArmRetargetor()
    session.teleop_name = "vive"
    session.save_path = None
    session.hand_action_provider = lambda: np.array([0.3, 0.4])
    session.arm_command_enabled_provider = lambda: next(enabled)

    assert session.teleop(session_events=events, state_policy="keyboard_control") == "exit"
    assert len(session.hand.moves) == 2
    assert len(session.arm.moves) == 1
    assert session.retargetor.actions == 1
    # Initial teleop setup plus rebase on the deadman's rising edge.
    assert len(session.retargetor.starts) == 2


def test_capture_session_accepts_unimanual_external_hand_provider():
    session = capture_module.CaptureSession(
        camera=False,
        arm=None,
        hand=None,
        teleop=None,
        hand_side="right",
        hand_action_provider=lambda: np.zeros(16),
    )

    assert session.hand_action_provider is not None


def test_capture_session_routes_per_hand_network_interfaces(monkeypatch):
    hand_calls = []

    def fake_get_hand(**kwargs):
        hand_calls.append(kwargs)
        return FakeHand()

    monkeypatch.setattr(capture_module, "get_hand", fake_get_hand)

    session = capture_module.CaptureSession(
        camera=False,
        hand="inspire",
        hand_side="bimanual",
        ip=True,
        hand_kwargs={
            "right": {
                "interface": "enp8s0f1",
                "host": "192.168.11.211",
            },
            "left": {
                "interface": "enp8s0f2",
                "host": "192.168.11.210",
            },
        },
    )

    assert session.hand_left is not None
    assert session.hand_right is not None
    assert hand_calls == [
        {
            "hand_name": "inspire",
            "tactile": False,
            "ip": True,
            "hand_side": "left",
            "interface": "enp8s0f2",
            "host": "192.168.11.210",
        },
        {
            "hand_name": "inspire",
            "tactile": False,
            "ip": True,
            "hand_side": "right",
            "interface": "enp8s0f1",
            "host": "192.168.11.211",
        },
    ]


def test_capture_session_no_timestamp_skips_monitor_connection(monkeypatch):
    class UnexpectedTimestampMonitor:
        def __init__(self, **_kwargs):
            raise AssertionError("TimestampMonitor must not be created")

    monkeypatch.setattr(
        capture_module,
        "remote_camera_controller",
        lambda **_kwargs: object(),
    )
    monkeypatch.setattr(capture_module, "UTGE900", lambda **_kwargs: object())
    monkeypatch.setattr(
        capture_module,
        "TimestampMonitor",
        UnexpectedTimestampMonitor,
    )

    session = capture_module.CaptureSession(
        camera=True,
        timestamp=False,
    )

    assert session.camera is not None
    assert session.sync_generator is not None
    assert session.timestamp_monitor is None


def test_inspire_retargeting_uses_lefthand_branch_for_left_thumb():
    pose = {"wrist": np.eye(4), "thumb_metacarpal": np.eye(4)}
    for finger_name in ("thumb", "index", "middle", "ring", "pinky"):
        distal = np.eye(4)
        if finger_name == "thumb":
            distal[:3, 3] = [1.0, 1.0, 1.0]
        pose[f"{finger_name}_distal"] = distal

    right_retargetor = _resolve_hand("inspire", is_right=True)
    left_retargetor = _resolve_hand("inspire", is_right=False)
    right_action = right_retargetor(pose)
    left_action = left_retargetor(pose)

    np.testing.assert_allclose(right_action, inspire(pose))
    assert right_action[5] == pytest.approx(500.0)
    assert left_action[5] == pytest.approx(0.0)
    expected_left_thumb = (
        np.arcsin(1.0 / np.sqrt(3.0)) / np.pi * 2000.0 * 3.5
        - 1000.0
    )
    assert left_action[4] == pytest.approx(expected_left_thumb)


def test_inspire_maps_manus_angles_to_rh56_register_order():
    ergonomics = {}
    flexions = {
        "Pinky": 157.7,
        "Ring": 0.0,
        "Middle": 157.7 / 2.0,
        "Index": 157.7 / 4.0,
    }
    for finger, total in flexions.items():
        for joint in ("MCP", "PIP", "DIP"):
            ergonomics[f"{finger}{joint}Stretch"] = total / 3.0
    ergonomics["ThumbMCPStretch"] = 66.6 / 4.0
    ergonomics["ThumbMCPSpread"] = 75.0 / 4.0

    command = inspire_from_manus_ergonomics(ergonomics)

    np.testing.assert_allclose(
        command,
        [0.0, 1000.0, 500.0, 750.0, 250.0, 200.0],
    )


def test_bimanual_inspire_retargetor_uses_manus_ergonomics():
    ergonomics = {}
    for finger in ("Pinky", "Ring", "Middle", "Index"):
        for joint in ("MCP", "PIP", "DIP"):
            ergonomics[f"{finger}{joint}Stretch"] = 0.0
    ergonomics["ThumbMCPStretch"] = 0.0
    ergonomics["ThumbMCPSpread"] = 0.0

    retargetor = Retargetor(
        hand_name="inspire",
        hand_side="Bimanual",
    )
    retargetor.start({"Left": np.eye(4), "Right": np.eye(4)})
    actions = retargetor.get_action(
        {
            "Left": {"wrist": np.eye(4)},
            "Right": {"wrist": np.eye(4)},
            "ergonomics": {
                "Left": ergonomics,
                "Right": ergonomics,
            },
        }
    )

    np.testing.assert_allclose(
        actions[2],
        [1000.0, 1000.0, 1000.0, 1000.0, 0.0, 0.0],
    )
    np.testing.assert_allclose(actions[3], actions[2])


def test_bimanual_inspire_locks_thumb_four_when_both_thumb_five_are_high():
    ergonomics = {}
    for finger in ("Pinky", "Ring", "Middle", "Index"):
        for joint in ("MCP", "PIP", "DIP"):
            ergonomics[f"{finger}{joint}Stretch"] = 0.0
    ergonomics["ThumbMCPStretch"] = 66.6
    ergonomics["ThumbMCPSpread"] = 0.0

    retargetor = Retargetor(
        hand_name="inspire",
        hand_side="Bimanual",
    )
    retargetor.start({"Left": np.eye(4), "Right": np.eye(4)})
    actions = retargetor.get_action(
        {
            "Left": {"wrist": np.eye(4)},
            "Right": {"wrist": np.eye(4)},
            "ergonomics": {
                "Left": ergonomics,
                "Right": ergonomics,
            },
        }
    )

    assert actions[2][5] == pytest.approx(800.0)
    assert actions[3][5] == pytest.approx(800.0)
    assert actions[2][4] == pytest.approx(1000.0)
    assert actions[3][4] == pytest.approx(1000.0)


def test_bimanual_inspire_locks_thumb_four_independently_per_hand():
    low = {}
    for finger in ("Pinky", "Ring", "Middle", "Index"):
        for joint in ("MCP", "PIP", "DIP"):
            low[f"{finger}{joint}Stretch"] = 0.0
    low["ThumbMCPStretch"] = 0.0
    low["ThumbMCPSpread"] = 0.0
    high = dict(low)
    high["ThumbMCPStretch"] = 66.6

    retargetor = Retargetor(
        hand_name="inspire",
        hand_side="Bimanual",
    )
    retargetor.start({"Left": np.eye(4), "Right": np.eye(4)})
    actions = retargetor.get_action(
        {
            "Left": {"wrist": np.eye(4)},
            "Right": {"wrist": np.eye(4)},
            "ergonomics": {
                "Left": high,
                "Right": low,
            },
        }
    )

    assert actions[2][5] == pytest.approx(800.0)
    assert actions[3][5] == pytest.approx(0.0)
    assert actions[2][4] == pytest.approx(1000.0)
    assert actions[3][4] == pytest.approx(0.0)


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


def test_vive_bimanual_uses_side_specific_xarm_wrist_rotations():
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


def test_xarm_translation_delta_uses_device_to_global_rotation():
    retargetor = Retargetor(
        arm_name="xarm",
        hand_side="Right",
        teleop_name="vive",
    )
    retargetor.device2global = np.diag([-1.0, 1.0, -1.0, 1.0])
    retargetor.start(np.eye(4))

    initial_wrist = np.eye(4)
    retargetor.get_action({"Right": {"wrist": initial_wrist}})

    moved_wrist = np.eye(4)
    moved_wrist[:3, 3] = [0.1, 0.2, 0.3]
    arm_action, _ = retargetor.get_action(
        {"Right": {"wrist": moved_wrist}}
    )

    np.testing.assert_allclose(arm_action[:3, 3], [-0.1, 0.2, -0.3])
