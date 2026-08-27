from collections import deque
import threading
import time

import numpy as np
import pytest

from paradex.inference.act_xarm_allegro.controls import DeadmanState
from paradex.inference.act_xarm_allegro.core import (
    ACTION_DIM,
    RunnerConfig,
    STATE_DIM,
    SafetyConfig,
    SafetyFilter,
    decode_action,
    rotation_6d_to_matrix,
)
from paradex.inference.act_xarm_allegro.runner import HardwareRunner
from paradex.inference.act_xarm_allegro import runner as runner_module
from paradex.inference.act_xarm_allegro.transport import RobotFeedback


def make_config() -> SafetyConfig:
    return SafetyConfig(
        state_lower=np.full(STATE_DIM, -2.0),
        state_upper=np.full(STATE_DIM, 2.0),
        action_lower=np.full(ACTION_DIM, -2.0),
        action_upper=np.full(ACTION_DIM, 2.0),
        control_hz=10.0,
        max_linear_speed_m_s=1.0,
        max_angular_speed_deg_s=100.0,
        position_margin_m=0.0,
        rotation_margin_deg=0.0,
        max_hand_speed_rad_s=1.0,
    )


def identity_action() -> np.ndarray:
    action = np.zeros(ACTION_DIM)
    action[3:9] = [1, 0, 0, 0, 1, 0]
    return action


def test_rotation_projection_and_decode_contract():
    rotation = rotation_6d_to_matrix([2, 0, 0, 1, 3, 0])
    np.testing.assert_allclose(rotation.T @ rotation, np.eye(3), atol=1e-8)
    assert np.linalg.det(rotation) == pytest.approx(1.0)
    decoded = decode_action(identity_action())
    assert decoded.tcp_transform.shape == (4, 4)
    assert decoded.allegro_target.shape == (16,)
    assert decoded.raw.shape == (25,)


@pytest.mark.parametrize("value", ([0, 0, 0, 0, 1, 0], [1, 0, 0, 2, 0, 0]))
def test_rotation_projection_rejects_degenerate_inputs(value):
    with pytest.raises(ValueError):
        rotation_6d_to_matrix(value)


def test_state_order_and_start_support_gate():
    arm = np.arange(6, dtype=float)
    hand = np.arange(16, dtype=float) + 10
    state = np.concatenate((arm, hand))
    np.testing.assert_array_equal(state[:6], arm)
    np.testing.assert_array_equal(state[6:], hand)
    config = make_config()
    config.state_upper[:] = 30
    assert SafetyFilter(config).validate_start(state, np.eye(4)).accepted
    state[0] = 31
    verdict = SafetyFilter(config).validate_start(state, np.eye(4))
    assert verdict.accepted
    warning = SafetyFilter(config).training_support_warning(state, np.eye(4))
    assert warning is not None
    assert "xarm_joint_0=31.000000 outside [-2.000000, 30.000000]" in warning


def test_start_gate_rejects_non_rigid_tcp_feedback():
    tcp = np.eye(4)
    tcp[0, 0] = 2.0
    verdict = SafetyFilter(make_config()).validate_start(np.zeros(22), tcp)
    assert verdict.reason == "start_tcp_rotation_invalid"


def test_allegro_start_target_must_be_physically_bounded():
    config = make_config()
    config.allegro_start_target = np.zeros(16)
    config.__post_init__()
    np.testing.assert_array_equal(config.allegro_start_target, np.zeros(16))

    with pytest.raises(ValueError, match="allegro_start_target violates physical bounds"):
        SafetyConfig(
            state_lower=np.full(STATE_DIM, -2.0),
            state_upper=np.full(STATE_DIM, 2.0),
            action_lower=np.full(ACTION_DIM, -2.0),
            action_upper=np.full(ACTION_DIM, 2.0),
            allegro_start_target=np.full(16, 99.0),
        )


def test_action_rate_and_bounds_are_rejected_not_clipped():
    safety = SafetyFilter(make_config())
    previous_tcp = np.eye(4)
    previous_hand = np.zeros(16)
    accepted = decode_action(identity_action())
    assert safety.validate_action(accepted, previous_tcp, previous_hand).accepted

    fast = identity_action()
    fast[0] = 0.11
    assert safety.validate_action(decode_action(fast), previous_tcp, previous_hand).reason == "tcp_translation_rate_exceeded"

    out_of_bounds = identity_action()
    out_of_bounds[9] = 3.0
    verdict = safety.validate_action(decode_action(out_of_bounds), previous_tcp, previous_hand)
    assert not verdict.accepted
    assert verdict.bounded_action is None


def test_freshness_gate():
    safety = SafetyFilter(make_config())
    now = time.monotonic_ns()
    assert safety.validate_freshness(now, now, now).accepted
    old = now - 101_000_000
    assert safety.validate_freshness(old, now, now).reason == "camera_observation_stale"


def test_deadman_transitions_and_rearm_gate():
    state = DeadmanState()
    state.press("esc")
    state.press("r")
    assert not state.consume_rearm(False)
    assert state.snapshot().aborted
    state.press("r")
    assert state.consume_rearm(True)
    armed = state.snapshot()
    assert not armed.aborted and armed.enable_generation == 1
    state.press("r")
    assert not state.snapshot().rearm_requested
    state.press("esc")
    assert state.snapshot().aborted


def test_fault_clears_remaining_action_queue():
    runner = HardwareRunner.__new__(HardwareRunner)
    runner.queue = deque([np.zeros(25), np.ones(25)])
    runner.faults = 0
    runner.safety = type("S", (), {"config": type("C", (), {"max_consecutive_faults": 3})()})()
    runner.config = type("C", (), {"mode": "shadow"})()
    runner.logger = type("L", (), {"event": lambda *_args, **_kwargs: None})()
    runner._fault("test")
    assert len(runner.queue) == 0
    assert runner.faults == 1


def test_live_preposition_issues_hand_only_without_convergence_gate():
    now = time.monotonic_ns()
    feedback = RobotFeedback(
        state=np.zeros(22),
        tcp_transform=np.eye(4),
        state_monotonic_ns=now,
        bridge_monotonic_ns=now,
        connected=True,
        error=False,
        latched=False,
        allow_live=True,
        status="ready",
    )

    class FakeBridge:
        def __init__(self):
            self.targets = []
            self.rearmed = 0
            self.aborted = 0

        def send_rearm(self):
            self.rearmed += 1

        def receive_feedback(self, **_kwargs):
            if self.aborted:
                return RobotFeedback(
                    state=feedback.state,
                    tcp_transform=feedback.tcp_transform,
                    state_monotonic_ns=time.monotonic_ns(),
                    bridge_monotonic_ns=time.monotonic_ns(),
                    connected=True,
                    error=False,
                    latched=False,
                    allow_live=True,
                    status="hand_holding",
                )
            return feedback

        def send_hand_target(self, target, **_kwargs):
            self.targets.append(np.asarray(target).copy())

        def send_hand_hold(self):
            self.aborted += 1

    runner = HardwareRunner.__new__(HardwareRunner)
    runner.bridge = FakeBridge()
    runner.safety = SafetyFilter(
        SafetyConfig(
            state_lower=np.full(22, -2.0),
            state_upper=np.full(22, 2.0),
            action_lower=np.full(25, -2.0),
            action_upper=np.full(25, 2.0),
            allegro_start_target=np.zeros(16),
        )
    )
    runner.config = type(
        "Config",
        (),
        {
            "preposition_timeout_seconds": 1.0,
            "preposition_tolerance_rad": 0.20,
            "control_hz": 30.0,
        },
    )()
    runner.logger = type("Logger", (), {"event": lambda *_args, **_kwargs: None})()

    result = runner._preposition_allegro(feedback)

    assert result is feedback
    assert runner.bridge.rearmed == 1
    assert len(runner.bridge.targets) == 3
    assert runner.bridge.aborted == 1


def test_async_planner_does_not_pause_fixed_rate_action_publication(monkeypatch, tmp_path):
    control_hz = 40.0
    action_steps = 5
    action = identity_action()
    now = time.monotonic_ns()
    feedback = RobotFeedback(
        state=np.zeros(22),
        tcp_transform=np.eye(4),
        state_monotonic_ns=now,
        bridge_monotonic_ns=now,
        connected=True,
        error=False,
        latched=False,
        allow_live=True,
        status="ready",
    )

    class FakePolicy:
        def infer(self, _images, _state, steps):
            # Longer than one publisher period, but shorter than the four-step
            # refill runway.
            time.sleep(0.060)
            return np.repeat(action[None], steps, axis=0), 60.0

    class FakeBridge:
        def __init__(self, *_args, **_kwargs):
            pass

        def receive_feedback(self, **_kwargs):
            return feedback

        def close(self):
            pass

    class FakeCameras:
        def __init__(self, *_args, **_kwargs):
            pass

        def start(self):
            pass

        def get_pair(self, **_kwargs):
            stamp = time.monotonic_ns()
            keys = (
                "observation.images.cam_23029839",
                "observation.images.cam_25452066",
            )
            return type(
                "Pair",
                (),
                {
                    "images": {key: np.zeros((4, 4, 3), dtype=np.uint8) for key in keys},
                    "frame_ids": {key: 1 for key in keys},
                    "raw_frame_ids": {key: 1 for key in keys},
                    "jpeg_bytes": {},
                    "received_monotonic_ns": stamp,
                },
            )()

        def close(self):
            pass

    class FakeLogger:
        def __init__(self):
            self.action_times = []
            self.lock = threading.Lock()

        def event(self, kind, **_fields):
            if kind == "action":
                with self.lock:
                    self.action_times.append(time.monotonic())

        def inference_boundary(self, *_args, **_kwargs):
            time.sleep(0.010)

    monkeypatch.setattr(runner_module, "load_policy", lambda _config: FakePolicy())
    monkeypatch.setattr(runner_module, "HardwareBridgeClient", FakeBridge)
    monkeypatch.setattr(runner_module, "SynchronizedCameraStream", FakeCameras)

    logger = FakeLogger()
    config = RunnerConfig(
        mode="shadow",
        control_hz=control_hz,
        action_steps=action_steps,
        duration_seconds=1.0,
        output_dir=tmp_path,
    )
    runner = HardwareRunner(config, SafetyFilter(make_config()), logger, max_chunks=2)

    assert runner.run() == 2
    assert len(logger.action_times) == 2 * action_steps
    intervals = np.diff(logger.action_times)
    assert np.median(intervals) == pytest.approx(1.0 / control_hz, abs=0.012)
    assert np.max(intervals) < 0.050
