from collections import deque
import time

import numpy as np
import pytest

from paradex.inference.act_xarm_allegro.controls import DeadmanState
from paradex.inference.act_xarm_allegro.core import (
    ACTION_DIM,
    STATE_DIM,
    RunnerConfig,
    SafetyConfig,
    SafetyFilter,
    decode_action,
    rotation_6d_to_matrix,
)
from paradex.inference.act_xarm_allegro.runner import (
    HardwareRunner,
    TemporalActionEnsembler,
)


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
    assert not SafetyFilter(config).validate_start(state, np.eye(4)).accepted


def test_start_gate_rejects_non_rigid_tcp_feedback():
    tcp = np.eye(4)
    tcp[0, 0] = 2.0
    verdict = SafetyFilter(make_config()).validate_start(np.zeros(22), tcp)
    assert verdict.reason == "start_tcp_rotation_invalid"


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
    snapshot = state.snapshot()
    assert not snapshot.aborted
    assert snapshot.enable_generation == 1


def test_runner_defaults_to_the_legacy_direct_gate_behavior():
    assert not RunnerConfig(mode="shadow").enforce_safety_gates


def test_fault_clears_remaining_action_queue():
    runner = HardwareRunner.__new__(HardwareRunner)
    runner.queue = deque([np.zeros(25), np.ones(25)])
    runner.ensembler = TemporalActionEnsembler(decay=0.01)
    runner.faults = 0
    runner.safety = type("S", (), {"config": type("C", (), {"max_consecutive_faults": 3})()})()
    runner.config = type("C", (), {"mode": "shadow"})()
    runner.logger = type("L", (), {"event": lambda *_args, **_kwargs: None})()
    runner._fault("test")
    assert len(runner.queue) == 0
    assert runner.faults == 1


def test_temporal_ensemble_averages_overlapping_chunk_predictions():
    ensembler = TemporalActionEnsembler(decay=0.0)
    first = np.tile(np.arange(1.0, 5.0)[:, None], (1, 25))
    second = np.tile(np.arange(10.0, 14.0)[:, None], (1, 25))

    ensembler.add(first)
    np.testing.assert_array_equal(ensembler.take(2), first[:2])

    ensembler.add(second)
    actions, contributors = ensembler.take(2, return_contributors=True)
    np.testing.assert_array_equal(actions[:, 0], [6.5, 7.5])
    np.testing.assert_array_equal(contributors, [2, 2])


def test_temporal_ensemble_prefers_newer_predictions_when_decay_is_positive():
    ensembler = TemporalActionEnsembler(decay=1.0)
    old = np.zeros((3, 25))
    new = np.full((3, 25), 10.0)

    ensembler.add(old)
    ensembler.take(1)
    ensembler.add(new)
    action, contributors = ensembler.take(1, return_contributors=True)

    assert contributors.tolist() == [2]
    assert action[0, 0] > 5.0
