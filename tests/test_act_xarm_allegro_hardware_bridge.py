import json

import numpy as np

from paradex.inference.act_xarm_allegro.hardware_bridge import (
    XArmAllegroHardware,
    _state_header,
)


class FakeController:
    def __init__(self, data, error=np.bool_(False)):
        self._data = data
        self._error = error

    def get_data(self):
        return self._data

    def is_error(self):
        return self._error

    def move(self, target):
        self.last_target = np.asarray(target).copy()


def test_feedback_and_header_normalize_numpy_scalars_for_json():
    hardware = XArmAllegroHardware.__new__(XArmAllegroHardware)
    hardware.arm = FakeController(
        {
            "qpos": np.zeros(6),
            "position": np.eye(4),
            "state_monotonic_time": np.float64(10.0),
        }
    )
    hardware.hand = FakeController(
        {
            "qpos": np.zeros(16),
            "is_connected": np.bool_(True),
            "state_monotonic_time": np.float64(10.1),
        }
    )

    _state, _tcp, connected, error, state_ns = hardware.feedback()
    assert type(connected) is bool
    assert type(error) is bool
    assert type(state_ns) is int

    header = _state_header(
        state_ns=np.int64(state_ns),
        connected=np.bool_(connected),
        error=np.bool_(error),
        latched=np.bool_(False),
        allow_live=np.bool_(False),
        status="ready",
    )
    encoded = json.dumps(header)
    assert json.loads(encoded)["connected"] is True


def test_hand_only_target_and_hold_do_not_command_arm():
    hardware = XArmAllegroHardware.__new__(XArmAllegroHardware)
    hardware.arm = FakeController(
        {
            "qpos": np.zeros(6),
            "position": np.eye(4),
            "state_monotonic_time": 10.0,
        }
    )
    hardware.hand = FakeController(
        {
            "qpos": np.arange(16, dtype=float),
            "is_connected": True,
            "state_monotonic_time": 10.0,
        }
    )

    target = np.linspace(0.0, 1.0, 16)
    hardware.hand_target(target)
    np.testing.assert_array_equal(hardware.hand.last_target, target)
    assert not hasattr(hardware.arm, "last_target")

    hardware.hold_hand()
    np.testing.assert_array_equal(hardware.hand.last_target, np.arange(16, dtype=float))
    assert not hasattr(hardware.arm, "last_target")
