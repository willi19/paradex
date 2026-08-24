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
