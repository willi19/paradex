import numpy as np

from paradex.retargetor.hand_regargetor import allegro_v5
from paradex.retargetor.experiment.capture_robot_allegro_alignment import (
    AlignmentUiAllegroRetargetor,
)


class _RecordingRetargeter:
    def __init__(self):
        self.calls = []

    def __call__(self, frame, **kwargs):
        self.calls.append((frame, kwargs))
        return np.arange(16, dtype=np.float64)


def _adapter(mode):
    adapter = object.__new__(AlignmentUiAllegroRetargetor)
    adapter.hand_side = "Right"
    adapter.hand_name = "allegro_v5"
    adapter.retargeter_mode = mode
    adapter.hand_retargetor = _RecordingRetargeter()
    adapter._compute_wrist_pose = lambda side, data: np.eye(4)
    return adapter


def test_direct_mode_constructs_the_alignment_ui_allegro_retargeter():
    adapter = AlignmentUiAllegroRetargetor(arm_name=None, mode="direct")

    assert adapter.hand_retargetor is allegro_v5


def test_direct_mode_passes_right_manus_ergonomics_like_alignment_ui():
    adapter = _adapter("direct")
    frame = {"wrist": np.eye(4)}
    ergonomics = {"ThumbMCPStretch": 42.0}

    wrist, action = adapter.get_action(
        {"Right": frame, "ergonomics": {"Right": ergonomics}}
    )

    np.testing.assert_allclose(wrist, np.eye(4))
    np.testing.assert_allclose(action, np.arange(16))
    assert adapter.hand_retargetor.calls == [
        (frame, {"ergonomics": ergonomics})
    ]
