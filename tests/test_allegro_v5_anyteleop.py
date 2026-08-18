import numpy as np

from paradex.retargetor.allegro_v5_anyteleop import (
    AllegroV5AnyTeleopRetargeter,
    AllegroV5TipKinematics,
    manus_wrist_relative_tips,
)
from paradex.retargetor.hand_regargetor import clip_allegro_v5_safe_action
from paradex.retargetor.unimanual import _resolve_hand
from src.dataset_acquisition.hri.allegro_retargeter_alignment_ui import (
    _make_retargeter,
    _retargeter_function_name,
)


def _frame(middle_tip_z=0.11):
    frame = {"wrist": np.eye(4)}
    for name, point in {
        "index_tip": (-0.03, 0.02, 0.10),
        "middle_tip": (0.00, 0.02, middle_tip_z),
        "ring_tip": (0.03, 0.02, 0.10),
        "thumb_tip": (-0.06, -0.03, 0.04),
    }.items():
        transform = np.eye(4)
        transform[:3, 3] = point
        frame[name] = transform
    return frame


def test_allegro_v5_tip_fk_exposes_four_finite_tip_positions():
    tips = AllegroV5TipKinematics().fingertip_positions(np.zeros(16))
    assert tips.shape == (4, 3)
    assert np.all(np.isfinite(tips))
    # The three non-thumb fingertips must be distinct in the Allegro palm.
    assert len({tuple(point) for point in tips[:3]}) == 3


def test_manus_tip_geometry_is_wrist_relative():
    frame = _frame()
    baseline = manus_wrist_relative_tips(frame)

    global_motion = np.eye(4)
    global_motion[:3, 3] = [4.0, -2.0, 1.0]
    moved = {name: global_motion @ transform for name, transform in frame.items()}

    np.testing.assert_allclose(manus_wrist_relative_tips(moved), baseline)


def test_geometric_retargeter_is_opt_in_and_responds_to_tip_geometry():
    # A fixed direct command lets this test isolate the geometric objective.
    retargeter = AllegroV5AnyTeleopRetargeter(
        direct_retargeter=lambda _frame: np.zeros(16),
        max_nfev=32,
    )
    first = retargeter(_frame())
    second = retargeter(_frame(middle_tip_z=0.05))
    lower = clip_allegro_v5_safe_action(np.full(16, -np.inf))
    upper = clip_allegro_v5_safe_action(np.full(16, np.inf))

    assert np.all(first >= lower)
    assert np.all(first <= upper)
    assert np.all(second >= lower)
    assert np.all(second <= upper)
    assert np.max(np.abs(second - first)) > 1.0e-3


def test_geometric_retargeter_uses_direct_path_for_incomplete_manus_frame():
    direct = np.linspace(-0.2, 0.8, 16)
    retargeter = AllegroV5AnyTeleopRetargeter(direct_retargeter=lambda _frame: direct)

    np.testing.assert_allclose(retargeter({"wrist": np.eye(4)}), clip_allegro_v5_safe_action(direct))


def test_anyteleop_selector_creates_a_separate_stateful_retargeter():
    selected = _resolve_hand("allegro_v5_anyteleop")
    assert isinstance(selected, AllegroV5AnyTeleopRetargeter)


def test_alignment_ui_exposes_anyteleop_without_replacing_the_v5_driver():
    selected = _make_retargeter("allegro_v5", "anyteleop")
    assert isinstance(selected, AllegroV5AnyTeleopRetargeter)
    assert _retargeter_function_name("allegro_v5", "anyteleop").endswith(
        "AllegroV5AnyTeleopRetargeter"
    )
