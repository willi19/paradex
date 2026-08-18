import numpy as np
import pytest

from paradex.retargetor.allegro_v5_wonik import AllegroV5WonikManusRetargeter
from paradex.retargetor.allegro_alignment import retargeter_action_to_live_controller_qpos
from paradex.retargetor.unimanual import _resolve_hand


def _ergonomics():
    return {
        "ThumbMCPStretch": 50.91769027709961,
        "ThumbMCPSpread": 2.6855485439300537,
        "ThumbPIPStretch": 3.6038050651550293,
        "ThumbDIPStretch": -10.924560546875,
        "IndexSpread": -5.5026726722717285,
        "IndexMCPStretch": -1.117326021194458,
        "IndexPIPStretch": 0.0,
        "IndexDIPStretch": 0.0,
        "MiddleSpread": -9.212068557739258,
        "MiddleMCPStretch": 7.294604778289795,
        "MiddlePIPStretch": 0.0,
        "MiddleDIPStretch": 0.0,
        "RingSpread": -13.066163063049316,
        "RingMCPStretch": 10.886436462402344,
        "RingPIPStretch": 0.0,
        "RingDIPStretch": 0.0,
    }


def test_wonik_retargeter_uses_named_manus_ergonomics_and_v5_driver_order():
    ergonomics = _ergonomics()
    retargeter = AllegroV5WonikManusRetargeter(ema_alpha=1.0)
    action = retargeter({}, ergonomics=ergonomics)

    # With alpha=1, this is the source mapping's first frame after its V4
    # limits, reordered to this repo's index/middle/ring/thumb driver order.
    np.testing.assert_allclose(
        action,
        np.array(
            [
                0.0,
                0.0,
                0.0,
                0.0,
                -0.037656977482,
                0.190972309324,
                0.0,
                0.0,
                -0.022804756028,
                0.285006236202,
                0.0,
                0.174532925199,
                1.396,
                1.163,
                0.0,
                -0.007,
            ]
        ),
        atol=1e-8,
    )


def test_wonik_retargeter_requires_complete_named_ergonomics():
    ergonomics = _ergonomics()
    del ergonomics["RingDIPStretch"]
    with pytest.raises(ValueError, match="RingDIPStretch"):
        AllegroV5WonikManusRetargeter()({}, ergonomics=ergonomics)


def test_wonik_retargeter_is_selectable_from_capture_pipeline():
    assert isinstance(_resolve_hand("allegro_v5_wonik"), AllegroV5WonikManusRetargeter)


def test_wonik_action_uses_the_v5_controller_driver_order():
    action = np.linspace(-0.1, 0.5, 16)
    assert retargeter_action_to_live_controller_qpos(action, "allegro_v5_wonik") == list(action)
