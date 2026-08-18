"""Wonik Robotics' rule-based MANUS-to-Allegro retargeter for Allegro V5.

This is a small adaptation of
https://github.com/Wonikrobotics-git/allegro_hand_teleoperation/
``glove_based/rule_based_retargeting.py`` (MIT License, Copyright 2025
Wonik Robotics).  It deliberately preserves the source mapping constants and
EMA, while adapting its V4 ``thumb, index, middle, ring`` output order to this
repository's V5 driver order: ``index, middle, ring, thumb``.
"""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np

from paradex.retargetor.hand_regargetor import clip_allegro_v5_safe_action


# Source-code V4 limits, in Wonik's ``thumb, index, middle, ring`` order.
_WONIK_UPPER = np.array(
    [1.396, 1.163, 1.644, 1.719] + [0.47, 1.61, 1.709, 1.618] * 3,
    dtype=np.float64,
)
_WONIK_LOWER = np.array(
    [0.7, 0.3, -0.189, -0.162] + [-0.47, -0.196, -0.174, -0.227] * 3,
    dtype=np.float64,
)

# The official implementation reads this sequence directly from
# ``msg.ergonomics``.  Use field names so input remains correct when a ROS2
# publisher changes the list ordering.
_ERGONOMIC_FIELDS = {
    "thumb": (
        "ThumbMCPStretch",
        "ThumbMCPSpread",
        "ThumbPIPStretch",
        "ThumbDIPStretch",
    ),
    "index": (
        "IndexSpread",
        "IndexMCPStretch",
        "IndexPIPStretch",
        "IndexDIPStretch",
    ),
    "middle": (
        "MiddleSpread",
        "MiddleMCPStretch",
        "MiddlePIPStretch",
        "MiddleDIPStretch",
    ),
    "ring": (
        "RingSpread",
        "RingMCPStretch",
        "RingPIPStretch",
        "RingDIPStretch",
    ),
}


def _ergonomic_vector(ergonomics: Mapping[str, float]) -> tuple[np.ndarray, ...]:
    """Extract the source retargeter's four named MANUS value blocks."""
    values = []
    for finger in ("thumb", "index", "middle", "ring"):
        fields = _ERGONOMIC_FIELDS[finger]
        missing = [field for field in fields if field not in ergonomics]
        if missing:
            raise ValueError(
                "MANUS ergonomics is missing Wonik Allegro inputs: "
                + ", ".join(missing)
            )
        block = np.asarray([ergonomics[field] for field in fields], dtype=np.float64)
        if not np.all(np.isfinite(block)):
            raise ValueError(
                f"MANUS ergonomics for {finger} contains non-finite values"
            )
        values.append(block)
    return tuple(values)


class AllegroV5WonikManusRetargeter:
    """Stateful adaptation of Wonik's Manus glove rule-based retargeter."""

    def __init__(self, ema_alpha: float = 0.2):
        if not np.isfinite(ema_alpha) or not 0.0 < ema_alpha <= 1.0:
            raise ValueError("ema_alpha must be finite and in (0, 1]")
        self.ema_alpha = float(ema_alpha)
        self._previous_v4_action: np.ndarray | None = None

    def reset(self) -> None:
        """Clear temporal EMA history at the start of a teleoperation run."""
        self._previous_v4_action = None

    def __call__(self, _frame, *, ergonomics: Mapping[str, float] | None = None):
        """Map named MANUS ergonomics to a safe V5 driver-order action."""
        if ergonomics is None:
            raise ValueError("Wonik Allegro retargeting requires MANUS ergonomics")
        thumb, index, middle, ring = _ergonomic_vector(ergonomics)

        # Exact rule-based construction from Wonik's public MIT source.
        angle_deg = np.concatenate(
            (
                [
                    90.0 - 1.75 * thumb[1],
                    -45.0 + 3.0 * thumb[0],
                    -30.0 + 3.0 * thumb[2],
                    thumb[3],
                ],
                index,
                [middle[0] + 20.0],
                middle[1:],
                ring[:3],
                [ring[3] + 5.0],
            )
        )
        action_v4 = np.deg2rad(angle_deg)
        action_v4[0] *= 2.5
        action_v4[1] = action_v4[1] * 2.0 + np.deg2rad(90.0)
        action_v4[3] *= 2.0
        action_v4[4] *= -0.5
        action_v4[5] *= 1.5
        action_v4[7] *= 2.0
        action_v4[8] *= -0.2
        action_v4[9] *= 1.5
        action_v4[11] *= 2.0
        action_v4[12] *= 0.1
        action_v4[13] *= 1.5
        action_v4[15] *= 2.0
        action_v4 = np.clip(action_v4, _WONIK_LOWER, _WONIK_UPPER)

        if self._previous_v4_action is not None:
            action_v4 = (
                self.ema_alpha * action_v4
                + (1.0 - self.ema_alpha) * self._previous_v4_action
            )
        self._previous_v4_action = action_v4.copy()

        # Wonik source: thumb, index, middle, ring.  This repo's V5 ROS
        # controller: index, middle, ring, thumb.
        action_v5 = np.concatenate(
            (action_v4[4:8], action_v4[8:12], action_v4[12:16], action_v4[:4])
        )
        return clip_allegro_v5_safe_action(action_v5)
