"""Named Allegro joint contracts for retargeter-alignment samples.

The controller reports the 16 joints in ``ah_jointXY`` order, while the
combined xArm + Allegro URDF uses semantic finger names.  The retargeter
function returns another explicit order (index, middle, ring, thumb).  This
module records and converts each ordering instead of treating any of them as
an unnamed 16-value vector.
"""

from __future__ import annotations

from typing import Mapping, Sequence

import numpy as np

from paradex.retargetor.hand_alignment_common import HandStateContractError


ALLEGRO_FEEDBACK_JOINT_NAMES = tuple(
    f"ah_joint{finger}{joint}" for finger in range(4) for joint in range(4)
)
# Allegro v5 publishes driver slots 0..15 as index, middle, ring, thumb.
# Reorder the *names* into URDF semantic order (thumb, index, middle, ring),
# while leaving the qpos vector paired to its reported driver names.
# ``capture_robot.py --hand allegro_v5 --hand-side right`` uses this contract.
ALLEGRO_V5_DRIVER_JOINT_NAMES = tuple(f"joint_{index}_0" for index in range(16))
ALLEGRO_V5_FEEDBACK_JOINT_NAMES = (
    *ALLEGRO_V5_DRIVER_JOINT_NAMES[12:16],
    *ALLEGRO_V5_DRIVER_JOINT_NAMES[0:12],
)
ALLEGRO_URDF_JOINT_NAMES = (
    "thumb_base", "thumb_proximal", "thumb_medial", "thumb_distal",
    "index_base", "index_proximal", "index_medial", "index_distal",
    "middle_base", "middle_proximal", "middle_medial", "middle_distal",
    "ring_base", "ring_proximal", "ring_medial", "ring_distal",
)
# ``hand_regargetor.allegro()`` fills its 16-value return vector in this order.
ALLEGRO_RETARGETER_JOINT_NAMES = (
    "index_base", "index_proximal", "index_medial", "index_distal",
    "middle_base", "middle_proximal", "middle_medial", "middle_distal",
    "ring_base", "ring_proximal", "ring_medial", "ring_distal",
    "thumb_base", "thumb_proximal", "thumb_medial", "thumb_distal",
)


def _named_values(values: Sequence[float], names: Sequence[str]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64).reshape(-1)
    names = tuple(str(name) for name in names)
    if array.shape != (len(names),):
        raise HandStateContractError(
            f"values shape {array.shape} does not match {len(names)} joint names"
        )
    if len(set(names)) != len(names):
        raise HandStateContractError("joint names must be unique")
    if not np.all(np.isfinite(array)):
        raise HandStateContractError("Allegro joint state contains non-finite values")
    return dict(zip(names, array.tolist()))


def feedback_to_urdf_qpos(
    values: Sequence[float],
    joint_names: Sequence[str] = ALLEGRO_FEEDBACK_JOINT_NAMES,
) -> dict[str, float]:
    """Map named controller-radian feedback to semantic Allegro URDF joints."""
    by_name = _named_values(values, joint_names)
    feedback_order = next(
        (
            candidate
            for candidate in (
                ALLEGRO_FEEDBACK_JOINT_NAMES,
                ALLEGRO_V5_FEEDBACK_JOINT_NAMES,
            )
            if all(name in by_name for name in candidate)
        ),
        None,
    )
    if feedback_order is None:
        expected = ", ".join(ALLEGRO_FEEDBACK_JOINT_NAMES)
        v5_expected = ", ".join(ALLEGRO_V5_FEEDBACK_JOINT_NAMES)
        raise HandStateContractError(
            "Allegro feedback does not match a supported joint-name contract. "
            f"Expected legacy [{expected}] or v5 [{v5_expected}]."
        )
    return {
        urdf_name: float(by_name[feedback_name])
        for feedback_name, urdf_name in zip(
            feedback_order, ALLEGRO_URDF_JOINT_NAMES
        )
    }


def urdf_qpos_from_hand_qpos(
    hand_qpos: Mapping[str, float], urdf_joint_names: Sequence[str]
) -> np.ndarray:
    """Build a full combined-URDF configuration with the arm fixed at zero."""
    result = np.zeros(len(urdf_joint_names), dtype=np.float64)
    index = {str(name): i for i, name in enumerate(urdf_joint_names)}
    missing_urdf = [name for name in ALLEGRO_URDF_JOINT_NAMES if name not in index]
    missing_qpos = [name for name in ALLEGRO_URDF_JOINT_NAMES if name not in hand_qpos]
    if missing_urdf:
        raise HandStateContractError(
            "URDF does not expose expected Allegro joints: " + ", ".join(missing_urdf)
        )
    if missing_qpos:
        raise HandStateContractError(
            "Allegro qpos is missing joints: " + ", ".join(missing_qpos)
        )
    for name in ALLEGRO_URDF_JOINT_NAMES:
        value = float(hand_qpos[name])
        if not np.isfinite(value):
            raise HandStateContractError(f"non-finite Allegro qpos for {name}")
        result[index[name]] = value
    return result


def urdf_qpos_to_retargeter_action(hand_qpos: Mapping[str, float]) -> list[float]:
    """Return the named-order 16-vector accepted by ``hand_regargetor.allegro``."""
    missing = [name for name in ALLEGRO_RETARGETER_JOINT_NAMES if name not in hand_qpos]
    if missing:
        raise HandStateContractError(
            "Allegro qpos is missing retargeter joints: " + ", ".join(missing)
        )
    return [float(hand_qpos[name]) for name in ALLEGRO_RETARGETER_JOINT_NAMES]


def retargeter_action_to_urdf_qpos(action: Sequence[float]) -> dict[str, float]:
    """Map index/middle/ring/thumb action values to semantic URDF names."""
    by_name = _named_values(action, ALLEGRO_RETARGETER_JOINT_NAMES)
    return {name: float(by_name[name]) for name in ALLEGRO_URDF_JOINT_NAMES}


def retargeter_action_to_controller_qpos(action: Sequence[float]) -> list[float]:
    """Convert ``hand_regargetor.allegro`` output to controller logical order."""
    return list(retargeter_action_to_urdf_qpos(action).values())


def retargeter_action_to_live_controller_qpos(
    action: Sequence[float], hand_name: str
) -> list[float]:
    """Apply the same action ordering as ``CaptureSession`` for each driver.

    The v5 CaptureSession calls ``allegro_v5(frame)`` and passes the resulting
    index/middle/ring/thumb vector directly to ``allegro_v5_controller_ros2``.
    The older driver uses its named logical controller order instead.
    """
    if hand_name in (
        "allegro_v5",
        "allegro_v5_anyteleop",
        "allegro_v5_wonik",
    ):
        return list(_named_values(action, ALLEGRO_RETARGETER_JOINT_NAMES).values())
    if hand_name == "allegro":
        return retargeter_action_to_controller_qpos(action)
    raise ValueError(f"Unsupported Allegro hand driver: {hand_name}")
