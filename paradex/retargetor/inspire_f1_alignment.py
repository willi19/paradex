"""Stable data contract for Inspire F1 retargeter-alignment captures.

The live hand driver exposes six raw motor values.  This module gives those
values an explicit joint-name contract and converts them to the six actuated
hand joints in ``xarm_inspire_f1_right.urdf``.  Keeping this conversion here
prevents UI tools and later fitting scripts from silently relying on array
position alone.
"""

from __future__ import annotations

from typing import Mapping, Sequence

import numpy as np

from paradex.retargetor.hand_alignment_common import (
    HandStateContractError,
    read_json,
    serialize_manus_frame,
    sha256_file,
    write_json,
)


F1_RIGHT_WIRE_JOINT_NAMES = (
    "right_thumb_1_joint",
    "right_thumb_2_joint",
    "right_index_1_joint",
    "right_middle_1_joint",
    "right_ring_1_joint",
    "right_little_1_joint",
)
# ``hand_regargetor.inspire_f1()`` returns this legacy logical order.
F1_RETARGETER_ACTION_JOINT_NAMES = (
    "right_little_1_joint",
    "right_ring_1_joint",
    "right_middle_1_joint",
    "right_index_1_joint",
    "right_thumb_1_joint",
    "right_thumb_2_joint",
)

# Hardware-side raw motor endpoints and their matching URDF joint ranges.
# These are shared with h2r/capture_object6d.py, where the same six-channel
# /right/joint_states contract is documented.
F1_RIGHT_RAW_OPEN = {
    "right_thumb_1_joint": 1800.0,
    "right_thumb_2_joint": 1350.0,
    "right_index_1_joint": 1740.0,
    "right_middle_1_joint": 1740.0,
    "right_ring_1_joint": 1740.0,
    "right_little_1_joint": 1740.0,
}
F1_RIGHT_RAW_CLOSE = {
    "right_thumb_1_joint": 600.0,
    "right_thumb_2_joint": 1100.0,
    "right_index_1_joint": 900.0,
    "right_middle_1_joint": 900.0,
    "right_ring_1_joint": 900.0,
    "right_little_1_joint": 900.0,
}
F1_RIGHT_QPOS_UPPER = {
    "right_thumb_1_joint": 2.0944,
    "right_thumb_2_joint": 0.4746,
    "right_index_1_joint": 1.5286,
    "right_middle_1_joint": 1.5286,
    "right_ring_1_joint": 1.5286,
    "right_little_1_joint": 1.5286,
}


def _as_named_vector(values: Sequence[float], names: Sequence[str]) -> dict[str, float]:
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    names = tuple(str(name) for name in names)
    if values.shape != (len(names),):
        raise HandStateContractError(
            f"values shape {values.shape} does not match {len(names)} joint names"
        )
    if len(set(names)) != len(names):
        raise HandStateContractError("joint names must be unique")
    missing = [name for name in F1_RIGHT_WIRE_JOINT_NAMES if name not in names]
    if missing:
        raise HandStateContractError(
            "F1 state is missing wire joints: " + ", ".join(missing)
        )
    if not np.all(np.isfinite(values)):
        raise HandStateContractError("F1 state contains non-finite values")
    return dict(zip(names, values.tolist()))


def raw_f1_to_master_qpos(
    raw_values: Sequence[float],
    joint_names: Sequence[str] = F1_RIGHT_WIRE_JOINT_NAMES,
    *,
    clip: bool = True,
) -> dict[str, float]:
    """Convert named F1 raw motor values to named URDF master joint radians."""
    raw_by_name = _as_named_vector(raw_values, joint_names)
    result = {}
    for name in F1_RIGHT_WIRE_JOINT_NAMES:
        open_value = F1_RIGHT_RAW_OPEN[name]
        close_value = F1_RIGHT_RAW_CLOSE[name]
        fraction_closed = (open_value - raw_by_name[name]) / (open_value - close_value)
        if clip:
            fraction_closed = float(np.clip(fraction_closed, 0.0, 1.0))
        result[name] = float(fraction_closed * F1_RIGHT_QPOS_UPPER[name])
    return result


def master_qpos_to_raw_f1(master_qpos: Mapping[str, float]) -> dict[str, float]:
    """Convert named URDF master radians back to named F1 raw motor values."""
    result = {}
    for name in F1_RIGHT_WIRE_JOINT_NAMES:
        if name not in master_qpos:
            raise HandStateContractError(f"missing master qpos for {name}")
        value = float(master_qpos[name])
        if not np.isfinite(value):
            raise HandStateContractError(f"non-finite master qpos for {name}")
        fraction_closed = float(np.clip(value / F1_RIGHT_QPOS_UPPER[name], 0.0, 1.0))
        result[name] = float(
            F1_RIGHT_RAW_OPEN[name]
            + (F1_RIGHT_RAW_CLOSE[name] - F1_RIGHT_RAW_OPEN[name]) * fraction_closed
        )
    return result


def retargeter_action_to_raw_f1(action: Sequence[float]) -> dict[str, float]:
    """Name the six-value output of ``hand_regargetor.inspire_f1``.

    Returning a mapping lets command publishers send values in their explicit
    wire order rather than relying on the legacy return-vector positions.
    """
    values = _as_named_vector(action, F1_RETARGETER_ACTION_JOINT_NAMES)
    return {name: values[name] for name in F1_RIGHT_WIRE_JOINT_NAMES}


def urdf_qpos_from_master_qpos(
    master_qpos: Mapping[str, float], urdf_joint_names: Sequence[str]
) -> np.ndarray:
    """Return a full URDF vector with arm joints at zero and F1 masters set."""
    result = np.zeros(len(urdf_joint_names), dtype=np.float64)
    index = {str(name): i for i, name in enumerate(urdf_joint_names)}
    missing = [name for name in F1_RIGHT_WIRE_JOINT_NAMES if name not in index]
    if missing:
        raise HandStateContractError(
            "URDF does not expose expected F1 joints: " + ", ".join(missing)
        )
    for name in F1_RIGHT_WIRE_JOINT_NAMES:
        result[index[name]] = float(master_qpos[name])
    return result
