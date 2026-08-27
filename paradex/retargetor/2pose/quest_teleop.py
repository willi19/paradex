"""Coordinate helpers for right-controller Quest-to-xArm teleoperation."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.spatial.transform import Rotation


# Edit this matrix to tune the controller-axis -> xArm-axis mapping.
#
#     vector_xarm = CONTROLLER_TO_XARM_AXIS @ vector_controller
#
# Columns are controller axes [X(right), Y(up), Z(backward)].
# Rows are xArm axes       [X(forward), Y(left), Z(up)].
# The default mapping is controller forward/right/up -> xArm forward/right/up.
CONTROLLER_TO_XARM_AXIS = np.array(
    [
        # Controller:   X     Y     Z
        [0.0, 0.0, -1.0],  # -> xArm X
        [-1.0, 0.0, 0.0],  # -> xArm Y
        [0.0, 1.0, 0.0],  # -> xArm Z
    ],
    dtype=float,
)

# Backward-compatible name for existing imports/tests.
OPENXR_TO_XARM_ROTATION = CONTROLLER_TO_XARM_AXIS


@dataclass
class QuestTeleopGate:
    """Require fresh tracking and a new Grip press after tracking loss."""

    active: bool = False
    rearm_required: bool = False

    def update(
        self,
        *,
        pose_fresh: bool,
        grip_fresh: bool,
        grip_held: bool,
    ) -> str:
        if not pose_fresh:
            self.active = False
            self.rearm_required = True
            return "pose_stale"
        if not grip_fresh:
            self.active = False
            self.rearm_required = True
            return "grip_stale"
        if self.rearm_required:
            self.active = False
            if grip_held:
                return "release_to_rearm"
            self.rearm_required = False
            return "grip_released"
        if not grip_held:
            self.active = False
            return "grip_released"
        if not self.active:
            self.active = True
            return "start"
        return "active"

    def retry_activation(self) -> None:
        """Allow a failed arm-pose read to be retried on the next cycle."""
        self.active = False


def quest_grip_deadman_pressed(
    *,
    grip: float,
    updated_at: float | None,
    now: float,
    threshold: float,
    max_age: float,
) -> bool:
    """Return whether a fresh Quest Grip sample enables teleoperation."""
    if updated_at is None:
        return False
    values = (grip, updated_at, now, threshold, max_age)
    if not all(np.isfinite(value) for value in values):
        return False
    return 0.0 <= now - updated_at <= max_age and grip >= threshold


class QuestGripTeleopStateAdapter:
    """Delegate a teleop device while sourcing state 0/1 from Quest Grip."""

    def __init__(self, device, state_provider):
        if device is None:
            raise ValueError("Quest Grip state adapter requires a teleop device")
        if not callable(state_provider):
            raise ValueError("Quest Grip state provider must be callable")
        self._device = device
        self._state_provider = state_provider

    def get_state(self) -> int:
        state = int(self._state_provider())
        if state not in (0, 1):
            raise ValueError(f"Quest Grip state provider returned invalid state: {state}")
        return state

    def __getattr__(self, name):
        return getattr(self._device, name)


def openxr_pose_matrix(position, quaternion_xyzw) -> np.ndarray:
    """Convert an OpenXR position/quaternion pair to a homogeneous pose."""
    position = np.asarray(position, dtype=float).reshape(-1)
    quaternion = np.asarray(quaternion_xyzw, dtype=float).reshape(-1)
    if (
        position.shape != (3,)
        or quaternion.shape != (4,)
        or not np.all(np.isfinite(position))
        or not np.all(np.isfinite(quaternion))
        or np.linalg.norm(quaternion) < 1e-8
    ):
        raise ValueError(
            "Quest pose must contain finite position and quaternion values"
        )
    pose = np.eye(4, dtype=float)
    pose[:3, :3] = Rotation.from_quat(quaternion).as_matrix()
    pose[:3, 3] = position
    return pose


def quest_delta_to_xarm_target(
    initial_controller_pose,
    controller_pose,
    initial_xarm_pose,
    *,
    translation_scale: float = 1.0,
) -> np.ndarray:
    """Apply a controller-relative OpenXR delta to an initial xArm pose."""
    initial_controller = np.asarray(initial_controller_pose, dtype=float)
    controller = np.asarray(controller_pose, dtype=float)
    initial_xarm = np.asarray(initial_xarm_pose, dtype=float)
    if any(
        pose.shape != (4, 4) for pose in (initial_controller, controller, initial_xarm)
    ):
        raise ValueError("Quest and xArm poses must be 4x4 transforms")
    if not all(
        np.all(np.isfinite(pose))
        for pose in (initial_controller, controller, initial_xarm)
    ):
        raise ValueError("Quest and xArm poses must be finite")
    translation_scale = float(translation_scale)
    if not np.isfinite(translation_scale) or translation_scale <= 0.0:
        raise ValueError("translation_scale must be a positive finite value")

    basis = CONTROLLER_TO_XARM_AXIS
    controller_delta_rotation = initial_controller[:3, :3].T @ controller[:3, :3]
    controller_delta_translation = controller[:3, 3] - initial_controller[:3, 3]

    target = initial_xarm.copy()
    target[:3, :3] = initial_xarm[:3, :3] @ basis @ controller_delta_rotation @ basis.T
    target[:3, 3] = initial_xarm[:3, 3] + translation_scale * (
        basis @ controller_delta_translation
    )
    return target
