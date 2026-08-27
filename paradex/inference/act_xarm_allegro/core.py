"""Pure data contracts and safety logic for ACT real-robot inference.

This module intentionally has no ROS, LeRobot, or camera-driver imports.  It is
shared by the CUDA inference process, the ROS hardware bridge, and unit tests.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Mapping

import numpy as np


POLICY_REPO_ID = "hahahataeyun/hrdexdb-act-two-view-all-v1-act-100k"
DATASET_REPO_ID = "hahahataeyun/hrdexdb-act-two-view-all-v1"
STATE_KEY = "observation.state"
ACTION_KEY = "action"
STATE_DIM = 22
ACTION_DIM = 25
EXPECTED_STATE_AXES = (
    *(f"xarm_joint_{index}" for index in range(6)),
    *(f"allegro_state_{index}" for index in range(16)),
)
EXPECTED_ACTION_AXES = (
    "tcp_x",
    "tcp_y",
    "tcp_z",
    "rotation_6d_col0_x",
    "rotation_6d_col0_y",
    "rotation_6d_col0_z",
    "rotation_6d_col1_x",
    "rotation_6d_col1_y",
    "rotation_6d_col1_z",
    *(f"allegro_command_{index}" for index in range(16)),
)
ALLEGRO_PHYSICAL_LOWER = np.array(
    [-0.47, -0.196, -0.174, -0.227] * 3 + [0.0, -0.26, 0.0, -0.09],
    dtype=np.float64,
)
ALLEGRO_PHYSICAL_UPPER = np.array(
    [0.47, 1.8, 1.709, 1.618] * 3 + [1.78, 1.78, 1.9, 1.8],
    dtype=np.float64,
)


@dataclass(frozen=True)
class CameraBinding:
    policy_key: str
    physical_serial: str
    capture_pc: str


DEFAULT_CAMERA_BINDINGS = (
    CameraBinding(
        "observation.images.cam_23029839",
        "26256735",
        "capture16",
    ),
    CameraBinding(
        "observation.images.cam_25452066",
        "25452066",
        "capture18",
    ),
)


@dataclass(frozen=True)
class ObservationPacket:
    images: Mapping[str, np.ndarray]
    state: np.ndarray
    frame_ids: Mapping[str, int]
    captured_monotonic_ns: int
    state_monotonic_ns: int
    jpeg_bytes: Mapping[str, bytes] = field(default_factory=dict)
    raw_frame_ids: Mapping[str, int] = field(default_factory=dict)


@dataclass(frozen=True)
class DecodedAction:
    tcp_transform: np.ndarray
    allegro_target: np.ndarray
    raw: np.ndarray


@dataclass(frozen=True)
class SafetyVerdict:
    accepted: bool
    reason: str
    bounded_action: DecodedAction | None = None


@dataclass
class SafetyConfig:
    state_lower: np.ndarray
    state_upper: np.ndarray
    action_lower: np.ndarray
    action_upper: np.ndarray
    allegro_start_target: np.ndarray | None = None
    control_hz: float = 30.0
    # Same conservative live Cartesian bounds used by CaptureSession's Vive path.
    max_linear_speed_m_s: float = 0.70
    max_angular_speed_deg_s: float = 240.0
    position_margin_m: float = 0.003
    rotation_margin_deg: float = 1.5
    max_hand_speed_rad_s: float = 4.0
    max_observation_age_ms: float = 100.0
    max_state_age_ms: float = 100.0
    max_consecutive_faults: int = 3

    def __post_init__(self) -> None:
        self.state_lower = _finite_vector(self.state_lower, STATE_DIM, "state_lower")
        self.state_upper = _finite_vector(self.state_upper, STATE_DIM, "state_upper")
        self.action_lower = _finite_vector(self.action_lower, ACTION_DIM, "action_lower")
        self.action_upper = _finite_vector(self.action_upper, ACTION_DIM, "action_upper")
        if self.allegro_start_target is not None:
            self.allegro_start_target = _finite_vector(
                self.allegro_start_target, 16, "allegro_start_target"
            )
        if np.any(self.state_lower > self.state_upper):
            raise ValueError("state_lower must not exceed state_upper")
        if np.any(self.action_lower > self.action_upper):
            raise ValueError("action_lower must not exceed action_upper")
        if self.allegro_start_target is not None and (
            np.any(self.allegro_start_target < ALLEGRO_PHYSICAL_LOWER)
            or np.any(self.allegro_start_target > ALLEGRO_PHYSICAL_UPPER)
        ):
            raise ValueError("allegro_start_target violates physical bounds")
        for name in (
            "control_hz",
            "max_linear_speed_m_s",
            "max_angular_speed_deg_s",
            "max_hand_speed_rad_s",
            "max_observation_age_ms",
            "max_state_age_ms",
        ):
            value = float(getattr(self, name))
            if not np.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be positive and finite")
        if self.max_consecutive_faults <= 0:
            raise ValueError("max_consecutive_faults must be positive")


@dataclass
class RunnerConfig:
    mode: str = "shadow"
    policy_path: str = POLICY_REPO_ID
    policy_revision: str | None = None
    dataset_repo_id: str = DATASET_REPO_ID
    device: str = "cuda"
    control_hz: float = 30.0
    action_steps: int = 10
    duration_seconds: float | None = None
    max_chunks_per_enable: int = 0
    output_dir: Path = Path("~/shared_data/inference/act_xarm_allegro").expanduser()
    camera_bindings: tuple[CameraBinding, ...] = DEFAULT_CAMERA_BINDINGS
    state_endpoint: str = "tcp://127.0.0.1:5561"
    command_endpoint: str = "tcp://127.0.0.1:5562"
    enable_live: bool = False
    # This runner publishes the checkpoint output as-is.  Dataset support,
    # workspace and per-tick rate checks are retained only for offline analysis
    # through SafetyFilter; they are deliberately not live execution gates.
    enforce_safety_gates: bool = False
    preposition_allegro: bool = True
    preposition_timeout_seconds: float = 5.0
    # Allegro joint-state feedback has a repeatable controller/encoder offset
    # relative to a commanded pose. This is a convergence sanity bound, not an
    # exact setpoint requirement.
    preposition_tolerance_rad: float = 0.20

    def __post_init__(self) -> None:
        if self.mode not in {"contract", "replay", "shadow", "live"}:
            raise ValueError(f"Unsupported mode: {self.mode}")
        if self.mode == "live" and not self.enable_live:
            raise ValueError("live mode requires enable_live=True")
        if self.control_hz <= 0 or not np.isfinite(self.control_hz):
            raise ValueError("control_hz must be positive and finite")
        if self.action_steps <= 0:
            raise ValueError("action_steps must be positive")
        if self.max_chunks_per_enable < 0:
            raise ValueError("max_chunks_per_enable must be non-negative")
        if self.preposition_timeout_seconds <= 0 or not np.isfinite(
            self.preposition_timeout_seconds
        ):
            raise ValueError("preposition_timeout_seconds must be positive and finite")
        if self.preposition_tolerance_rad <= 0 or not np.isfinite(
            self.preposition_tolerance_rad
        ):
            raise ValueError("preposition_tolerance_rad must be positive and finite")


def _finite_vector(value: np.ndarray, size: int, name: str) -> np.ndarray:
    vector = np.asarray(value, dtype=np.float64).reshape(-1)
    if vector.shape != (size,) or not np.all(np.isfinite(vector)):
        raise ValueError(f"{name} must be a finite {size}-vector")
    return vector.copy()


def rotation_6d_to_matrix(rotation_6d: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """Project two predicted rotation columns onto SO(3)."""

    value = _finite_vector(rotation_6d, 6, "rotation_6d")
    first = value[:3]
    second = value[3:]
    first_norm = float(np.linalg.norm(first))
    if first_norm < eps:
        raise ValueError("rotation_6d first column is degenerate")
    col0 = first / first_norm
    second_orthogonal = second - np.dot(col0, second) * col0
    second_norm = float(np.linalg.norm(second_orthogonal))
    if second_norm < eps:
        raise ValueError("rotation_6d columns are collinear")
    col1 = second_orthogonal / second_norm
    col2 = np.cross(col0, col1)
    matrix = np.column_stack((col0, col1, col2))
    if not np.all(np.isfinite(matrix)) or np.linalg.det(matrix) < 1.0 - 1e-5:
        raise ValueError("rotation_6d projection did not produce a valid SO(3) matrix")
    return matrix


def decode_action(action: np.ndarray) -> DecodedAction:
    raw = _finite_vector(action, ACTION_DIM, "action")
    tcp_transform = np.eye(4, dtype=np.float64)
    tcp_transform[:3, 3] = raw[:3]
    tcp_transform[:3, :3] = rotation_6d_to_matrix(raw[3:9])
    return DecodedAction(
        tcp_transform=tcp_transform,
        allegro_target=raw[9:].copy(),
        raw=raw,
    )


def rotation_distance_deg(first: np.ndarray, second: np.ndarray) -> float:
    relative = np.asarray(first, dtype=np.float64).T @ np.asarray(second, dtype=np.float64)
    cosine = np.clip((np.trace(relative) - 1.0) * 0.5, -1.0, 1.0)
    return float(np.degrees(np.arccos(cosine)))


class SafetyFilter:
    """Reject unsafe commands; never silently clip a policy action."""

    def __init__(self, config: SafetyConfig):
        self.config = config

    def validate_start(self, state: np.ndarray, tcp_transform: np.ndarray) -> SafetyVerdict:
        try:
            state_vector = _finite_vector(state, STATE_DIM, "state")
            pose = np.asarray(tcp_transform, dtype=np.float64)
            if pose.shape != (4, 4) or not np.all(np.isfinite(pose)):
                return SafetyVerdict(False, "start_tcp_pose_invalid")
            rotation = pose[:3, :3]
            if not np.allclose(pose[3], [0.0, 0.0, 0.0, 1.0], atol=1e-6):
                return SafetyVerdict(False, "start_tcp_pose_invalid")
            if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-5) or not np.isclose(
                np.linalg.det(rotation), 1.0, atol=1e-5
            ):
                return SafetyVerdict(False, "start_tcp_rotation_invalid")
        except ValueError as exc:
            return SafetyVerdict(False, f"start_state_invalid:{exc}")
        return SafetyVerdict(True, "ok")

    def training_support_warning(self, state: np.ndarray, tcp_transform: np.ndarray) -> str | None:
        """Describe model-distribution extrapolation without blocking live arming."""

        state_vector = _finite_vector(state, STATE_DIM, "state")
        pose = np.asarray(tcp_transform, dtype=np.float64)
        warnings = []
        if np.any(state_vector < self.config.state_lower) or np.any(state_vector > self.config.state_upper):
            outside = np.flatnonzero(
                (state_vector < self.config.state_lower)
                | (state_vector > self.config.state_upper)
            )
            details = []
            for index in outside:
                value = state_vector[index]
                lower = self.config.state_lower[index]
                upper = self.config.state_upper[index]
                details.append(
                    f"{EXPECTED_STATE_AXES[index]}={value:.6f} "
                    f"outside [{lower:.6f}, {upper:.6f}]"
                )
            warnings.append("state_outside_training_support:" + "; ".join(details))
        xyz = pose[:3, 3]
        if np.any(xyz < self.config.action_lower[:3]) or np.any(
            xyz > self.config.action_upper[:3]
        ):
            warnings.append("tcp_outside_training_workspace")
        return "; ".join(warnings) if warnings else None

    def validate_action(
        self,
        decoded: DecodedAction,
        previous_tcp: np.ndarray,
        previous_hand: np.ndarray,
    ) -> SafetyVerdict:
        raw = decoded.raw
        if np.any(raw < self.config.action_lower) or np.any(raw > self.config.action_upper):
            return SafetyVerdict(False, "action_outside_dataset_or_physical_bounds")

        previous_tcp = np.asarray(previous_tcp, dtype=np.float64)
        previous_hand = np.asarray(previous_hand, dtype=np.float64).reshape(-1)
        if previous_tcp.shape != (4, 4) or previous_hand.shape != (16,):
            return SafetyVerdict(False, "previous_command_shape_invalid")
        if not np.all(np.isfinite(previous_tcp)) or not np.all(np.isfinite(previous_hand)):
            return SafetyVerdict(False, "previous_command_non_finite")

        position_delta = float(
            np.linalg.norm(decoded.tcp_transform[:3, 3] - previous_tcp[:3, 3])
        )
        max_position_delta = (
            self.config.max_linear_speed_m_s / self.config.control_hz
            + self.config.position_margin_m
        )
        if position_delta > max_position_delta:
            return SafetyVerdict(False, "tcp_translation_rate_exceeded")

        rotation_delta = rotation_distance_deg(
            previous_tcp[:3, :3], decoded.tcp_transform[:3, :3]
        )
        max_rotation_delta = (
            self.config.max_angular_speed_deg_s / self.config.control_hz
            + self.config.rotation_margin_deg
        )
        if rotation_delta > max_rotation_delta:
            return SafetyVerdict(False, "tcp_rotation_rate_exceeded")

        max_hand_delta = self.config.max_hand_speed_rad_s / self.config.control_hz
        if np.max(np.abs(decoded.allegro_target - previous_hand)) > max_hand_delta:
            return SafetyVerdict(False, "allegro_rate_exceeded")
        return SafetyVerdict(True, "ok", decoded)

    def validate_freshness(
        self,
        observation_monotonic_ns: int,
        state_monotonic_ns: int,
        now_monotonic_ns: int,
    ) -> SafetyVerdict:
        observation_age_ms = (now_monotonic_ns - observation_monotonic_ns) / 1e6
        state_age_ms = (now_monotonic_ns - state_monotonic_ns) / 1e6
        if observation_age_ms < 0 or observation_age_ms > self.config.max_observation_age_ms:
            return SafetyVerdict(False, "camera_observation_stale")
        if state_age_ms < 0 or state_age_ms > self.config.max_state_age_ms:
            return SafetyVerdict(False, "robot_state_stale")
        return SafetyVerdict(True, "ok")
