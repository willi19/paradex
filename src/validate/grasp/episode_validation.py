"""Read-only validation of robot/human grasp episodes in the ECCV 2026 layout.

The module deliberately has no output-path API.  Dataset files are opened only
for reading and reports are returned in memory, so callers cannot accidentally
write validation artifacts into the source dataset.
"""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import trimesh
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation

from paradex.visualization.robot import RobotModule


DEFAULT_MESH_ROOT = Path("/home/temp_id/shared_data/mesh_new")
DEFAULT_CAPTURE_ROOT = Path("/home/temp_id/shared_data/capture/eccv2026/v0")
DEFAULT_ROBOT_URDF = (
    Path(__file__).resolve().parents[3] / "rsc" / "robot" / "xarm_allegro_v5.urdf"
)
_ROBOT_RESOURCE_ROOT = Path(__file__).resolve().parents[3] / "rsc" / "robot"

_OBJECT_NAME_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]*")
_POSE_KEY_RE = re.compile(r"frame_(\d+)")
_MANO_TIP_VERTICES = {
    "thumb": 744,
    "index": 320,
    "middle": 443,
    "ring": 554,
    "pinky": 672,
}
_ROBOT_FINGER_LINKS = {
    "index": ("link_3_0", "link_3_0_tip"),
    "middle": ("link_7_0", "link_7_0_tip"),
    "ring": ("link_11_0", "link_11_0_tip"),
    "thumb": ("link_15_0", "link_15_0_tip"),
}


class DatasetFormatError(ValueError):
    """Raised when an episode exists but violates the expected input contract."""


class CaptureKind(str, Enum):
    """Supported hand-trajectory sources."""

    ROBOT = "robot"
    HUMAN = "human"


class ContactLossKind(str, Enum):
    """Classification after an established grasp loses persistent contact."""

    NORMAL_MOTION = "normal_motion"
    TRACKING_ERROR = "tracking_error"
    INSUFFICIENT_OBSERVATION = "insufficient_observation"


class HandType(str, Enum):
    """User-facing hand/model selector."""

    HUMAN = "human"
    ALLEGRO = "allegro"
    INSPIRE = "inspire"
    INSPIRE_F1 = "inspire_f1"


_HAND_ALIASES = {
    "human": HandType.HUMAN,
    "allegro": HandType.ALLEGRO,
    "allegro_v5": HandType.ALLEGRO,
    "inspire": HandType.INSPIRE,
    "inspire_dftp": HandType.INSPIRE,
    "inspire_f1": HandType.INSPIRE_F1,
}


@dataclass(frozen=True)
class _RobotHandSpec:
    capture_directory: str
    urdf: Path
    position_file: str
    time_file: str
    reference_link: str
    finger_links: Mapping[str, tuple[str, ...]]


_ROBOT_HAND_SPECS = {
    HandType.ALLEGRO: _RobotHandSpec(
        capture_directory="allegro_v5",
        urdf=DEFAULT_ROBOT_URDF,
        position_file="position.npy",
        time_file="time.npy",
        reference_link="palm_link",
        finger_links=_ROBOT_FINGER_LINKS,
    ),
    HandType.INSPIRE: _RobotHandSpec(
        capture_directory="inspire_dftp",
        urdf=_ROBOT_RESOURCE_ROOT / "xarm_inspire_DFTP.urdf",
        position_file="position.npy",
        time_file="time.npy",
        reference_link="base_link",
        finger_links={
            "thumb": ("right_thumb_4",),
            "index": ("right_index_2",),
            "middle": ("right_middle_2",),
            "ring": ("right_ring_2",),
            "pinky": ("right_little_2",),
        },
    ),
    HandType.INSPIRE_F1: _RobotHandSpec(
        capture_directory="inspire_f1",
        urdf=_ROBOT_RESOURCE_ROOT / "xarm_inspire_f1_right.urdf",
        position_file="right_joint_states.npy",
        time_file="right_joint_states_time.npy",
        reference_link="base_link",
        finger_links={
            "thumb": ("thumb_force_sensor", "thumb_tip"),
            "index": ("index_force_sensor", "index_tip"),
            "middle": ("middle_force_sensor", "middle_tip"),
            "ring": ("ring_force_sensor", "ring_tip"),
            "pinky": ("little_force_sensor", "little_tip"),
        },
    ),
}


@dataclass(frozen=True)
class EpisodePaths:
    """Resolved, read-only inputs for one capture episode."""

    kind: CaptureKind
    hand: HandType
    object_name: str
    episode: int
    episode_root: Path
    object_mesh: Path
    object_poses: Path
    hand_trajectory: Path
    arm_trajectory: Path | None = None
    camera_timestamps: Path | None = None
    camera_to_robot: Path | None = None

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["kind"] = self.kind.value
        result["hand"] = self.hand.value
        for key, value in tuple(result.items()):
            if isinstance(value, Path):
                result[key] = str(value)
        return result


@dataclass(frozen=True)
class ValidationThresholds:
    """Numerical criteria, in SI units unless a name says otherwise."""

    max_frames: int = 120
    contact_distance_m: float = 0.01
    min_contact_fingers: int = 2
    min_grasp_frames: int = 10
    max_contact_gap_samples: int = 1
    min_contact_loss_samples: int = 2
    min_object_motion_m: float = 0.01
    min_gravity_observation_s: float = 0.1
    min_gravity_displacement_m: float = 0.005
    min_gravity_velocity_change_m_s: float = 0.05
    joint_limit_tolerance_rad: float = 1e-3
    joint_velocity_scale: float = 1.2
    surface_sample_count: int = 6_000

    def __post_init__(self) -> None:
        positive = {
            "max_frames": self.max_frames,
            "contact_distance_m": self.contact_distance_m,
            "min_contact_fingers": self.min_contact_fingers,
            "min_grasp_frames": self.min_grasp_frames,
            "min_contact_loss_samples": self.min_contact_loss_samples,
            "min_gravity_observation_s": self.min_gravity_observation_s,
            "joint_velocity_scale": self.joint_velocity_scale,
            "surface_sample_count": self.surface_sample_count,
        }
        bad = [name for name, value in positive.items() if value <= 0]
        if bad:
            raise ValueError(f"thresholds must be positive: {', '.join(bad)}")
        if self.max_contact_gap_samples < 0:
            raise ValueError("max_contact_gap_samples must be non-negative")
        non_negative = {
            "min_object_motion_m": self.min_object_motion_m,
            "min_gravity_displacement_m": self.min_gravity_displacement_m,
            "min_gravity_velocity_change_m_s": (
                self.min_gravity_velocity_change_m_s
            ),
        }
        bad_non_negative = [
            name for name, value in non_negative.items() if value < 0
        ]
        if bad_non_negative:
            raise ValueError("distance thresholds must be non-negative")


@dataclass(frozen=True)
class ValidationIssue:
    code: str
    message: str
    frame: int | None = None
    value: float | None = None
    threshold: float | None = None


@dataclass(frozen=True)
class ContactPhase:
    start_frame: int
    end_frame: int
    start_sample: int
    end_sample: int
    sampled_frame_count: int


@dataclass(frozen=True)
class ContactLossEvent:
    """Persistent contact loss classified by gravity-consistent object motion."""

    kind: ContactLossKind
    frame: int
    last_contact_frame: int
    no_contact_sample_count: int
    gravity_direction: tuple[float, float, float]
    observation_duration_s: float
    gravity_displacement_m: float
    initial_gravity_velocity_m_s: float
    final_gravity_velocity_m_s: float
    gravity_velocity_change_m_s: float

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["kind"] = self.kind.value
        return result


@dataclass
class EpisodeValidationReport:
    """Machine-readable result for one robot or human episode."""

    kind: CaptureKind
    hand: HandType
    object_name: str
    episode: int
    valid: bool
    inputs: EpisodePaths
    frame_count: int
    sampled_frame_count: int
    contact_phase: ContactPhase | None
    contact_loss_event: ContactLossEvent | None = None
    metrics: dict[str, float | int | list[str] | None] = field(default_factory=dict)
    issues: list[ValidationIssue] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind.value,
            "hand": self.hand.value,
            "object": self.object_name,
            "episode": self.episode,
            "valid": self.valid,
            "inputs": self.inputs.to_dict(),
            "frame_count": self.frame_count,
            "sampled_frame_count": self.sampled_frame_count,
            "contact_phase": (
                asdict(self.contact_phase) if self.contact_phase is not None else None
            ),
            "contact_loss_event": (
                self.contact_loss_event.to_dict()
                if self.contact_loss_event is not None
                else None
            ),
            "metrics": self.metrics,
            "issues": [asdict(issue) for issue in self.issues],
            "notes": self.notes,
        }


@dataclass(frozen=True)
class _FrameState:
    frame: int
    time_s: float
    contact_fingers: tuple[str, ...]
    min_hand_object_distance_m: float
    world_from_reference: np.ndarray
    world_from_object: np.ndarray


class _ObjectSurface:
    """Deterministic approximate surface-distance query in object coordinates."""

    def __init__(self, mesh: trimesh.Trimesh, sample_count: int):
        points = np.concatenate(
            (
                _evenly_spaced_rows(np.asarray(mesh.vertices), sample_count),
                _evenly_spaced_rows(np.asarray(mesh.triangles_center), sample_count),
            ),
            axis=0,
        )
        if len(points) == 0 or not np.isfinite(points).all():
            raise DatasetFormatError("object mesh has no finite surface points")
        self._tree = cKDTree(points)

    def distance(
        self, world_points: np.ndarray, world_from_object: np.ndarray
    ) -> np.ndarray:
        object_from_world = np.linalg.inv(world_from_object)
        local_points = trimesh.transform_points(world_points, object_from_world)
        distances, _ = self._tree.query(local_points, workers=-1)
        return np.asarray(distances, dtype=float)


def _safe_child(root: Path, *parts: str) -> Path:
    root = root.expanduser().resolve()
    result = root.joinpath(*parts).resolve()
    try:
        result.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"path escapes configured dataset root: {result}") from exc
    return result


def _validate_object_and_episode(object_name: str, episode: int) -> None:
    if _OBJECT_NAME_RE.fullmatch(object_name) is None or object_name in {".", ".."}:
        raise ValueError(
            "object name must be one directory component containing only "
            "letters, digits, '.', '_' or '-'"
        )
    if isinstance(episode, bool) or int(episode) != episode or episode < 0:
        raise ValueError("episode must be a non-negative integer")


def discover_episode_paths(
    object_name: str,
    episode: int,
    hand: str,
    *,
    mesh_root: Path = DEFAULT_MESH_ROOT,
    capture_root: Path = DEFAULT_CAPTURE_ROOT,
) -> EpisodePaths:
    """Resolve one explicitly selected hand source without creating any paths."""

    _validate_object_and_episode(object_name, episode)
    normalized_hand = _HAND_ALIASES.get(hand)
    if normalized_hand is None:
        choices = ", ".join(item.value for item in HandType)
        raise ValueError(f"hand must be one of: {choices}")

    mesh = _safe_child(mesh_root, object_name, f"{object_name}.obj")
    if not mesh.is_file():
        raise FileNotFoundError(f"object mesh not found: {mesh}")

    if normalized_hand is HandType.HUMAN:
        root = _safe_child(capture_root, "human", object_name, str(episode))
        object_poses = root / "object_6d_pose_v2.npz"
        hand_root = root / "hand"
        c2r = root / "C2R.npy"
        required: tuple[Path, ...] = (
            hand_root / "mano",
            hand_root / "mano_params",
            object_poses,
            c2r,
        )
        missing = [path for path in required if not path.exists()]
        if missing:
            raise FileNotFoundError(
                "incomplete human episode; missing: "
                + ", ".join(str(path) for path in missing)
            )
        return EpisodePaths(
            kind=CaptureKind.HUMAN,
            hand=normalized_hand,
            object_name=object_name,
            episode=episode,
            episode_root=root,
            object_mesh=mesh,
            object_poses=object_poses,
            hand_trajectory=hand_root,
            camera_to_robot=c2r,
        )

    spec = _ROBOT_HAND_SPECS[normalized_hand]
    root = _safe_child(
        capture_root, spec.capture_directory, object_name, str(episode)
    )
    object_poses = root / "object_6d_pose_v2.npz"
    hand_root = root / "raw" / "hand"
    arm_root = root / "raw" / "arm"
    timestamps = root / "raw" / "timestamps" / "timestamp.npy"
    c2r = root / "C2R.npy"
    required = (
        hand_root / spec.position_file,
        hand_root / spec.time_file,
        arm_root / "position.npy",
        arm_root / "time.npy",
        timestamps,
        c2r,
        object_poses,
    )
    missing = [path for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            f"incomplete {normalized_hand.value} episode; missing: "
            + ", ".join(str(path) for path in missing)
        )
    return EpisodePaths(
        kind=CaptureKind.ROBOT,
        hand=normalized_hand,
        object_name=object_name,
        episode=episode,
        episode_root=root,
        object_mesh=mesh,
        object_poses=object_poses,
        hand_trajectory=hand_root,
        arm_trajectory=arm_root,
        camera_timestamps=timestamps,
        camera_to_robot=c2r,
    )


def _load_mesh(path: Path) -> trimesh.Trimesh:
    loaded = trimesh.load(path, force="mesh", process=False)
    if isinstance(loaded, trimesh.Trimesh):
        mesh = loaded
    elif isinstance(loaded, trimesh.Scene):
        mesh = trimesh.util.concatenate(tuple(loaded.geometry.values()))
    else:
        raise DatasetFormatError(f"unsupported mesh payload at {path}")
    if len(mesh.vertices) == 0 or len(mesh.faces) == 0:
        raise DatasetFormatError(f"mesh is empty: {path}")
    if not np.isfinite(mesh.vertices).all():
        raise DatasetFormatError(f"mesh contains non-finite vertices: {path}")
    return mesh


def _load_object_poses(path: Path) -> np.ndarray:
    with np.load(path, allow_pickle=False) as payload:
        keyed: list[tuple[int, np.ndarray]] = []
        for key in payload.files:
            match = _POSE_KEY_RE.fullmatch(key)
            if match is not None:
                keyed.append((int(match.group(1)), np.asarray(payload[key], dtype=float)))
        if keyed:
            keyed.sort(key=lambda item: item[0])
            expected = list(range(keyed[0][0], keyed[0][0] + len(keyed)))
            actual = [index for index, _ in keyed]
            if actual != expected:
                raise DatasetFormatError(
                    f"object pose frame keys are not contiguous in {path}"
                )
            poses = np.stack([pose for _, pose in keyed])
        else:
            for key in ("poses", "trajectory", "T", "arr_0"):
                if key in payload.files:
                    poses = np.asarray(payload[key], dtype=float)
                    break
            else:
                raise DatasetFormatError(f"no frame_* pose matrices found in {path}")

    if poses.ndim != 3 or poses.shape[1:] != (4, 4):
        raise DatasetFormatError(
            f"object poses must have shape [N,4,4], got {poses.shape} at {path}"
        )
    if len(poses) < 2 or not np.isfinite(poses).all():
        raise DatasetFormatError(f"object poses must contain finite multiple frames: {path}")
    bottom_error = float(
        np.max(np.abs(poses[:, 3, :] - np.array([0.0, 0.0, 0.0, 1.0])))
    )
    rotations = poses[:, :3, :3]
    orth_error = float(
        np.max(
            np.linalg.norm(
                np.einsum("nji,njk->nik", rotations, rotations) - np.eye(3),
                axis=(1, 2),
            )
        )
    )
    det_error = float(np.max(np.abs(np.linalg.det(rotations) - 1.0)))
    if bottom_error > 1e-4 or orth_error > 1e-3 or det_error > 1e-3:
        raise DatasetFormatError(
            "invalid SE(3) object pose matrices: "
            f"bottom={bottom_error:.3g}, orthogonality={orth_error:.3g}, "
            f"determinant={det_error:.3g}"
        )
    return poses


def _load_npy_numeric(path: Path, *, trusted_object_array: bool = False) -> np.ndarray:
    try:
        value = np.load(path, allow_pickle=False)
    except ValueError:
        if not trusted_object_array:
            raise
        # Historical arm recordings use dtype=object for numeric scalar rows.
        value = np.load(path, allow_pickle=True)
    result = np.asarray(value, dtype=float)
    if result.size == 0 or not np.isfinite(result).all():
        raise DatasetFormatError(f"array is empty or non-finite: {path}")
    return result


def _resample(times: np.ndarray, values: np.ndarray, target: np.ndarray) -> np.ndarray:
    times = np.asarray(times, dtype=float).reshape(-1)
    values = np.asarray(values, dtype=float)
    target = np.asarray(target, dtype=float).reshape(-1)
    if values.ndim != 2 or len(times) != len(values):
        raise DatasetFormatError(
            f"trajectory/time shape mismatch: values={values.shape}, times={times.shape}"
        )
    order = np.argsort(times, kind="stable")
    times = times[order]
    values = values[order]
    times, unique_indices = np.unique(times, return_index=True)
    values = values[unique_indices]
    if len(times) < 2 or np.any(np.diff(target) <= 0):
        raise DatasetFormatError("trajectory timestamps must be strictly increasing")
    columns = [np.interp(target, times, values[:, index]) for index in range(values.shape[1])]
    return np.stack(columns, axis=1)


def _evenly_spaced_rows(values: np.ndarray, limit: int) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    if len(values) <= limit:
        return values
    indices = np.linspace(0, len(values) - 1, limit, dtype=int)
    return values[indices]


def _sample_indices(frame_count: int, max_frames: int) -> np.ndarray:
    return np.unique(
        np.linspace(0, frame_count - 1, min(frame_count, max_frames), dtype=int)
    )


def _rotation_angle_deg(rotation_matrix: np.ndarray) -> float:
    return float(np.degrees(Rotation.from_matrix(rotation_matrix).magnitude()))


def _bridge_short_false_gaps(mask: np.ndarray, maximum_gap: int) -> np.ndarray:
    bridged = np.asarray(mask, dtype=bool).copy()
    if maximum_gap == 0:
        return bridged
    true_indices = np.flatnonzero(bridged)
    for left, right in zip(true_indices[:-1], true_indices[1:]):
        if 0 < right - left - 1 <= maximum_gap:
            bridged[left + 1 : right] = True
    return bridged


def _longest_true_run(mask: np.ndarray) -> tuple[int, int] | None:
    best: tuple[int, int] | None = None
    start: int | None = None
    for index, active in enumerate(np.r_[mask, False]):
        if active and start is None:
            start = index
        elif not active and start is not None:
            candidate = (start, index - 1)
            if best is None or candidate[1] - candidate[0] > best[1] - best[0]:
                best = candidate
            start = None
    return best


def _pose_motion_series(
    poses: np.ndarray,
    times: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return per-frame speed and acceleration, with zeros where undefined."""

    dt = np.diff(times)
    if len(poses) < 2 or len(poses) != len(times) or np.any(dt <= 0):
        raise DatasetFormatError(
            "object poses and timestamps must contain aligned increasing samples"
        )

    linear_velocity = np.diff(poses[:, :3, 3], axis=0) / dt[:, None]
    world_rotation_delta = np.einsum(
        "nij,njk->nik",
        poses[1:, :3, :3],
        np.transpose(poses[:-1, :3, :3], (0, 2, 1)),
    )
    angular_velocity = (
        np.degrees(Rotation.from_matrix(world_rotation_delta).as_rotvec())
        / dt[:, None]
    )

    linear_speed = np.zeros(len(poses), dtype=float)
    angular_speed = np.zeros(len(poses), dtype=float)
    linear_acceleration = np.zeros(len(poses), dtype=float)
    angular_acceleration = np.zeros(len(poses), dtype=float)
    linear_speed[1:] = np.linalg.norm(linear_velocity, axis=1)
    angular_speed[1:] = np.linalg.norm(angular_velocity, axis=1)
    if len(poses) > 2:
        acceleration_dt = (dt[:-1] + dt[1:]) / 2.0
        linear_acceleration[2:] = np.linalg.norm(
            np.diff(linear_velocity, axis=0) / acceleration_dt[:, None],
            axis=1,
        )
        angular_acceleration[2:] = np.linalg.norm(
            np.diff(angular_velocity, axis=0) / acceleration_dt[:, None],
            axis=1,
        )
    return linear_speed, angular_speed, linear_acceleration, angular_acceleration


def _evaluate_states(
    paths: EpisodePaths,
    states: Sequence[_FrameState],
    frame_count: int,
    thresholds: ValidationThresholds,
    initial_issues: Iterable[ValidationIssue] = (),
    dense_object_poses: np.ndarray | None = None,
    dense_object_times: np.ndarray | None = None,
    gravity_direction: np.ndarray | None = None,
) -> EpisodeValidationReport:
    issues = list(initial_issues)
    contact_counts = np.asarray([len(state.contact_fingers) for state in states])
    raw_contact = contact_counts >= thresholds.min_contact_fingers
    active = _bridge_short_false_gaps(
        raw_contact, thresholds.max_contact_gap_samples
    )
    run = _longest_true_run(active)
    phase: ContactPhase | None = None
    contact_loss_event: ContactLossEvent | None = None

    if dense_object_poses is None:
        motion_poses = np.stack([state.world_from_object for state in states])
        motion_times = np.asarray([state.time_s for state in states])
        motion_frames = np.asarray([state.frame for state in states])
    else:
        motion_poses = np.asarray(dense_object_poses, dtype=float)
        if dense_object_times is None:
            raise ValueError(
                "dense_object_times is required with dense_object_poses"
            )
        motion_times = np.asarray(dense_object_times, dtype=float)
        motion_frames = np.arange(len(motion_poses))
    (
        linear_speed,
        angular_speed,
        linear_acceleration,
        angular_acceleration,
    ) = _pose_motion_series(motion_poses, motion_times)
    gravity = np.asarray(
        [0.0, 0.0, -1.0]
        if gravity_direction is None
        else gravity_direction,
        dtype=float,
    ).reshape(-1)
    if gravity.shape != (3,) or not np.isfinite(gravity).all():
        raise ValueError("gravity_direction must contain three finite values")
    gravity_norm = float(np.linalg.norm(gravity))
    if gravity_norm <= 0.0:
        raise ValueError("gravity_direction must be non-zero")
    gravity /= gravity_norm

    metrics: dict[str, float | int | list[str] | None] = {
        "contact_distance_m": thresholds.contact_distance_m,
        "max_contact_fingers": int(np.max(contact_counts, initial=0)),
        "min_hand_object_distance_m": float(
            min(state.min_hand_object_distance_m for state in states)
        ),
    }
    if run is None:
        issues.append(
            ValidationIssue(
                code="NO_STABLE_CONTACT_PHASE",
                message=(
                    "no sampled interval reaches the required simultaneous "
                    f"{thresholds.min_contact_fingers}-finger contact"
                ),
                threshold=float(thresholds.min_contact_fingers),
            )
        )
    else:
        start, end = run
        phase = ContactPhase(
            start_frame=states[start].frame,
            end_frame=states[end].frame,
            start_sample=start,
            end_sample=end,
            sampled_frame_count=end - start + 1,
        )
        original_span = phase.end_frame - phase.start_frame + 1
        metrics["grasp_frame_span"] = original_span
        metrics["grasp_duration_s"] = states[end].time_s - states[start].time_s
        if original_span < thresholds.min_grasp_frames:
            issues.append(
                ValidationIssue(
                    code="GRASP_TOO_SHORT",
                    message="the stable contact phase is shorter than required",
                    frame=phase.start_frame,
                    value=float(original_span),
                    threshold=float(thresholds.min_grasp_frames),
                )
            )

        phase_states = states[start : end + 1]
        base_relative = (
            np.linalg.inv(phase_states[0].world_from_reference)
            @ phase_states[0].world_from_object
        )
        translation_errors: list[float] = []
        rotation_errors: list[float] = []
        object_motion: list[float] = []
        first_object_translation = phase_states[0].world_from_object[:3, 3]
        for state in phase_states:
            relative = (
                np.linalg.inv(state.world_from_reference)
                @ state.world_from_object
            )
            delta = np.linalg.inv(base_relative) @ relative
            translation_errors.append(float(np.linalg.norm(delta[:3, 3])))
            rotation_errors.append(_rotation_angle_deg(delta[:3, :3]))
            object_motion.append(
                float(
                    np.linalg.norm(
                        state.world_from_object[:3, 3] - first_object_translation
                    )
                )
            )

        max_translation = max(translation_errors)
        max_rotation = max(rotation_errors)
        max_motion = max(object_motion)
        metrics["max_relative_translation_m"] = max_translation
        metrics["max_relative_rotation_deg"] = max_rotation
        metrics["object_motion_during_grasp_m"] = max_motion
        if max_motion < thresholds.min_object_motion_m:
            issues.append(
                ValidationIssue(
                    code="OBJECT_NOT_MOVED",
                    message="contact occurred but the object did not move enough",
                    frame=phase.end_frame,
                    value=max_motion,
                    threshold=thresholds.min_object_motion_m,
                )
            )

        if (
            original_span >= thresholds.min_grasp_frames
            and end < len(states) - 1
        ):
            loss_index = end + 1
            next_contact = np.flatnonzero(active[loss_index:])
            if len(next_contact):
                no_contact_sample_count = int(next_contact[0])
            else:
                no_contact_sample_count = len(active) - loss_index
            if no_contact_sample_count >= thresholds.min_contact_loss_samples:
                last_contact_frame = states[end].frame
                loss_frame = states[loss_index].frame
                if len(next_contact):
                    last_loss_index = (
                        loss_index + no_contact_sample_count - 1
                    )
                    loss_window_end_frame = states[last_loss_index].frame
                else:
                    loss_window_end_frame = int(motion_frames[-1])
                observation_mask = (
                    (motion_frames >= last_contact_frame)
                    & (motion_frames <= loss_window_end_frame)
                )
                observation_indices = np.flatnonzero(observation_mask)
                observation_times = motion_times[observation_indices]
                gravity_positions = (
                    motion_poses[observation_indices, :3, 3] @ gravity
                )
                observation_duration = (
                    float(observation_times[-1] - observation_times[0])
                    if len(observation_times) >= 2
                    else 0.0
                )
                gravity_displacement = (
                    float(gravity_positions[-1] - gravity_positions[0])
                    if len(gravity_positions) >= 2
                    else 0.0
                )
                gravity_velocity = (
                    np.diff(gravity_positions) / np.diff(observation_times)
                    if len(observation_times) >= 2
                    else np.empty(0, dtype=float)
                )
                if len(gravity_velocity):
                    velocity_window = max(1, len(gravity_velocity) // 3)
                    initial_gravity_velocity = float(
                        np.median(gravity_velocity[:velocity_window])
                    )
                    final_gravity_velocity = float(
                        np.median(gravity_velocity[-velocity_window:])
                    )
                else:
                    initial_gravity_velocity = 0.0
                    final_gravity_velocity = 0.0
                gravity_velocity_change = (
                    final_gravity_velocity - initial_gravity_velocity
                )
                if (
                    observation_duration
                    < thresholds.min_gravity_observation_s
                    or not len(gravity_velocity)
                ):
                    kind = ContactLossKind.INSUFFICIENT_OBSERVATION
                elif (
                    gravity_displacement
                    >= thresholds.min_gravity_displacement_m
                    or gravity_velocity_change
                    >= thresholds.min_gravity_velocity_change_m_s
                ):
                    kind = ContactLossKind.NORMAL_MOTION
                else:
                    kind = ContactLossKind.TRACKING_ERROR

                contact_loss_event = ContactLossEvent(
                    kind=kind,
                    frame=loss_frame,
                    last_contact_frame=last_contact_frame,
                    no_contact_sample_count=no_contact_sample_count,
                    gravity_direction=tuple(float(value) for value in gravity),
                    observation_duration_s=observation_duration,
                    gravity_displacement_m=gravity_displacement,
                    initial_gravity_velocity_m_s=initial_gravity_velocity,
                    final_gravity_velocity_m_s=final_gravity_velocity,
                    gravity_velocity_change_m_s=gravity_velocity_change,
                )

    metrics["max_object_linear_speed_m_s"] = float(np.max(linear_speed))
    metrics["max_object_angular_speed_deg_s"] = float(np.max(angular_speed))
    metrics["max_object_linear_acceleration_m_s2"] = float(
        np.max(linear_acceleration)
    )
    metrics["max_object_angular_acceleration_deg_s2"] = float(
        np.max(angular_acceleration)
    )

    if (
        contact_loss_event is not None
        and contact_loss_event.kind is ContactLossKind.TRACKING_ERROR
    ):
        issues.append(
            ValidationIssue(
                code="OBJECT_POST_CONTACT_GRAVITY_INCONSISTENT",
                message=(
                    "after persistent contact loss, object motion shows neither "
                    "enough displacement nor increasing velocity along gravity"
                ),
                frame=contact_loss_event.frame,
                value=contact_loss_event.gravity_velocity_change_m_s,
                threshold=thresholds.min_gravity_velocity_change_m_s,
            )
        )

    return EpisodeValidationReport(
        kind=paths.kind,
        hand=paths.hand,
        object_name=paths.object_name,
        episode=paths.episode,
        valid=not issues,
        inputs=paths,
        frame_count=frame_count,
        sampled_frame_count=len(states),
        contact_phase=phase,
        contact_loss_event=contact_loss_event,
        metrics=metrics,
        issues=issues,
        notes=[
            "Mesh distances use deterministic surface samples; they are an "
            "approximation, not exact continuous collision detection.",
            "All dataset inputs are opened read-only; reports exist only in memory/stdout.",
        ],
    )


def _robot_geometry_nodes(
    robot: RobotModule,
    spec: _RobotHandSpec,
) -> tuple[dict[str, str], list[str]]:
    scene = robot.scene
    parent_by_node = scene.graph.transforms.parents
    node_by_link = {
        str(parent_by_node[node]): str(node) for node in scene.graph.nodes_geometry
    }
    missing = sorted(
        {
            link
            for links in spec.finger_links.values()
            for link in links
            if link not in node_by_link
        }
    )
    if spec.reference_link not in scene.graph.nodes or missing:
        raise DatasetFormatError(
            "robot URDF lacks the configured palm/fingertip geometry: "
            f"reference={spec.reference_link}, missing={missing}"
        )
    arm_links = {"link_base", *(f"link{index}" for index in range(1, 7))}
    hand_nodes = [
        node
        for link, node in node_by_link.items()
        if link not in arm_links
    ]
    return node_by_link, hand_nodes


def _node_world_points(
    robot: RobotModule, node: str, sample_limit: int
) -> np.ndarray:
    scene = robot.scene
    transform, geometry_name = scene.graph.get(node)
    mesh = scene.geometry[geometry_name]
    points = np.concatenate(
        (
            _evenly_spaced_rows(np.asarray(mesh.vertices), sample_limit),
            _evenly_spaced_rows(np.asarray(mesh.triangles_center), sample_limit),
        ),
        axis=0,
    )
    return trimesh.transform_points(points, transform)


def _expand_inspire_master_joints(
    master: Mapping[str, np.ndarray],
    joint_names: Sequence[str],
    mimic_factors: Mapping[str, tuple[str, float]],
) -> np.ndarray:
    columns: list[np.ndarray] = []
    for joint_name in joint_names:
        if joint_name in master:
            columns.append(master[joint_name])
        elif joint_name in mimic_factors:
            source, factor = mimic_factors[joint_name]
            columns.append(master[source] * factor)
        else:
            raise DatasetFormatError(
                f"no Inspire trajectory mapping for URDF joint: {joint_name}"
            )
    return np.stack(columns, axis=1)


def _convert_hand_trajectory(
    hand: HandType,
    action: np.ndarray,
    joint_names: Sequence[str],
) -> np.ndarray:
    if action.ndim != 2:
        raise DatasetFormatError(f"hand trajectory must be 2-D, got {action.shape}")
    if hand is HandType.ALLEGRO:
        if action.shape[1] != len(joint_names):
            raise DatasetFormatError(
                f"Allegro trajectory has {action.shape[1]} joints, expected "
                f"{len(joint_names)}"
            )
        return action
    if action.shape[1] != 6:
        raise DatasetFormatError(
            f"{hand.value} raw trajectory must have 6 channels, got {action.shape}"
        )

    if hand is HandType.INSPIRE:
        values = np.zeros_like(action, dtype=float)
        values[:, 0] = 1.40 * (1.0 - action[:, 5] / 1000.0)
        values[:, 1] = 0.60 * (1.0 - action[:, 4] / 1000.0)
        for output_index, input_index in enumerate((3, 2, 1, 0), start=2):
            encoder = action[:, input_index]
            degrees = (
                -4e-8 * encoder**3
                + 3e-5 * encoder**2
                - 0.0704 * encoder
                + 83.572
            )
            values[:, output_index] = np.radians(degrees)
        master = {
            "right_thumb_1_joint": values[:, 0],
            "right_thumb_2_joint": values[:, 1],
            "right_index_1_joint": values[:, 2],
            "right_middle_1_joint": values[:, 3],
            "right_ring_1_joint": values[:, 4],
            "right_little_1_joint": values[:, 5],
        }
        factors = {
            "right_thumb_3_joint": ("right_thumb_2_joint", 0.60),
            "right_thumb_4_joint": ("right_thumb_2_joint", 0.80),
            "right_index_2_joint": ("right_index_1_joint", 1.05),
            "right_middle_2_joint": ("right_middle_1_joint", 1.05),
            "right_ring_2_joint": ("right_ring_1_joint", 1.05),
            "right_little_2_joint": ("right_little_1_joint", 1.18),
        }
        return _expand_inspire_master_joints(master, joint_names, factors)

    radians = np.pi / 1800.0
    master = {
        "right_thumb_1_joint": (1740.0 - action[:, 5]) * radians,
        "right_thumb_2_joint": (1740.0 - action[:, 4]) * radians,
        "right_index_1_joint": (1740.0 - action[:, 3]) * radians,
        "right_middle_1_joint": (1740.0 - action[:, 2]) * radians,
        "right_ring_1_joint": (1350.0 - action[:, 1]) * radians,
        "right_little_1_joint": (1800.0 - action[:, 0]) * radians,
    }
    factors = {
        "right_thumb_3_joint": ("right_thumb_2_joint", 1.2953),
        "right_thumb_4_joint": ("right_thumb_2_joint", 1.1608),
        "right_index_2_joint": ("right_index_1_joint", 1.1545),
        "right_middle_2_joint": ("right_middle_1_joint", 1.1545),
        "right_ring_2_joint": ("right_ring_1_joint", 1.1545),
        "right_little_2_joint": ("right_little_1_joint", 1.1545),
    }
    return _expand_inspire_master_joints(master, joint_names, factors)


def _validate_robot_episode(
    paths: EpisodePaths,
    object_mesh: trimesh.Trimesh,
    object_poses_camera: np.ndarray,
    thresholds: ValidationThresholds,
    robot_urdf: Path | None,
) -> EpisodeValidationReport:
    assert paths.arm_trajectory is not None
    assert paths.camera_timestamps is not None
    assert paths.camera_to_robot is not None

    spec = _ROBOT_HAND_SPECS[paths.hand]
    arm = _load_npy_numeric(
        paths.arm_trajectory / "position.npy", trusted_object_array=True
    )
    arm_time = _load_npy_numeric(
        paths.arm_trajectory / "time.npy", trusted_object_array=True
    )
    hand_action = _load_npy_numeric(
        paths.hand_trajectory / spec.position_file,
        trusted_object_array=paths.hand is HandType.INSPIRE_F1,
    )
    hand_time = _load_npy_numeric(
        paths.hand_trajectory / spec.time_file,
        trusted_object_array=paths.hand is HandType.INSPIRE_F1,
    )
    camera_time = _load_npy_numeric(paths.camera_timestamps).reshape(-1)

    frame_count = len(object_poses_camera)
    if len(camera_time) < 2:
        raise DatasetFormatError("robot camera timeline must contain at least 2 frames")
    object_time = np.linspace(camera_time[0], camera_time[-1], frame_count)

    selected_urdf = robot_urdf or spec.urdf
    robot = RobotModule(str(selected_urdf))
    arm_dof = arm.shape[1]
    hand_joint_names = robot.joint_names[arm_dof:]
    hand_qpos = _convert_hand_trajectory(
        paths.hand,
        _resample(hand_time, hand_action, object_time),
        hand_joint_names,
    )
    qpos = np.concatenate(
        (
            _resample(arm_time, arm, object_time),
            hand_qpos,
        ),
        axis=1,
    )

    c2r = _load_npy_numeric(paths.camera_to_robot)
    if c2r.shape != (4, 4):
        raise DatasetFormatError(f"C2R.npy must be 4x4, got {c2r.shape}")
    robot_from_camera = np.linalg.inv(c2r)
    object_poses = np.einsum(
        "ij,njk->nik", robot_from_camera, object_poses_camera
    )

    if qpos.shape[1] != len(robot.joint_names):
        raise DatasetFormatError(
            f"arm+hand trajectory has {qpos.shape[1]} DoF but URDF expects "
            f"{len(robot.joint_names)}"
        )
    robot.update_cfg(qpos[0])
    node_by_link, hand_nodes = _robot_geometry_nodes(robot, spec)
    surface = _ObjectSurface(object_mesh, thresholds.surface_sample_count)
    sample_indices = _sample_indices(frame_count, thresholds.max_frames)
    issues: list[ValidationIssue] = []

    for joint_index, joint_name in enumerate(robot.joint_names):
        lower, upper = robot.joint_limits[joint_name]
        values = qpos[:, joint_index]
        bad = np.flatnonzero(
            (values < lower - thresholds.joint_limit_tolerance_rad)
            | (values > upper + thresholds.joint_limit_tolerance_rad)
        )
        if len(bad):
            index = int(bad[0])
            issues.append(
                ValidationIssue(
                    code="JOINT_LIMIT",
                    message=f"{joint_name} is outside [{lower:.4g}, {upper:.4g}] rad",
                    frame=index,
                    value=float(values[index]),
                )
            )

        velocity_limit = robot.joint_map[joint_name].limit.velocity
        if velocity_limit is not None and velocity_limit > 0:
            velocity = np.abs(np.diff(values) / np.diff(object_time))
            allowed = float(velocity_limit * thresholds.joint_velocity_scale)
            bad_velocity = np.flatnonzero(velocity > allowed)
            if len(bad_velocity):
                index = int(bad_velocity[0] + 1)
                issues.append(
                    ValidationIssue(
                        code="JOINT_VELOCITY",
                        message=f"{joint_name} exceeds the scaled URDF velocity limit",
                        frame=index,
                        value=float(velocity[index - 1]),
                        threshold=allowed,
                    )
                )

    states: list[_FrameState] = []
    node_sample_limit = max(50, thresholds.surface_sample_count // 12)
    for frame in sample_indices:
        robot.update_cfg(qpos[frame])
        scene = robot.scene
        points_by_link: dict[str, np.ndarray] = {}
        for link, node in node_by_link.items():
            if node in hand_nodes:
                points_by_link[link] = _node_world_points(
                    robot, node, node_sample_limit
                )

        contacts: list[str] = []
        for finger, links in spec.finger_links.items():
            finger_points = np.concatenate(
                [points_by_link[link] for link in links], axis=0
            )
            if (
                float(surface.distance(finger_points, object_poses[frame]).min())
                <= thresholds.contact_distance_m
            ):
                contacts.append(finger)
        hand_points = np.concatenate(list(points_by_link.values()), axis=0)
        states.append(
            _FrameState(
                frame=int(frame),
                time_s=float(object_time[frame] - object_time[0]),
                contact_fingers=tuple(contacts),
                min_hand_object_distance_m=float(
                    surface.distance(hand_points, object_poses[frame]).min()
                ),
                world_from_reference=np.asarray(
                    scene.graph.get(spec.reference_link)[0], dtype=float
                ),
                world_from_object=object_poses[frame],
            )
        )
    report = _evaluate_states(
        paths,
        states,
        frame_count,
        thresholds,
        initial_issues=issues,
        dense_object_poses=object_poses,
        dense_object_times=object_time - object_time[0],
        gravity_direction=np.array([0.0, 0.0, -1.0]),
    )
    report.notes.append(
        f"{paths.hand.value} robot qpos is arm position + converted hand state, "
        "independently interpolated to the camera time span; object poses use "
        "inv(C2R) @ camera_pose."
    )
    report.notes.append(
        "Robot self-collision is assumed absent and is not evaluated."
    )
    return report


def _load_human_root_pose(path: Path) -> np.ndarray:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        rotation = np.asarray(payload["global_orient"], dtype=float).reshape(3, 3)
        joints = np.asarray(payload["joints"], dtype=float)
    except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
        raise DatasetFormatError(f"invalid MANO parameter JSON: {path}") from exc
    if joints.ndim != 2 or joints.shape[1] != 3 or len(joints) == 0:
        raise DatasetFormatError(f"MANO joints must have shape [J,3]: {path}")
    pose = np.eye(4)
    pose[:3, :3] = rotation
    pose[:3, 3] = joints[0]
    return pose


def _validate_human_episode(
    paths: EpisodePaths,
    object_mesh: trimesh.Trimesh,
    object_poses: np.ndarray,
    thresholds: ValidationThresholds,
) -> EpisodeValidationReport:
    assert paths.camera_to_robot is not None
    mano_dir = paths.hand_trajectory / "mano"
    params_dir = paths.hand_trajectory / "mano_params"
    mesh_files = sorted(mano_dir.glob("*.obj"))
    param_files = sorted(params_dir.glob("*.json"))
    frame_count = len(object_poses)
    if len(mesh_files) != frame_count or len(param_files) != frame_count:
        raise DatasetFormatError(
            "human trajectory length mismatch: "
            f"MANO meshes={len(mesh_files)}, params={len(param_files)}, "
            f"object poses={frame_count}"
        )
    expected_mesh_names = [f"{index:05d}.obj" for index in range(frame_count)]
    expected_param_names = [f"{index:05d}.json" for index in range(frame_count)]
    if [path.name for path in mesh_files] != expected_mesh_names:
        raise DatasetFormatError("MANO mesh filenames must be contiguous NNNNN.obj")
    if [path.name for path in param_files] != expected_param_names:
        raise DatasetFormatError("MANO parameter filenames must be contiguous NNNNN.json")

    c2r = _load_npy_numeric(paths.camera_to_robot)
    if c2r.shape != (4, 4):
        raise DatasetFormatError(f"C2R.npy must be 4x4, got {c2r.shape}")
    gravity_camera = c2r[:3, :3] @ np.array([0.0, 0.0, -1.0])

    surface = _ObjectSurface(object_mesh, thresholds.surface_sample_count)
    sample_indices = _sample_indices(frame_count, thresholds.max_frames)
    states: list[_FrameState] = []
    for frame in sample_indices:
        hand_mesh = _load_mesh(mesh_files[frame])
        if len(hand_mesh.vertices) <= max(_MANO_TIP_VERTICES.values()):
            raise DatasetFormatError(
                f"MANO mesh has too few vertices for fingertip indices: "
                f"{mesh_files[frame]}"
            )
        tip_points = np.stack(
            [hand_mesh.vertices[index] for index in _MANO_TIP_VERTICES.values()]
        )
        tip_distances = surface.distance(tip_points, object_poses[frame])
        contacts = tuple(
            name
            for name, distance in zip(_MANO_TIP_VERTICES, tip_distances)
            if distance <= thresholds.contact_distance_m
        )
        hand_points = np.concatenate(
            (
                _evenly_spaced_rows(
                    hand_mesh.vertices, thresholds.surface_sample_count // 2
                ),
                _evenly_spaced_rows(
                    hand_mesh.triangles_center, thresholds.surface_sample_count // 2
                ),
            ),
            axis=0,
        )
        states.append(
            _FrameState(
                frame=int(frame),
                time_s=float(frame / 30.0),
                contact_fingers=contacts,
                min_hand_object_distance_m=float(
                    surface.distance(hand_points, object_poses[frame]).min()
                ),
                world_from_reference=_load_human_root_pose(param_files[frame]),
                world_from_object=object_poses[frame],
            )
        )
    report = _evaluate_states(
        paths,
        states,
        frame_count,
        thresholds,
        dense_object_poses=object_poses,
        dense_object_times=np.arange(frame_count, dtype=float) / 30.0,
        gravity_direction=gravity_camera,
    )
    report.notes.append(
        "Human MANO meshes and object poses are paired by identical zero-based "
        "frame index in their shared camera coordinate system (30 Hz assumed); "
        "C2R rotates robot-frame gravity into that camera frame."
    )
    return report


def validate_dataset_episode(
    object_name: str,
    episode: int,
    hand: str,
    *,
    thresholds: ValidationThresholds | None = None,
    mesh_root: Path = DEFAULT_MESH_ROOT,
    capture_root: Path = DEFAULT_CAPTURE_ROOT,
    robot_urdf: Path | None = None,
) -> EpisodeValidationReport:
    """Validate one explicitly selected hand/object/episode capture.

    No input file is modified and no report file is created.  The return value
    can be serialized by a caller or printed by the bundled CLI.
    """

    thresholds = thresholds or ValidationThresholds()
    paths = discover_episode_paths(
        object_name,
        episode,
        hand,
        mesh_root=mesh_root,
        capture_root=capture_root,
    )
    object_mesh = _load_mesh(paths.object_mesh)
    object_poses = _load_object_poses(paths.object_poses)
    if paths.kind is CaptureKind.ROBOT:
        return _validate_robot_episode(
            paths, object_mesh, object_poses, thresholds, robot_urdf
        )
    return _validate_human_episode(
        paths, object_mesh, object_poses, thresholds
    )
