"""Validation of robot/human grasp episodes in the ECCV 2026 layout.

Dataset inputs are always opened read-only and reports are returned in memory.
Frame projection is excluded from the active validation path.
"""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Mapping, Sequence

import cv2
import numpy as np
import trimesh
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation

from paradex.visualization.robot import RobotModule


DEFAULT_MESH_ROOT = Path("/home/temp_id/shared_data/mesh_new")
DEFAULT_CAPTURE_ROOT = Path("/home/temp_id/shared_data/capture/eccv2026/v0")
DEFAULT_PROJECTION_OVERLAY_ROOT = Path(__file__).resolve().parent / (
    "projection_overlays"
)
DEFAULT_ROBOT_URDF = (
    Path(__file__).resolve().parents[3] / "rsc" / "robot" / "xarm_allegro_v5.urdf"
)
_ROBOT_RESOURCE_ROOT = Path(__file__).resolve().parents[3] / "rsc" / "robot"

_OBJECT_NAME_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]*")
_POSE_KEY_RE = re.compile(r"frame_(\d+)")
_OBJECT_POSE_FILENAMES = (
    "object_6d_pose.npz",
    "object_6d_pose_v1.npz",
)
_PROJECTION_CAMERA_SERIAL = "23263780"
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
    FLOOR_SUPPORTED = "floor_supported"
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
    projection_video: Path | None = None
    camera_intrinsics: Path | None = None
    camera_extrinsics: Path | None = None

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
    min_contact_fingers: int = 1
    min_grasp_frames: int = 10
    max_contact_gap_samples: int = 1
    min_contact_loss_samples: int = 2
    min_object_motion_m: float = 0.01
    min_gravity_observation_s: float = 0.1
    min_gravity_displacement_m: float = 0.010
    min_gravity_velocity_change_m_s: float = 0.05
    floor_contact_tolerance_m: float = 0.005
    max_unreached_floor_height_m: float = 0.2
    min_floor_contact_frames: int = 3
    floor_reach_grace_s: float = 1.0
    floor_fall_time_scale: float = 3.0
    projection_metric: str = "edge_median"
    max_projection_median_edge_distance_px: float = 20.0
    projection_edge_tolerance_px: float = 20.0
    min_projection_mask_iou: float = 0.5
    projection_mask_source: str = "temporal_foreground"
    projection_mask_background_samples: int = 9
    projection_mask_difference_threshold: float = 30.0
    projection_mask_search_margin_px: int = 120
    max_pregrasp_translation_m: float = 0.05
    max_pregrasp_rotation_deg: float = 120.0
    pregrasp_baseline_frames: int = 15
    pregrasp_grace_frames: int = 15
    max_object_position_jump_m: float = 0.1
    object_jump_local_factor: float = 8.0
    object_jump_window_frames: int = 5
    object_jump_baseline_floor_m: float = 0.002
    surface_sample_count: int = 6_000

    def __post_init__(self) -> None:
        positive = {
            "max_frames": self.max_frames,
            "contact_distance_m": self.contact_distance_m,
            "min_contact_fingers": self.min_contact_fingers,
            "min_grasp_frames": self.min_grasp_frames,
            "min_contact_loss_samples": self.min_contact_loss_samples,
            "min_gravity_observation_s": self.min_gravity_observation_s,
            "min_floor_contact_frames": self.min_floor_contact_frames,
            "floor_reach_grace_s": self.floor_reach_grace_s,
            "floor_fall_time_scale": self.floor_fall_time_scale,
            "max_projection_median_edge_distance_px": (
                self.max_projection_median_edge_distance_px
            ),
            "projection_edge_tolerance_px": self.projection_edge_tolerance_px,
            "projection_mask_background_samples": (
                self.projection_mask_background_samples
            ),
            "projection_mask_difference_threshold": (
                self.projection_mask_difference_threshold
            ),
            "max_pregrasp_translation_m": self.max_pregrasp_translation_m,
            "max_pregrasp_rotation_deg": self.max_pregrasp_rotation_deg,
            "pregrasp_baseline_frames": self.pregrasp_baseline_frames,
            "max_object_position_jump_m": self.max_object_position_jump_m,
            "object_jump_local_factor": self.object_jump_local_factor,
            "object_jump_window_frames": self.object_jump_window_frames,
            "object_jump_baseline_floor_m": (
                self.object_jump_baseline_floor_m
            ),
            "surface_sample_count": self.surface_sample_count,
        }
        bad = [name for name, value in positive.items() if value <= 0]
        if bad:
            raise ValueError(f"thresholds must be positive: {', '.join(bad)}")
        if self.max_contact_gap_samples < 0:
            raise ValueError("max_contact_gap_samples must be non-negative")
        if self.min_contact_fingers != 1:
            raise ValueError(
                "min_contact_fingers is fixed to 1: any fingertip contact "
                "defines grasp contact"
            )
        if self.pregrasp_grace_frames < 0:
            raise ValueError("pregrasp_grace_frames must be non-negative")
        if self.projection_metric not in {"edge_median", "mask_iou"}:
            raise ValueError(
                "projection_metric must be 'edge_median' or 'mask_iou'"
            )
        if self.projection_mask_source not in {
            "temporal_foreground",
            "banana_hsv",
        }:
            raise ValueError(
                "projection_mask_source must be 'temporal_foreground' or "
                "'banana_hsv'"
            )
        if not 0.0 < self.min_projection_mask_iou <= 1.0:
            raise ValueError("min_projection_mask_iou must be in (0, 1]")
        if self.projection_mask_search_margin_px < 0:
            raise ValueError(
                "projection_mask_search_margin_px must be non-negative"
            )
        non_negative = {
            "min_object_motion_m": self.min_object_motion_m,
            "min_gravity_displacement_m": self.min_gravity_displacement_m,
            "min_gravity_velocity_change_m_s": (
                self.min_gravity_velocity_change_m_s
            ),
            "floor_contact_tolerance_m": self.floor_contact_tolerance_m,
            "max_unreached_floor_height_m": (
                self.max_unreached_floor_height_m
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
    floor_height_at_loss_m: float | None
    minimum_floor_height_m: float | None
    floor_contact_frame: int | None
    floor_contact_deadline_s: float | None

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["kind"] = self.kind.value
        return result


@dataclass(frozen=True)
class ProjectionFrameAlignment:
    """Image-edge agreement for one projected object silhouette."""

    position: str
    video_frame: int
    pose_frame: int
    projected_pixel_count: int
    boundary_pixel_count: int
    median_edge_distance_px: float | None
    p90_edge_distance_px: float | None
    edge_coverage: float
    valid: bool
    observed_mask_pixel_count: int | None = None
    mask_intersection_pixel_count: int | None = None
    mask_iou: float | None = None
    mask_precision: float | None = None
    mask_recall: float | None = None


@dataclass(frozen=True)
class ProjectionAlignment:
    """First/last-frame alignment result for the validation camera."""

    camera_serial: str
    metric: str
    video_frame_count: int
    pose_frame_count: int
    edge_tolerance_px: float
    max_median_edge_distance_px: float
    min_mask_iou: float
    frames: tuple[ProjectionFrameAlignment, ...]
    overlay_path: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "camera_serial": self.camera_serial,
            "metric": self.metric,
            "video_frame_count": self.video_frame_count,
            "pose_frame_count": self.pose_frame_count,
            "edge_tolerance_px": self.edge_tolerance_px,
            "max_median_edge_distance_px": (
                self.max_median_edge_distance_px
            ),
            "min_mask_iou": self.min_mask_iou,
            "frames": [asdict(frame) for frame in self.frames],
            "overlay_path": self.overlay_path,
        }


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
    projection_alignment: ProjectionAlignment | None = None
    metrics: dict[str, Any] = field(default_factory=dict)
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
            "projection_alignment": (
                self.projection_alignment.to_dict()
                if self.projection_alignment is not None
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


def _validate_projection_overlay_dir(
    output_dir: Path,
    *,
    mesh_root: Path,
    capture_root: Path,
) -> Path:
    resolved = output_dir.expanduser().resolve()
    for source_root in (mesh_root, capture_root):
        source = source_root.expanduser().resolve()
        try:
            resolved.relative_to(source)
        except ValueError:
            continue
        raise ValueError(
            "projection overlays must be outside source data roots: "
            f"{resolved}"
        )
    if resolved.exists() and not resolved.is_dir():
        raise ValueError(
            f"projection overlay output is not a directory: {resolved}"
        )
    return resolved


def _validate_object_and_episode(object_name: str, episode: int) -> None:
    if _OBJECT_NAME_RE.fullmatch(object_name) is None or object_name in {".", ".."}:
        raise ValueError(
            "object name must be one directory component containing only "
            "letters, digits, '.', '_' or '-'"
        )
    if isinstance(episode, bool) or int(episode) != episode or episode < 0:
        raise ValueError("episode must be a non-negative integer")


def _resolve_object_pose_path(episode_root: Path) -> Path:
    """Select the non-v2 object trajectory, preferring the unsuffixed file."""

    for filename in _OBJECT_POSE_FILENAMES:
        candidate = episode_root / filename
        if candidate.is_file():
            return candidate
    expected = ", ".join(str(episode_root / name) for name in _OBJECT_POSE_FILENAMES)
    raise FileNotFoundError(
        "non-v2 object pose not found; expected one of: " + expected
    )


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
        object_poses = _resolve_object_pose_path(root)
        hand_root = root / "hand"
        c2r = root / "C2R.npy"
        required: tuple[Path, ...] = (
            object_poses,
            hand_root / "mano",
            hand_root / "mano_params",
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
    object_poses = _resolve_object_pose_path(root)
    hand_root = root / "raw" / "hand"
    arm_root = root / "raw" / "arm"
    timestamps = root / "raw" / "timestamps" / "timestamp.npy"
    c2r = root / "C2R.npy"
    required = (
        object_poses,
        hand_root / spec.position_file,
        hand_root / spec.time_file,
        arm_root / "position.npy",
        arm_root / "time.npy",
        timestamps,
        c2r,
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


def _homogeneous_camera_extrinsic(value: Any) -> np.ndarray:
    matrix = np.asarray(value, dtype=float)
    if matrix.shape == (3, 4):
        matrix = np.vstack((matrix, np.array([0.0, 0.0, 0.0, 1.0])))
    if matrix.shape != (4, 4) or not np.isfinite(matrix).all():
        raise DatasetFormatError(
            f"{_PROJECTION_CAMERA_SERIAL} extrinsic must be finite 3x4 or 4x4"
        )
    return matrix


def _load_projection_camera(
    paths: EpisodePaths,
) -> tuple[np.ndarray, np.ndarray, int, int]:
    assert paths.camera_intrinsics is not None
    assert paths.camera_extrinsics is not None
    try:
        intrinsics_payload = json.loads(
            paths.camera_intrinsics.read_text(encoding="utf-8")
        )
        extrinsics_payload = json.loads(
            paths.camera_extrinsics.read_text(encoding="utf-8")
        )
        camera = intrinsics_payload[_PROJECTION_CAMERA_SERIAL]
        intrinsic = np.asarray(camera["intrinsics_undistort"], dtype=float)
        calibration_width = int(camera["width"])
        calibration_height = int(camera["height"])
        camera_from_world = _homogeneous_camera_extrinsic(
            extrinsics_payload[_PROJECTION_CAMERA_SERIAL]
        )
    except (
        KeyError,
        TypeError,
        ValueError,
        json.JSONDecodeError,
    ) as exc:
        raise DatasetFormatError(
            f"invalid {_PROJECTION_CAMERA_SERIAL} camera calibration"
        ) from exc
    if intrinsic.shape != (3, 3) or not np.isfinite(intrinsic).all():
        raise DatasetFormatError(
            f"{_PROJECTION_CAMERA_SERIAL} intrinsics must be finite 3x3"
        )
    if calibration_width <= 0 or calibration_height <= 0:
        raise DatasetFormatError(
            f"{_PROJECTION_CAMERA_SERIAL} calibration dimensions must be positive"
        )
    return (
        intrinsic,
        camera_from_world,
        calibration_width,
        calibration_height,
    )


def _read_projection_video_frames(
    path: Path,
) -> tuple[int, list[tuple[str, int, np.ndarray]]]:
    capture = cv2.VideoCapture(str(path))
    try:
        if not capture.isOpened():
            raise DatasetFormatError(f"cannot open projection video: {path}")
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        if frame_count < 2:
            raise DatasetFormatError(
                f"projection video must contain at least 2 frames: {path}"
            )
        indices = (0, frame_count - 1)
        frames: list[tuple[str, int, np.ndarray]] = []
        for position, frame_index in zip(("first", "last"), indices):
            if not capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index):
                raise DatasetFormatError(
                    f"cannot seek projection video frame {frame_index}: {path}"
                )
            success, image = capture.read()
            if not success or image is None or image.ndim != 3:
                raise DatasetFormatError(
                    f"cannot decode projection video frame {frame_index}: {path}"
                )
            frames.append((position, frame_index, image))
        return frame_count, frames
    finally:
        capture.release()


def _estimate_projection_background(
    path: Path,
    frame_count: int,
    sample_count: int,
) -> np.ndarray:
    """Estimate a static background plate with a temporal pixel median."""

    capture = cv2.VideoCapture(str(path))
    try:
        if not capture.isOpened():
            raise DatasetFormatError(f"cannot open projection video: {path}")
        sample_indices = np.unique(
            np.linspace(
                0,
                frame_count - 1,
                min(frame_count, sample_count),
                dtype=int,
            )
        )
        samples: list[np.ndarray] = []
        expected_shape: tuple[int, ...] | None = None
        for frame_index in sample_indices:
            if not capture.set(cv2.CAP_PROP_POS_FRAMES, int(frame_index)):
                raise DatasetFormatError(
                    f"cannot seek background frame {frame_index}: {path}"
                )
            success, image = capture.read()
            if not success or image is None or image.ndim != 3:
                raise DatasetFormatError(
                    f"cannot decode background frame {frame_index}: {path}"
                )
            if expected_shape is None:
                expected_shape = image.shape
            elif image.shape != expected_shape:
                raise DatasetFormatError(
                    f"projection video resolution changes at frame {frame_index}"
                )
            samples.append(image)
        stack = np.stack(samples, axis=0)
        stack.sort(axis=0)
        return stack[len(stack) // 2].copy()
    finally:
        capture.release()


def _projection_observed_mask(
    image: np.ndarray,
    background: np.ndarray | None,
    projected_mask: np.ndarray,
    thresholds: ValidationThresholds,
) -> np.ndarray:
    """Extract the foreground component anchored nearest the projection."""

    if thresholds.projection_mask_source == "banana_hsv":
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        foreground = (
            (hsv[:, :, 0] >= 5)
            & (hsv[:, :, 0] <= 55)
            & (hsv[:, :, 1] >= 55)
            & (hsv[:, :, 2] >= 45)
        )
    else:
        if background is None or image.shape != background.shape:
            raise DatasetFormatError(
                "projection frame and temporal background have different "
                "shapes"
            )
        difference = np.max(cv2.absdiff(image, background), axis=2)
        difference = cv2.GaussianBlur(difference, (5, 5), 0)
        foreground = (
            difference >= thresholds.projection_mask_difference_threshold
        )

        # Remove cloth shadows: they are darker than the background but retain
        # almost the same normalized chromaticity.
        image_f = image.astype(np.float32)
        background_f = background.astype(np.float32)
        image_sum = np.sum(image_f, axis=2) + 1e-6
        background_sum = np.sum(background_f, axis=2) + 1e-6
        brightness_ratio = image_sum / background_sum
        chroma_distance = np.sum(
            np.abs(
                image_f / image_sum[..., None]
                - background_f / background_sum[..., None]
            ),
            axis=2,
        )
        background_spread = np.max(background_f, axis=2) - np.min(
            background_f, axis=2
        )
        cloth_like = (
            (np.mean(background_f, axis=2) >= 90.0)
            & (background_spread <= 45.0)
        )
        shadow = (
            (brightness_ratio >= 0.35)
            & (brightness_ratio <= 0.97)
            & (chroma_distance <= 0.10)
            & cloth_like
        )
        foreground &= ~shadow

    mask = foreground.astype(np.uint8)
    kernel = np.ones((3, 3), dtype=np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=3)

    projected_points = cv2.findNonZero(projected_mask)
    if projected_points is None:
        return np.zeros_like(projected_mask)
    x, y, width, height = cv2.boundingRect(projected_points)
    margin = thresholds.projection_mask_search_margin_px
    x0 = max(0, x - margin)
    y0 = max(0, y - margin)
    x1 = min(mask.shape[1], x + width + margin)
    y1 = min(mask.shape[0], y + height + margin)
    roi = np.zeros_like(mask)
    roi[y0:y1, x0:x1] = 1
    mask &= roi

    component_count, labels, stats, centroids = (
        cv2.connectedComponentsWithStats(mask, connectivity=8)
    )
    if component_count <= 1:
        return np.zeros_like(projected_mask)
    projected_boolean = projected_mask > 0
    projected_area = int(np.count_nonzero(projected_boolean))
    projected_center = np.mean(
        np.argwhere(projected_boolean)[:, ::-1],
        axis=0,
    )
    minimum_area = max(64, int(0.01 * projected_area))
    best_component: int | None = None
    best_key: tuple[int, int, float, float] | None = None
    for component in range(1, component_count):
        area = int(stats[component, cv2.CC_STAT_AREA])
        if area < minimum_area:
            continue
        component_mask = labels == component
        overlap = int(
            np.count_nonzero(component_mask & projected_boolean)
        )
        center_distance = float(
            np.linalg.norm(centroids[component] - projected_center)
        )
        size_mismatch = abs(np.log(max(area, 1) / max(projected_area, 1)))
        key = (
            int(overlap > 0),
            overlap,
            -center_distance,
            -size_mismatch,
        )
        if best_key is None or key > best_key:
            best_key = key
            best_component = component
    if best_component is None:
        return np.zeros_like(projected_mask)

    observed = (labels == best_component).astype(np.uint8) * 255
    observed = cv2.morphologyEx(
        observed,
        cv2.MORPH_CLOSE,
        np.ones((7, 7), dtype=np.uint8),
        iterations=2,
    )
    return observed


def _projection_mask_iou(
    projected_mask: np.ndarray,
    observed_mask: np.ndarray,
) -> tuple[int, int, float, float, float]:
    projected = projected_mask > 0
    observed = observed_mask > 0
    projected_count = int(np.count_nonzero(projected))
    observed_count = int(np.count_nonzero(observed))
    intersection = int(np.count_nonzero(projected & observed))
    union = projected_count + observed_count - intersection
    iou = float(intersection / union) if union else 0.0
    precision = (
        float(intersection / projected_count) if projected_count else 0.0
    )
    recall = float(intersection / observed_count) if observed_count else 0.0
    return observed_count, intersection, iou, precision, recall


def _project_object_silhouette(
    mesh: trimesh.Trimesh,
    world_from_object: np.ndarray,
    camera_from_world: np.ndarray,
    intrinsic: np.ndarray,
    width: int,
    height: int,
) -> np.ndarray:
    camera_from_object = camera_from_world @ world_from_object
    camera_vertices = trimesh.transform_points(
        np.asarray(mesh.vertices), camera_from_object
    )
    faces = np.asarray(mesh.faces, dtype=np.int64)
    positive_faces = faces[
        np.all(camera_vertices[faces, 2] > 1e-4, axis=1)
    ]
    mask = np.zeros((height, width), dtype=np.uint8)
    if not len(positive_faces):
        return mask

    used_vertices = np.unique(positive_faces)
    projected = np.zeros((len(camera_vertices), 2), dtype=float)
    homogeneous_pixels = (
        intrinsic @ camera_vertices[used_vertices].T
    ).T
    projected[used_vertices] = (
        homogeneous_pixels[:, :2] / homogeneous_pixels[:, 2, None]
    )
    if not np.isfinite(projected[used_vertices]).all():
        raise DatasetFormatError("object projection contains non-finite pixels")

    face_pixels = projected[positive_faces]
    intersects_image = (
        (np.max(face_pixels[:, :, 0], axis=1) >= 0)
        & (np.min(face_pixels[:, :, 0], axis=1) < width)
        & (np.max(face_pixels[:, :, 1], axis=1) >= 0)
        & (np.min(face_pixels[:, :, 1], axis=1) < height)
    )
    visible_faces = positive_faces[intersects_image]
    if not len(visible_faces):
        return mask

    pixels = np.rint(projected)
    pixels[:, 0] = np.clip(pixels[:, 0], -2 * width, 3 * width)
    pixels[:, 1] = np.clip(pixels[:, 1], -2 * height, 3 * height)
    pixels = pixels.astype(np.int32)
    # Fill triangles one at a time. Passing every triangle to one fillPoly call
    # applies an even-odd rule to overlapping faces and creates false holes.
    for face in visible_faces:
        cv2.fillConvexPoly(mask, pixels[face], 255)
    return mask


def _projection_edge_alignment(
    image: np.ndarray,
    silhouette: np.ndarray,
    edge_tolerance_px: float,
) -> tuple[int, int, float | None, float | None, float]:
    projected_pixel_count = int(np.count_nonzero(silhouette))
    boundary = cv2.morphologyEx(
        silhouette,
        cv2.MORPH_GRADIENT,
        np.ones((3, 3), dtype=np.uint8),
    )
    boundary_mask = boundary > 0
    boundary_pixel_count = int(np.count_nonzero(boundary_mask))
    if projected_pixel_count == 0 or boundary_pixel_count == 0:
        return projected_pixel_count, boundary_pixel_count, None, None, 0.0

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(gray, 50, 150) > 0
    if not np.any(edges):
        return projected_pixel_count, boundary_pixel_count, None, None, 0.0
    distance = cv2.distanceTransform(
        (~edges).astype(np.uint8),
        cv2.DIST_L2,
        3,
    )
    boundary_distances = distance[boundary_mask]
    return (
        projected_pixel_count,
        boundary_pixel_count,
        float(np.median(boundary_distances)),
        float(np.percentile(boundary_distances, 90)),
        float(np.mean(boundary_distances <= edge_tolerance_px)),
    )


def _render_projection_panel(
    *,
    position: str,
    video_frame: int,
    pose_frame: int,
    image: np.ndarray,
    silhouette: np.ndarray,
    observed_mask: np.ndarray | None,
    median_distance: float | None,
    p90_distance: float | None,
    edge_coverage: float,
    mask_iou: float | None,
    mask_precision: float | None,
    mask_recall: float | None,
    thresholds: ValidationThresholds,
    valid: bool,
) -> np.ndarray:
    color = (40, 190, 40) if valid else (30, 30, 230)
    rendered = image.copy()
    colored = image.copy()
    if thresholds.projection_metric == "mask_iou":
        assert observed_mask is not None
        projected = silhouette > 0
        observed = observed_mask > 0
        colored[projected & ~observed] = (30, 30, 230)
        colored[observed & ~projected] = (230, 120, 30)
        colored[projected & observed] = (40, 190, 40)
    else:
        colored[silhouette > 0] = color
    rendered = cv2.addWeighted(colored, 0.32, rendered, 0.68, 0.0)
    contours, _ = cv2.findContours(
        silhouette,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )
    cv2.drawContours(rendered, contours, -1, color, thickness=3)
    if observed_mask is not None:
        observed_contours, _ = cv2.findContours(
            observed_mask,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE,
        )
        cv2.drawContours(
            rendered,
            observed_contours,
            -1,
            (230, 120, 30),
            thickness=3,
        )

    status = "PASS" if valid else "FAIL"
    if thresholds.projection_metric == "mask_iou":
        iou_text = "n/a" if mask_iou is None else f"{mask_iou:.3f}"
        precision_text = (
            "n/a" if mask_precision is None else f"{mask_precision:.3f}"
        )
        recall_text = "n/a" if mask_recall is None else f"{mask_recall:.3f}"
        lines = (
            f"camera {_PROJECTION_CAMERA_SERIAL} | {position} | {status}",
            f"video frame {video_frame} | pose frame {pose_frame}",
            (
                f"projection mask IoU {iou_text} "
                f"(limit {thresholds.min_projection_mask_iou:.3f})"
            ),
            f"mask precision {precision_text} | recall {recall_text}",
            "green=intersection red=projection blue=observed",
        )
    else:
        median_text = (
            "n/a"
            if median_distance is None
            else f"{median_distance:.2f}px"
        )
        p90_text = (
            "n/a" if p90_distance is None else f"{p90_distance:.2f}px"
        )
        lines = (
            f"camera {_PROJECTION_CAMERA_SERIAL} | {position} | {status}",
            f"video frame {video_frame} | pose frame {pose_frame}",
            (
                f"median edge distance {median_text} "
                f"(limit "
                f"{thresholds.max_projection_median_edge_distance_px:.2f}px)"
            ),
            f"p90 edge distance {p90_text}",
            (
                f"edge coverage@"
                f"{thresholds.projection_edge_tolerance_px:.0f}px "
                f"{edge_coverage * 100.0:.1f}%"
            ),
        )
    font_scale = max(0.65, min(rendered.shape[:2]) / 1200.0)
    thickness = max(1, int(round(font_scale * 2)))
    line_height = int(round(34 * font_scale))
    text_width = max(
        cv2.getTextSize(
            line,
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            thickness,
        )[0][0]
        for line in lines
    )
    box_height = line_height * len(lines) + 24
    cv2.rectangle(
        rendered,
        (12, 12),
        (min(text_width + 36, rendered.shape[1] - 1), box_height),
        (0, 0, 0),
        thickness=-1,
    )
    for index, line in enumerate(lines):
        cv2.putText(
            rendered,
            line,
            (24, 12 + line_height * (index + 1)),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            (255, 255, 255),
            thickness,
            cv2.LINE_AA,
        )

    return rendered


def _write_projection_composite(
    output_dir: Path,
    panels: Sequence[np.ndarray],
    *,
    episode: int,
    overwrite: bool,
) -> Path:
    if len(panels) != 2:
        raise ValueError("projection composite requires exactly two panels")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{episode}.jpg"
    if output_path.exists() and not overwrite:
        raise ValueError(
            f"projection overlay already exists; pass overwrite to replace it: "
            f"{output_path}"
        )
    target_height = min(panel.shape[0] for panel in panels)
    resized_panels = [
        (
            panel
            if panel.shape[0] == target_height
            else cv2.resize(
                panel,
                (
                    int(round(panel.shape[1] * target_height / panel.shape[0])),
                    target_height,
                ),
                interpolation=cv2.INTER_AREA,
            )
        )
        for panel in panels
    ]
    composite = np.concatenate(resized_panels, axis=1)
    if not cv2.imwrite(
        str(output_path),
        composite,
        [cv2.IMWRITE_JPEG_QUALITY, 92],
    ):
        raise ValueError(f"failed to write projection overlay: {output_path}")
    return output_path


def _evaluate_projection_alignment(
    paths: EpisodePaths,
    object_mesh: trimesh.Trimesh,
    object_poses_world: np.ndarray,
    thresholds: ValidationThresholds,
    *,
    overlay_dir: Path | None = None,
    overwrite_overlays: bool = False,
) -> tuple[ProjectionAlignment, list[ValidationIssue]]:
    assert paths.projection_video is not None
    intrinsic, camera_from_world, calibration_width, calibration_height = (
        _load_projection_camera(paths)
    )
    video_frame_count, video_frames = _read_projection_video_frames(
        paths.projection_video
    )
    background = (
        _estimate_projection_background(
            paths.projection_video,
            video_frame_count,
            thresholds.projection_mask_background_samples,
        )
        if (
            thresholds.projection_metric == "mask_iou"
            and thresholds.projection_mask_source == "temporal_foreground"
        )
        else None
    )
    pose_frame_count = len(object_poses_world)
    results: list[ProjectionFrameAlignment] = []
    issues: list[ValidationIssue] = []
    panels: list[np.ndarray] = []

    for position, video_frame, image in video_frames:
        height, width = image.shape[:2]
        scaled_intrinsic = intrinsic.copy()
        scaled_intrinsic[0, :] *= width / calibration_width
        scaled_intrinsic[1, :] *= height / calibration_height
        pose_frame = int(
            round(
                video_frame
                * (pose_frame_count - 1)
                / (video_frame_count - 1)
            )
        )
        silhouette = _project_object_silhouette(
            object_mesh,
            object_poses_world[pose_frame],
            camera_from_world,
            scaled_intrinsic,
            width,
            height,
        )
        observed_mask: np.ndarray | None = None
        observed_mask_pixel_count: int | None = None
        mask_intersection_pixel_count: int | None = None
        mask_iou: float | None = None
        mask_precision: float | None = None
        mask_recall: float | None = None
        if thresholds.projection_metric == "mask_iou":
            observed_mask = _projection_observed_mask(
                image,
                background,
                silhouette,
                thresholds,
            )
            projected_pixel_count = int(np.count_nonzero(silhouette))
            boundary = cv2.morphologyEx(
                silhouette,
                cv2.MORPH_GRADIENT,
                np.ones((3, 3), dtype=np.uint8),
            )
            boundary_pixel_count = int(np.count_nonzero(boundary))
            (
                observed_mask_pixel_count,
                mask_intersection_pixel_count,
                mask_iou,
                mask_precision,
                mask_recall,
            ) = _projection_mask_iou(silhouette, observed_mask)
            median_distance = None
            p90_distance = None
            edge_coverage = 0.0
            valid = mask_iou >= thresholds.min_projection_mask_iou
        else:
            (
                projected_pixel_count,
                boundary_pixel_count,
                median_distance,
                p90_distance,
                edge_coverage,
            ) = _projection_edge_alignment(
                image,
                silhouette,
                thresholds.projection_edge_tolerance_px,
            )
            valid = (
                median_distance is not None
                and median_distance
                <= thresholds.max_projection_median_edge_distance_px
            )
        if overlay_dir is not None:
            panels.append(
                _render_projection_panel(
                    position=position,
                    video_frame=video_frame,
                    pose_frame=pose_frame,
                    image=image,
                    silhouette=silhouette,
                    observed_mask=observed_mask,
                    median_distance=median_distance,
                    p90_distance=p90_distance,
                    edge_coverage=edge_coverage,
                    mask_iou=mask_iou,
                    mask_precision=mask_precision,
                    mask_recall=mask_recall,
                    thresholds=thresholds,
                    valid=valid,
                )
            )
        results.append(
            ProjectionFrameAlignment(
                position=position,
                video_frame=video_frame,
                pose_frame=pose_frame,
                projected_pixel_count=projected_pixel_count,
                boundary_pixel_count=boundary_pixel_count,
                median_edge_distance_px=median_distance,
                p90_edge_distance_px=p90_distance,
                edge_coverage=edge_coverage,
                observed_mask_pixel_count=observed_mask_pixel_count,
                mask_intersection_pixel_count=(
                    mask_intersection_pixel_count
                ),
                mask_iou=mask_iou,
                mask_precision=mask_precision,
                mask_recall=mask_recall,
                valid=valid,
            )
        )
        if not valid:
            issues.append(
                ValidationIssue(
                    code="OBJECT_CAMERA_PROJECTION_MISMATCH",
                    message=(
                        f"{_PROJECTION_CAMERA_SERIAL} {position} frame object "
                        + (
                            "projection mask has insufficient IoU with the "
                            "observed foreground mask"
                            if thresholds.projection_metric == "mask_iou"
                            else "silhouette does not align with image edges"
                        )
                    ),
                    frame=pose_frame,
                    value=(
                        mask_iou
                        if thresholds.projection_metric == "mask_iou"
                        else median_distance
                    ),
                    threshold=(
                        thresholds.min_projection_mask_iou
                        if thresholds.projection_metric == "mask_iou"
                        else thresholds.max_projection_median_edge_distance_px
                    ),
                )
            )

    overlay_path = (
        _write_projection_composite(
            overlay_dir,
            panels,
            episode=paths.episode,
            overwrite=overwrite_overlays,
        )
        if overlay_dir is not None
        else None
    )
    return (
        ProjectionAlignment(
            camera_serial=_PROJECTION_CAMERA_SERIAL,
            metric=thresholds.projection_metric,
            video_frame_count=video_frame_count,
            pose_frame_count=pose_frame_count,
            edge_tolerance_px=thresholds.projection_edge_tolerance_px,
            max_median_edge_distance_px=(
                thresholds.max_projection_median_edge_distance_px
            ),
            min_mask_iou=thresholds.min_projection_mask_iou,
            frames=tuple(results),
            overlay_path=(
                str(overlay_path) if overlay_path is not None else None
            ),
        ),
        issues,
    )


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


def _detect_object_position_jumps(
    poses: np.ndarray,
    thresholds: ValidationThresholds,
) -> tuple[dict[str, Any], list[ValidationIssue]]:
    """Detect isolated frame-to-frame translation discontinuities."""

    translations = np.asarray(poses[:, :3, 3], dtype=float)
    displacements = np.linalg.norm(np.diff(translations, axis=0), axis=1)
    maximum_index = int(np.argmax(displacements))
    jump_frames: list[int] = []
    issues: list[ValidationIssue] = []
    window = thresholds.object_jump_window_frames

    for step_index, displacement in enumerate(displacements):
        start = max(0, step_index - window)
        end = min(len(displacements), step_index + window + 1)
        local = np.concatenate(
            (
                displacements[start:step_index],
                displacements[step_index + 1 : end],
            )
        )
        local_median = (
            float(np.median(local))
            if len(local)
            else thresholds.object_jump_baseline_floor_m
        )
        effective_threshold = max(
            thresholds.max_object_position_jump_m,
            thresholds.object_jump_local_factor
            * max(
                local_median,
                thresholds.object_jump_baseline_floor_m,
            ),
        )
        if displacement <= effective_threshold:
            continue
        frame = step_index + 1
        jump_frames.append(frame)
        issues.append(
            ValidationIssue(
                code="OBJECT_POSITION_JUMP",
                message=(
                    "object translation has an abrupt frame-to-frame "
                    f"discontinuity; local median={local_median:.6f} m"
                ),
                frame=frame,
                value=float(displacement),
                threshold=float(effective_threshold),
            )
        )

    return (
        {
            "max_object_frame_translation_m": float(
                displacements[maximum_index]
            ),
            "max_object_frame_translation_frame": maximum_index + 1,
            "object_position_jump_count": len(jump_frames),
            "object_position_jump_frames": jump_frames,
        },
        issues,
    )


def _evaluate_pregrasp_motion(
    poses: np.ndarray,
    phase: ContactPhase | None,
    thresholds: ValidationThresholds,
) -> tuple[dict[str, Any], list[ValidationIssue]]:
    """Measure object excursion before the grasp-approach grace interval."""

    if phase is None:
        return (
            {
                "pregrasp_motion_status": (
                    "not_applicable_no_stable_grasp"
                ),
                "pregrasp_frame_count": 0,
            },
            [],
        )

    last_frame = min(
        phase.start_frame - thresholds.pregrasp_grace_frames - 1,
        len(poses) - 1,
    )
    if last_frame < thresholds.pregrasp_baseline_frames - 1:
        return (
            {
                "pregrasp_motion_status": (
                    "not_applicable_insufficient_observation"
                ),
                "pregrasp_frame_count": max(last_frame + 1, 0),
                "pregrasp_last_evaluated_frame": (
                    last_frame if last_frame >= 0 else None
                ),
                "pregrasp_grace_frames": thresholds.pregrasp_grace_frames,
            },
            [],
        )

    pregrasp_poses = np.asarray(poses[: last_frame + 1], dtype=float)

    baseline_count = min(
        thresholds.pregrasp_baseline_frames,
        len(pregrasp_poses),
    )
    initial_translation = np.median(
        pregrasp_poses[:baseline_count, :3, 3],
        axis=0,
    )
    translation_excursion = np.linalg.norm(
        pregrasp_poses[:, :3, 3] - initial_translation,
        axis=1,
    )
    translation_frame = int(np.argmax(translation_excursion))
    max_translation = float(translation_excursion[translation_frame])

    initial_rotation = Rotation.from_matrix(
        pregrasp_poses[:baseline_count, :3, :3]
    ).mean().as_matrix()
    rotation_delta = np.einsum(
        "nij,jk->nik",
        pregrasp_poses[:, :3, :3],
        initial_rotation.T,
    )
    rotation_excursion = np.degrees(
        Rotation.from_matrix(rotation_delta).magnitude()
    )
    rotation_frame = int(np.argmax(rotation_excursion))
    max_rotation = float(rotation_excursion[rotation_frame])

    step_translation = np.linalg.norm(
        np.diff(pregrasp_poses[:, :3, 3], axis=0),
        axis=1,
    )
    issues: list[ValidationIssue] = []
    if max_translation > thresholds.max_pregrasp_translation_m:
        issues.append(
            ValidationIssue(
                code="OBJECT_PRE_GRASP_TRANSLATION_EXCESSIVE",
                message=(
                    "before persistent grasp onset, object translation "
                    "deviates excessively from its robust initial position"
                ),
                frame=translation_frame,
                value=max_translation,
                threshold=thresholds.max_pregrasp_translation_m,
            )
        )
    if max_rotation > thresholds.max_pregrasp_rotation_deg:
        issues.append(
            ValidationIssue(
                code="OBJECT_PRE_GRASP_ROTATION_EXCESSIVE",
                message=(
                    "before persistent grasp onset, object rotation deviates "
                    "excessively from its initial orientation"
                ),
                frame=rotation_frame,
                value=max_rotation,
                threshold=thresholds.max_pregrasp_rotation_deg,
            )
        )

    return (
        {
            "pregrasp_motion_status": (
                "excessive" if issues else "normal"
            ),
            "pregrasp_frame_count": len(pregrasp_poses),
            "pregrasp_last_evaluated_frame": last_frame,
            "pregrasp_grace_frames": thresholds.pregrasp_grace_frames,
            "pregrasp_baseline_frame_count": baseline_count,
            "max_pregrasp_translation_m": max_translation,
            "max_pregrasp_translation_frame": translation_frame,
            "pregrasp_translation_path_length_m": float(
                np.sum(step_translation)
            ),
            "max_pregrasp_rotation_deg": max_rotation,
            "max_pregrasp_rotation_frame": rotation_frame,
        },
        issues,
    )


def _object_bottom_coordinates(
    poses: np.ndarray,
    object_vertices: np.ndarray,
    gravity: np.ndarray,
) -> np.ndarray:
    """Return the furthest object-surface coordinate along downward gravity."""

    vertices = np.asarray(object_vertices, dtype=float)
    if (
        vertices.ndim != 2
        or vertices.shape[1] != 3
        or len(vertices) == 0
        or not np.isfinite(vertices).all()
    ):
        raise ValueError("object_vertices must have finite shape [N,3]")
    local_gravity = np.einsum("nji,j->ni", poses[:, :3, :3], gravity)
    support = np.empty(len(poses), dtype=float)
    for start in range(0, len(poses), 256):
        end = min(start + 256, len(poses))
        support[start:end] = np.max(
            vertices @ local_gravity[start:end].T,
            axis=0,
        )
    return poses[:, :3, 3] @ gravity + support


def _first_persistent_true(mask: np.ndarray, minimum_frames: int) -> int | None:
    if len(mask) < minimum_frames:
        return None
    counts = np.convolve(
        np.asarray(mask, dtype=int),
        np.ones(minimum_frames, dtype=int),
        mode="valid",
    )
    matches = np.flatnonzero(counts == minimum_frames)
    return int(matches[0]) if len(matches) else None


def _evaluate_states(
    paths: EpisodePaths,
    states: Sequence[_FrameState],
    frame_count: int,
    thresholds: ValidationThresholds,
    dense_object_poses: np.ndarray | None = None,
    dense_object_times: np.ndarray | None = None,
    gravity_direction: np.ndarray | None = None,
    object_vertices: np.ndarray | None = None,
) -> EpisodeValidationReport:
    issues: list[ValidationIssue] = []
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
    floor_heights: np.ndarray | None = None
    if object_vertices is not None:
        bottom_coordinates = _object_bottom_coordinates(
            motion_poses,
            object_vertices,
            gravity,
        )
        floor_heights = bottom_coordinates[0] - bottom_coordinates

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
                floor_height_at_loss: float | None = None
                minimum_floor_height: float | None = None
                floor_contact_frame: int | None = None
                floor_contact_deadline: float | None = None
                floor_supported_at_loss = False
                if floor_heights is not None:
                    observation_floor_heights = floor_heights[
                        observation_indices
                    ]
                    floor_height_at_loss = float(
                        observation_floor_heights[0]
                    )
                    minimum_floor_height = float(
                        np.min(observation_floor_heights)
                    )
                    floor_contact_local = _first_persistent_true(
                        observation_floor_heights
                        <= thresholds.floor_contact_tolerance_m,
                        thresholds.min_floor_contact_frames,
                    )
                    if floor_contact_local is not None:
                        floor_contact_frame = int(
                            motion_frames[
                                observation_indices[floor_contact_local]
                            ]
                        )
                        floor_supported_at_loss = floor_contact_local == 0
                    fall_height = max(
                        floor_height_at_loss
                        - thresholds.floor_contact_tolerance_m,
                        0.0,
                    )
                    floor_contact_deadline = max(
                        thresholds.floor_reach_grace_s,
                        thresholds.floor_fall_time_scale
                        * np.sqrt(2.0 * fall_height / 9.81),
                    )

                if floor_supported_at_loss:
                    kind = ContactLossKind.FLOOR_SUPPORTED
                elif floor_contact_frame is not None:
                    kind = ContactLossKind.NORMAL_MOTION
                elif (
                    observation_duration
                    < thresholds.min_gravity_observation_s
                    or not len(gravity_velocity)
                ):
                    kind = ContactLossKind.INSUFFICIENT_OBSERVATION
                elif (
                    minimum_floor_height is not None
                    and minimum_floor_height
                    > thresholds.max_unreached_floor_height_m
                    and floor_contact_deadline is not None
                    and observation_duration >= floor_contact_deadline
                ):
                    kind = ContactLossKind.TRACKING_ERROR
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
                    floor_height_at_loss_m=floor_height_at_loss,
                    minimum_floor_height_m=minimum_floor_height,
                    floor_contact_frame=floor_contact_frame,
                    floor_contact_deadline_s=floor_contact_deadline,
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
        missed_floor = (
            contact_loss_event.minimum_floor_height_m is not None
            and contact_loss_event.minimum_floor_height_m
            > thresholds.max_unreached_floor_height_m
            and contact_loss_event.floor_contact_frame is None
            and contact_loss_event.floor_contact_deadline_s is not None
            and contact_loss_event.observation_duration_s
            >= contact_loss_event.floor_contact_deadline_s
        )
        issues.append(
            ValidationIssue(
                code=(
                    "OBJECT_DID_NOT_REACH_INFERRED_FLOOR"
                    if missed_floor
                    else "OBJECT_POST_CONTACT_GRAVITY_INCONSISTENT"
                ),
                message=(
                    "object remained above the inferred floor beyond the "
                    "generous fall deadline"
                    if missed_floor
                    else (
                        "after persistent contact loss, both gravity-axis "
                        "displacement and downward velocity change are below "
                        "their minimum acceptance thresholds "
                        f"({thresholds.min_gravity_displacement_m:.6f} m and "
                        f"{thresholds.min_gravity_velocity_change_m_s:.6f} "
                        "m/s)"
                    )
                ),
                frame=contact_loss_event.frame,
                value=(
                    contact_loss_event.minimum_floor_height_m
                    if missed_floor
                    else contact_loss_event.gravity_velocity_change_m_s
                ),
                threshold=(
                    thresholds.max_unreached_floor_height_m
                    if missed_floor
                    else thresholds.min_gravity_velocity_change_m_s
                ),
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
            "All dataset inputs are opened read-only.",
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
    object_poses_world: np.ndarray,
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

    frame_count = len(object_poses_world)
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
    robot_from_world = np.linalg.inv(c2r)
    object_poses = np.einsum(
        "ij,njk->nik", robot_from_world, object_poses_world
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
        dense_object_poses=object_poses,
        dense_object_times=object_time - object_time[0],
        gravity_direction=np.array([0.0, 0.0, -1.0]),
        object_vertices=np.asarray(object_mesh.vertices),
    )
    report.notes.append(
        f"{paths.hand.value} robot qpos is arm position + converted hand state, "
        "independently interpolated to the camera time span; object poses use "
        "inv(C2R) @ world_pose."
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
    gravity_world = c2r[:3, :3] @ np.array([0.0, 0.0, -1.0])

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
        gravity_direction=gravity_world,
        object_vertices=np.asarray(object_mesh.vertices),
    )
    report.notes.append(
        "Human MANO meshes and object poses are paired by identical zero-based "
        "frame index in their shared world coordinate system (30 Hz assumed); "
        "C2R rotates robot-frame gravity into that world frame."
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
    projection_overlay_dir: Path | None = None,
    overwrite_projection_overlays: bool = False,
) -> EpisodeValidationReport:
    """Validate one explicitly selected hand/object/episode capture.

    No input file is modified. The return value can be serialized by a caller
    or printed by the bundled CLI. Frame projection is not run and the
    compatibility output arguments do not create any images.
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
    motion_report = (
        _validate_human_episode(
            paths,
            object_mesh,
            object_poses,
            thresholds,
        )
        if paths.kind is CaptureKind.HUMAN
        else _validate_robot_episode(
            paths,
            object_mesh,
            object_poses,
            thresholds,
            robot_urdf,
        )
    )
    gravity_issue_codes = {
        "OBJECT_POST_CONTACT_GRAVITY_INCONSISTENT",
        "OBJECT_DID_NOT_REACH_INFERRED_FLOOR",
    }
    gravity_issues = [
        issue
        for issue in motion_report.issues
        if issue.code in gravity_issue_codes
    ]
    pregrasp_metrics, pregrasp_issues = _evaluate_pregrasp_motion(
        object_poses,
        motion_report.contact_phase,
        thresholds,
    )
    jump_metrics, jump_issues = _detect_object_position_jumps(
        object_poses,
        thresholds,
    )
    issues = [*gravity_issues, *pregrasp_issues, *jump_issues]
    gravity_status = (
        motion_report.contact_loss_event.kind.value
        if motion_report.contact_loss_event is not None
        else "not_applicable_no_persistent_contact_loss"
    )
    return EpisodeValidationReport(
        kind=paths.kind,
        hand=paths.hand,
        object_name=paths.object_name,
        episode=paths.episode,
        valid=not issues,
        inputs=paths,
        frame_count=len(object_poses),
        sampled_frame_count=motion_report.sampled_frame_count,
        contact_phase=motion_report.contact_phase,
        contact_loss_event=motion_report.contact_loss_event,
        metrics={
            "gravity_check_status": gravity_status,
            **pregrasp_metrics,
            **jump_metrics,
        },
        issues=issues,
        notes=[
            (
                "Validity combines three checks: limited object motion before "
                "persistent grasp onset, gravity/floor-consistent motion after "
                "persistent grasp release, and abrupt object-position jump "
                "detection. Frame projection is not evaluated."
            ),
            (
                "Hand-object contact is used only to locate persistent grasp "
                "onset and release. Missing contact, grasp duration, "
                "insufficient motion during grasp, self-collision, and "
                "robot-joint values do not directly invalidate the episode."
            ),
            (
                "Pre-grasp motion is measured from robust initial translation "
                "and orientation baselines, excluding a short grace interval "
                "immediately before persistent grasp onset."
            ),
            (
                "Position jumps must exceed both the absolute translation "
                "limit and an adaptive threshold derived from neighboring "
                "frame displacements."
            ),
        ],
    )
