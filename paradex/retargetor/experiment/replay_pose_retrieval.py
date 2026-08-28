"""Retrieve a banana episode and replay it with a release-relative dish transfer."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from functools import lru_cache
import json
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from paradex.retargetor.experiment import replay_object_relative as core
from paradex.retargetor.experiment import (
    replay_object_relative_first_frame as apex_core,
)


DEFAULT_EPISODE_ROOT = Path("/home/temp_id/shared_data/capture/0825_test_4/banana")
DEFAULT_EEF_APEX_ANNOTATION_NAME = "eef_apex_annotations.json"
ALLEGRO_V5_INDEX_TIP_LINK = "link_3_0_tip"
ALLEGRO_V5_THUMB_TIP_LINK = "link_15_0_tip"
DEFAULT_CANDIDATE_EPISODES = (
    6,
    7,
    9,
    10,
    11,
    13,
    15,
    18,
    19,
    21,
    22,
    23,
    24,
    25,
    28,
    30,
)


@dataclass(frozen=True)
class ObjectPoseMatch:
    episode: core.Episode
    source_object_robot: np.ndarray
    translation_error_m: float
    rotation_error_rad: float
    score: float


@dataclass(frozen=True)
class DishInterleave:
    trajectory: core.ReplayTrajectory
    apex_plan_frame: int
    target_eef_robot: np.ndarray
    suffix_correction: np.ndarray
    shifted_apex_object_robot: np.ndarray
    release_object_target_robot: np.ndarray
    predicted_release_object_robot: np.ndarray
    release_position_error_m: float
    transfer_frame_count: int
    transfer_seconds: float


@dataclass(frozen=True)
class BoxPinkTransfer:
    trajectory: core.ReplayTrajectory
    target_eef_robot: np.ndarray
    fingertip_midpoint_eef: np.ndarray
    target_fingertip_midpoint_robot: np.ndarray
    transfer_frame_count: int
    transfer_seconds: float
    vertical_frame_count: int
    vertical_seconds: float
    horizontal_frame_count: int
    horizontal_seconds: float
    release_frame_count: int
    release_seconds: float


def parse_episode_ids(value: str | Iterable[int]) -> tuple[int, ...]:
    if isinstance(value, str):
        try:
            episode_ids = tuple(
                int(item.strip()) for item in value.split(",") if item.strip()
            )
        except ValueError as exc:
            raise argparse.ArgumentTypeError(
                "episode IDs must be comma-separated integers"
            ) from exc
    else:
        episode_ids = tuple(int(item) for item in value)
    if not episode_ids or any(item < 0 for item in episode_ids):
        raise argparse.ArgumentTypeError(
            "candidate episodes must contain non-negative IDs"
        )
    if len(set(episode_ids)) != len(episode_ids):
        raise argparse.ArgumentTypeError("candidate episode IDs must be unique")
    return episode_ids


def load_candidate_episodes(
    episode_root: Path,
    episode_ids: Iterable[int],
) -> list[core.Episode]:
    args = argparse.Namespace(robot="allegro_v5")
    episodes: list[core.Episode] = []
    for episode_id in episode_ids:
        root = Path(episode_root).expanduser() / str(episode_id)
        if not root.is_dir():
            raise FileNotFoundError(f"candidate episode directory not found: {root}")
        episodes.append(core._load_episode_root(args, root))
    return episodes


def rank_by_object_pose(
    current_object_robot: np.ndarray,
    episodes: Iterable[core.Episode],
    *,
    translation_scale_m: float,
    rotation_scale_rad: float,
) -> list[ObjectPoseMatch]:
    """Rank source object poses by a dimensionless weighted SE(3) distance."""

    if translation_scale_m <= 0 or rotation_scale_rad <= 0:
        raise ValueError("retrieval translation/rotation scales must be positive")
    current = core._as_transform(current_object_robot, label="current object pose")
    matches: list[ObjectPoseMatch] = []
    for episode in episodes:
        source = np.linalg.inv(episode.source_c2r) @ episode.source_object_world
        translation_error = float(np.linalg.norm(source[:3, 3] - current[:3, 3]))
        rotation_error = core._rotation_error_rad(source[:3, :3], current[:3, :3])
        matches.append(
            ObjectPoseMatch(
                episode=episode,
                source_object_robot=source,
                translation_error_m=translation_error,
                rotation_error_rad=rotation_error,
                score=float(
                    np.hypot(
                        translation_error / translation_scale_m,
                        rotation_error / rotation_scale_rad,
                    )
                ),
            )
        )
    if not matches:
        raise ValueError("at least one candidate episode is required")
    return sorted(
        matches, key=lambda match: (match.score, int(match.episode.root.name))
    )


def _read_live_robot_state() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Read arm and hand through temporary read-only controller clients."""

    arm_pose, arm_qpos = core._live_xarm_preview_state()
    hand_qpos = core._live_allegro_v5_preview_qpos()
    return arm_pose, arm_qpos, hand_qpos


def _gotrack_pose_records_path(episode_root: Path) -> Path:
    candidates = sorted(
        Path(episode_root).glob(
            "object_tracking_foundpose_gotrack/*_gotrack_01/attempt_01/"
            "gotrack_tracking/gotrack_output/banana/world_pose_records.json"
        )
    )
    if len(candidates) == 1:
        return candidates[0]
    if not candidates:
        raise FileNotFoundError(
            f"GoTrack banana world pose records not found for episode {episode_root}"
        )
    raise ValueError(
        f"multiple *_gotrack_01 banana world pose records found: {candidates}"
    )


def load_gotrack_world_poses(
    episode_root: Path,
    video_frames: Iterable[int],
) -> tuple[Path, dict[int, np.ndarray]]:
    """Load selected zero-based video-frame banana poses from GoTrack JSON."""

    requested = {int(frame) for frame in video_frames}
    if not requested or any(frame < 0 for frame in requested):
        raise ValueError("GoTrack video frames must be non-negative and non-empty")
    path = _gotrack_pose_records_path(episode_root)
    try:
        records = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"could not read GoTrack pose records {path}: {exc}") from exc
    if not isinstance(records, list):
        raise ValueError(f"GoTrack pose records must contain a JSON list: {path}")

    poses: dict[int, np.ndarray] = {}
    for record in records:
        if not isinstance(record, dict) or "frame_index" not in record:
            continue
        frame = int(record["frame_index"])
        if frame not in requested:
            continue
        if frame in poses:
            raise ValueError(f"duplicate GoTrack pose for video frame {frame}: {path}")
        if record.get("status") != "ok":
            raise ValueError(
                f"GoTrack pose at video frame {frame} is not valid: "
                f"status={record.get('status')!r}, path={path}"
            )
        poses[frame] = core._as_transform(
            record.get("pose_world"),
            label=f"GoTrack banana pose_world frame {frame}",
        )
    missing = sorted(requested - poses.keys())
    if missing:
        raise ValueError(f"GoTrack poses missing at video frame(s) {missing}: {path}")
    return path, poses


def interleave_dish_transfer(
    trajectory: core.ReplayTrajectory,
    *,
    episode_apex_frame: int,
    mapped_apex_object_robot: np.ndarray,
    mapped_release_object_robot: np.ndarray,
    dish_point_robot: np.ndarray,
    clearance_m: float,
    linear_speed_mps: float,
    max_translation_m: float,
    min_seconds: float,
    rate_hz: float,
) -> DishInterleave:
    """Shift the apex EEF so the tracked release banana pose reaches the dish."""

    if (
        linear_speed_mps <= 0
        or max_translation_m <= 0
        or rate_hz <= 0
        or min_seconds < 0
        or clearance_m < 0
    ):
        raise ValueError(
            "dish transfer speed/max-distance/rate must be positive; duration and "
            "clearance must be non-negative"
        )
    dish_point = np.asarray(dish_point_robot, dtype=np.float64).reshape(-1)
    if dish_point.shape != (3,) or not np.all(np.isfinite(dish_point)):
        raise ValueError(
            f"dish point must be a finite xyz vector, got {dish_point_robot}"
        )
    apex_plan_frame = trajectory.transition_frame_count - 1 + episode_apex_frame
    if not 0 <= apex_plan_frame < len(trajectory.arm_poses) - 1:
        raise IndexError(
            f"episode apex frame {episode_apex_frame} maps outside replay suffix"
        )

    apex_pose = core._as_transform(
        trajectory.arm_poses[apex_plan_frame], label="transformed apex EEF pose"
    )
    mapped_apex_object = core._as_transform(
        mapped_apex_object_robot,
        label="mapped apex banana pose",
    )
    mapped_release_object = core._as_transform(
        mapped_release_object_robot,
        label="mapped right-before-release banana pose",
    )
    release_object_target = mapped_release_object.copy()
    release_object_target[:3, 3] = dish_point
    release_object_target[2, 3] += clearance_m
    suffix_correction = release_object_target @ np.linalg.inv(mapped_release_object)
    if not np.allclose(suffix_correction[:3, :3], np.eye(3), atol=1e-9):
        raise RuntimeError(
            "banana release position correction must be translation-only"
        )
    correction_distance = float(np.linalg.norm(suffix_correction[:3, 3]))
    if correction_distance > max_translation_m:
        raise RuntimeError(
            f"dish transfer distance {correction_distance:.3f} m exceeds configured "
            f"maximum {max_translation_m:.3f} m"
        )
    target_eef = suffix_correction @ apex_pose
    shifted_apex_object = suffix_correction @ mapped_apex_object
    predicted_release_object = suffix_correction @ mapped_release_object
    release_position_error = float(
        np.linalg.norm(predicted_release_object[:3, 3] - release_object_target[:3, 3])
    )
    if release_position_error > 1e-3:
        raise RuntimeError(
            f"predicted banana release position error {release_position_error:.6f} m "
            "exceeds 0.001 m"
        )
    transfer_poses, transfer_seconds = core._cartesian_approach_trajectory(
        apex_pose,
        target_eef,
        linear_speed_mps=linear_speed_mps,
        angular_speed_rps=1.0,
        min_seconds=min_seconds,
        rate_hz=rate_hz,
    )
    if transfer_seconds == 0.0:
        return DishInterleave(
            trajectory=trajectory,
            apex_plan_frame=apex_plan_frame,
            target_eef_robot=target_eef,
            suffix_correction=suffix_correction,
            shifted_apex_object_robot=shifted_apex_object,
            release_object_target_robot=release_object_target,
            predicted_release_object_robot=predicted_release_object,
            release_position_error_m=release_position_error,
            transfer_frame_count=0,
            transfer_seconds=0.0,
        )

    insert_at = apex_plan_frame + 1
    transfer_count = len(transfer_poses)
    transfer_times = trajectory.times[apex_plan_frame] + np.linspace(
        transfer_seconds / transfer_count,
        transfer_seconds,
        transfer_count,
        dtype=np.float64,
    )
    held_hand = np.repeat(
        trajectory.hand_actions[apex_plan_frame][None], transfer_count, axis=0
    )
    corrected_suffix = suffix_correction[None] @ trajectory.arm_poses[insert_at:]
    interleaved = core.ReplayTrajectory(
        arm_poses=np.concatenate(
            (
                trajectory.arm_poses[:insert_at],
                transfer_poses,
                corrected_suffix,
            ),
            axis=0,
        ),
        hand_actions=np.concatenate(
            (
                trajectory.hand_actions[:insert_at],
                held_hand,
                trajectory.hand_actions[insert_at:],
            ),
            axis=0,
        ),
        times=np.concatenate(
            (
                trajectory.times[:insert_at],
                transfer_times,
                trajectory.times[insert_at:] + transfer_seconds,
            )
        ),
        transition_frame_count=trajectory.transition_frame_count,
        transition_seconds=trajectory.transition_seconds,
    )
    if np.any(np.diff(interleaved.times) < 0):
        raise RuntimeError("dish-interleaved replay timestamps are not monotonic")
    return DishInterleave(
        trajectory=interleaved,
        apex_plan_frame=apex_plan_frame,
        target_eef_robot=target_eef,
        suffix_correction=suffix_correction,
        shifted_apex_object_robot=shifted_apex_object,
        release_object_target_robot=release_object_target,
        predicted_release_object_robot=predicted_release_object,
        release_position_error_m=release_position_error,
        transfer_frame_count=transfer_count,
        transfer_seconds=float(transfer_seconds),
    )


def _check_camera_arm_sync(
    *,
    label: str,
    camera_timestamp: float,
    arm_timestamp: float,
    execute: bool,
) -> float:
    error = abs(float(arm_timestamp) - float(camera_timestamp))
    if error > 0.010:
        print(f"[sync] warning: {label} camera-arm error is {error * 1000.0:.2f} ms")
    if execute and error > 0.050:
        raise RuntimeError(
            f"refusing execution: {label} camera-arm sync error "
            f"{error * 1000.0:.2f} ms exceeds 50 ms"
        )
    return error


def _execution_confirmed(args: argparse.Namespace) -> bool:
    """Return whether a planned trajectory should be sent to the robot."""

    if getattr(args, "auto_execute", False):
        print("[execute] auto-execute enabled; starting robot trajectory.")
        return True
    answer = input("Execute the robot trajectory? [y/N]: ").strip().lower()
    return answer == "y"


def _load_eef_apex_annotations(args: argparse.Namespace) -> tuple[Path, dict]:
    configured_path = getattr(args, "eef_apex_annotations", None)
    path = (
        Path(configured_path).expanduser()
        if configured_path is not None
        else Path(args.episode_root).expanduser().parent
        / DEFAULT_EEF_APEX_ANNOTATION_NAME
    )
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise FileNotFoundError(f"EEF apex annotation file not found: {path}")
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"could not read EEF apex annotations {path}: {exc}") from exc
    annotations = payload.get("annotations")
    if not isinstance(annotations, dict):
        raise ValueError(f"EEF apex annotation file has no annotations object: {path}")
    return path, annotations


def _episode_eef_apex(
    annotations: dict,
    *,
    object_name: str,
    episode: core.Episode,
) -> dict:
    object_annotations = annotations.get(object_name)
    annotation = (
        object_annotations.get(episode.root.name)
        if isinstance(object_annotations, dict)
        else None
    )
    if not isinstance(annotation, dict):
        raise ValueError(
            f"no EEF apex annotation for {object_name}/{episode.root.name}"
        )
    frame = int(annotation.get("arm_frame_index", -1))
    if not 0 <= frame < len(episode.arm_poses):
        raise ValueError(
            f"EEF apex arm frame {frame} is outside {object_name}/"
            f"{episode.root.name} trajectory of length {len(episode.arm_poses)}"
        )
    return annotation


@lru_cache(maxsize=4)
def _load_fingertip_fk_urdf(robot_urdf: str):
    import yourdfpy

    return yourdfpy.URDF.load(
        robot_urdf,
        build_scene_graph=True,
        load_meshes=False,
    )


def allegro_thumb_index_midpoint_eef(
    hand_action: np.ndarray,
    *,
    robot_urdf: Path,
) -> np.ndarray:
    """Return the thumb/index tip-link midpoint expressed in xArm link6."""

    urdf_path = str(Path(robot_urdf).expanduser().resolve())
    urdf = _load_fingertip_fk_urdf(urdf_path)
    joint_names = tuple(urdf.actuated_joint_names)
    if len(joint_names) != 22 or tuple(joint_names[:6]) != tuple(
        f"joint{i}" for i in range(1, 7)
    ):
        raise ValueError(
            "fingertip FK requires the 6-axis xArm + 16-axis Allegro V5 URDF"
        )
    hand_joint_names = joint_names[6:]
    joint_limits = {
        joint.name: (joint.limit.lower, joint.limit.upper)
        for joint in urdf.actuated_joints
        if joint.limit is not None
    }
    hand_qpos = core._allegro_v5_preview_qpos(
        np.asarray(hand_action, dtype=np.float64).reshape(1, 16),
        urdf_hand_joint_names=hand_joint_names,
        joint_limits=joint_limits,
    )[0]
    urdf.update_cfg(
        {
            name: value
            for name, value in zip(
                joint_names,
                np.concatenate((np.zeros(6, dtype=np.float64), hand_qpos)),
            )
        }
    )
    index_tip_eef = urdf.get_transform(ALLEGRO_V5_INDEX_TIP_LINK, "link6")[
        :3, 3
    ]
    thumb_tip_eef = urdf.get_transform(ALLEGRO_V5_THUMB_TIP_LINK, "link6")[
        :3, 3
    ]
    midpoint = (index_tip_eef + thumb_tip_eef) * 0.5
    if midpoint.shape != (3,) or not np.all(np.isfinite(midpoint)):
        raise ValueError(f"invalid Allegro fingertip midpoint from {robot_urdf}")
    return midpoint


def append_box_pink_transfer_and_release(
    trajectory: core.ReplayTrajectory,
    *,
    box_point_robot: np.ndarray,
    fingertip_midpoint_eef: np.ndarray,
    clearance_m: float,
    linear_speed_mps: float,
    max_translation_m: float,
    min_seconds: float,
    rate_hz: float,
    release_seconds: float,
) -> BoxPinkTransfer:
    """Move the fingertip midpoint to box x/y at a fixed robot-frame z, then open."""

    if (
        clearance_m < 0
        or linear_speed_mps <= 0
        or max_translation_m <= 0
        or min_seconds < 0
        or rate_hz <= 0
        or release_seconds <= 0
    ):
        raise ValueError(
            "box clearance/min-duration must be non-negative and transfer speed/"
            "distance/rate/release duration must be positive"
        )
    box_point = np.asarray(box_point_robot, dtype=np.float64).reshape(-1)
    if box_point.shape != (3,) or not np.all(np.isfinite(box_point)):
        raise ValueError(f"box-pink point must be a finite xyz vector, got {box_point_robot}")
    midpoint_eef = np.asarray(fingertip_midpoint_eef, dtype=np.float64).reshape(-1)
    if midpoint_eef.shape != (3,) or not np.all(np.isfinite(midpoint_eef)):
        raise ValueError(
            "thumb-index fingertip midpoint must be a finite EEF-frame xyz vector"
        )

    current_eef = core._as_transform(
        trajectory.arm_poses[-1], label="EEF pose at annotated apex"
    )
    target_eef = current_eef.copy()
    target_midpoint = box_point.copy()
    # Triangulation contributes only the horizontal target.  The configured
    # release height is an absolute robot-frame z, independent of box_point.z.
    target_midpoint[2] = clearance_m
    target_eef[:3, 3] = (
        target_midpoint - target_eef[:3, :3] @ midpoint_eef
    )
    vertical_target = current_eef.copy()
    vertical_target[2, 3] = target_eef[2, 3]
    vertical_distance = abs(float(target_eef[2, 3] - current_eef[2, 3]))
    horizontal_distance = float(
        np.linalg.norm(target_eef[:2, 3] - current_eef[:2, 3])
    )
    transfer_distance = vertical_distance + horizontal_distance
    if transfer_distance > max_translation_m:
        raise RuntimeError(
            f"box-pink axis-aligned path length {transfer_distance:.3f} m exceeds "
            f"configured maximum {max_translation_m:.3f} m"
        )
    segment_poses: list[np.ndarray] = []
    segment_times: list[np.ndarray] = []
    elapsed = float(trajectory.times[-1])

    def append_segment(
        start_pose: np.ndarray,
        end_pose: np.ndarray,
        distance: float,
    ) -> tuple[int, float]:
        nonlocal elapsed
        if distance <= 1.0e-9:
            return 0, 0.0
        poses, seconds = core._cartesian_approach_trajectory(
            start_pose,
            end_pose,
            linear_speed_mps=linear_speed_mps,
            angular_speed_rps=1.0,
            min_seconds=min_seconds,
            rate_hz=rate_hz,
        )
        count = len(poses)
        times = elapsed + np.linspace(
            seconds / count,
            seconds,
            count,
            dtype=np.float64,
        )
        segment_poses.append(poses)
        segment_times.append(times)
        elapsed = float(times[-1])
        return count, float(seconds)

    vertical_count, vertical_seconds = append_segment(
        current_eef,
        vertical_target,
        vertical_distance,
    )
    horizontal_count, horizontal_seconds = append_segment(
        vertical_target,
        target_eef,
        horizontal_distance,
    )
    transfer_count = vertical_count + horizontal_count
    transfer_seconds = vertical_seconds + horizontal_seconds
    transfer_poses = (
        np.concatenate(segment_poses, axis=0)
        if segment_poses
        else np.empty((0, 4, 4), dtype=np.float64)
    )
    transfer_times = (
        np.concatenate(segment_times)
        if segment_times
        else np.empty(0, dtype=np.float64)
    )
    transfer_hands = np.repeat(
        trajectory.hand_actions[-1][None], transfer_count, axis=0
    )

    release_count = max(1, int(np.ceil(release_seconds * rate_hz)))
    release_alphas = np.linspace(
        0.0, 1.0, release_count + 1, dtype=np.float64
    )[1:]
    closed_hand = trajectory.hand_actions[-1]
    open_hand = np.zeros(16, dtype=np.float64)
    release_hands = (
        closed_hand[None] * (1.0 - release_alphas[:, None])
        + open_hand[None] * release_alphas[:, None]
    )
    release_poses = np.repeat(target_eef[None], release_count, axis=0)
    release_times = elapsed + np.linspace(
        release_seconds / release_count,
        release_seconds,
        release_count,
        dtype=np.float64,
    )

    combined = core.ReplayTrajectory(
        arm_poses=np.concatenate(
            (trajectory.arm_poses, transfer_poses, release_poses), axis=0
        ),
        hand_actions=np.concatenate(
            (trajectory.hand_actions, transfer_hands, release_hands), axis=0
        ),
        times=np.concatenate((trajectory.times, transfer_times, release_times)),
        transition_frame_count=trajectory.transition_frame_count,
        transition_seconds=trajectory.transition_seconds,
    )
    return BoxPinkTransfer(
        trajectory=combined,
        target_eef_robot=target_eef,
        fingertip_midpoint_eef=midpoint_eef,
        target_fingertip_midpoint_robot=target_midpoint,
        transfer_frame_count=transfer_count,
        transfer_seconds=float(transfer_seconds),
        vertical_frame_count=vertical_count,
        vertical_seconds=vertical_seconds,
        horizontal_frame_count=horizontal_count,
        horizontal_seconds=horizontal_seconds,
        release_frame_count=release_count,
        release_seconds=float(release_seconds),
    )


def replay_closest_episode_naive(
    args: argparse.Namespace,
    *,
    current_object_robot: np.ndarray,
    current_c2r: np.ndarray,
    arm: Any | None = None,
    hand: Any | None = None,
    episodes: Iterable[core.Episode] | None = None,
) -> Path:
    """Replay the closest episode in full without annotations or dish transfer."""

    if (arm is None) != (hand is None):
        raise ValueError("arm and hand controllers must be supplied together")
    candidate_episodes = (
        load_candidate_episodes(args.episode_root, args.candidate_episodes)
        if episodes is None
        else list(episodes)
    )
    ranked = rank_by_object_pose(
        current_object_robot,
        candidate_episodes,
        translation_scale_m=args.retrieval_translation_scale_m,
        rotation_scale_rad=args.retrieval_rotation_scale_rad,
    )
    print(f"[retrieval] candidate ranking by frame-0 {args.object} pose:")
    for rank, match in enumerate(ranked, start=1):
        print(
            f"  {rank}. episode {match.episode.root.name}: "
            f"translation={match.translation_error_m:.4f} m, "
            f"rotation={np.rad2deg(match.rotation_error_rad):.1f} deg, "
            f"score={match.score:.3f}"
        )

    selected = ranked[0]
    episode = selected.episode
    replay_arm_poses = core.relative_arm_actions(
        selected.source_object_robot, current_object_robot, episode.arm_poses
    )
    replay_hand_actions = core._zoh_resample(
        episode.hand_times, episode.hand_commands, episode.arm_times
    )
    if arm is None:
        live_arm_pose, live_arm_qpos, live_hand_qpos = _read_live_robot_state()
    else:
        live_arm_pose, live_arm_qpos, live_hand_qpos = core._read_live_robot_state(
            arm, hand
        )
    trajectory = core._compose_replay_trajectory(
        args,
        live_arm_pose=live_arm_pose,
        live_hand_qpos=live_hand_qpos,
        episode_arm_poses=replay_arm_poses,
        episode_hand_actions=replay_hand_actions,
        episode_arm_times=episode.arm_times,
    )

    output_dir = Path(args.replay_output_dir).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)
    plan_path = output_dir / "naive_relative_replay_plan.npz"
    object_delta = current_object_robot @ np.linalg.inv(selected.source_object_robot)
    np.savez_compressed(
        plan_path,
        arm_action=trajectory.arm_poses,
        arm_time=trajectory.times,
        hand_action=trajectory.hand_actions,
        episode_arm_action=replay_arm_poses,
        episode_arm_time=episode.arm_times,
        episode_hand_action=replay_hand_actions,
        transition_frame_count=trajectory.transition_frame_count,
        transition_seconds=trajectory.transition_seconds,
        replay_speed_scale=args.rate_scale,
        episode_start_frame_in_plan=trajectory.transition_frame_count - 1,
        selected_episode=episode.root.name,
        selected_frame=0,
        source_object_robot=selected.source_object_robot,
        current_object_robot=current_object_robot,
        object_delta=object_delta,
        source_c2r=episode.source_c2r,
        current_c2r=current_c2r,
        retrieval_translation_error_m=selected.translation_error_m,
        retrieval_rotation_error_rad=selected.rotation_error_rad,
        retrieval_score=selected.score,
        candidate_episode=np.asarray([match.episode.root.name for match in ranked]),
        candidate_translation_error_m=np.asarray(
            [match.translation_error_m for match in ranked]
        ),
        candidate_rotation_error_rad=np.asarray(
            [match.rotation_error_rad for match in ranked]
        ),
        candidate_score=np.asarray([match.score for match in ranked]),
        live_wrist_robot=live_arm_pose,
    )
    print(f"[retrieval] selected episode {episode.root.name} from arm frame 0")
    print(f"[plan] saved {plan_path}")
    print(
        f"[plan] current state -> transformed episode frame 0: "
        f"{trajectory.transition_frame_count} frames over "
        f"{trajectory.transition_seconds:.2f}s; then replaying all "
        f"{len(replay_arm_poses)} source frames at {args.rate_scale:.2f}x"
    )

    preview_match = core.EpisodeMatch(
        episode=episode,
        frame_index=0,
        grasp_frame_index=int(
            np.argmin(
                np.linalg.norm(
                    (
                        np.linalg.inv(selected.source_object_robot)[None]
                        @ episode.arm_poses
                    )[:, :3, 3],
                    axis=1,
                )
            )
        ),
        wrist_object_distance_m=0.0,
        distance_delta_m=0.0,
        position_error_m=selected.translation_error_m,
        rotation_error_rad=selected.rotation_error_rad,
        score=selected.score,
    )
    if args.replay_preview:
        core._preview_replay(
            args,
            preview_match,
            trajectory.arm_poses,
            trajectory.hand_actions,
            trajectory.times,
            trajectory.transition_frame_count,
            current_object_robot,
            output_dir,
            live_arm_pose=live_arm_pose,
            live_arm_qpos=live_arm_qpos,
            live_hand_qpos=live_hand_qpos,
        )
    if not args.execute:
        print("[plan] complete; no robot or hand motion commands were sent.")
        return plan_path
    if not _execution_confirmed(args):
        print("[execute] cancelled; no robot commands sent.")
        return plan_path
    core._execute(
        args,
        trajectory.arm_poses,
        trajectory.hand_actions,
        trajectory.times,
        trajectory.transition_frame_count,
        arm=arm,
        hand=hand,
    )
    return plan_path


def replay_closest_episode_into_box_pink(
    args: argparse.Namespace,
    *,
    current_object_robot: np.ndarray,
    current_c2r: np.ndarray,
    box_point_robot: np.ndarray,
    arm: Any | None = None,
    hand: Any | None = None,
    episodes: Iterable[core.Episode] | None = None,
) -> Path:
    """Replay through the EEF apex, move above box-pink, and release."""

    if (arm is None) != (hand is None):
        raise ValueError("arm and hand controllers must be supplied together")
    candidate_episodes = (
        load_candidate_episodes(args.episode_root, args.candidate_episodes)
        if episodes is None
        else list(episodes)
    )
    annotation_path, annotations = _load_eef_apex_annotations(args)
    replayable_episodes: list[core.Episode] = []
    apex_by_episode: dict[str, dict] = {}
    for episode in candidate_episodes:
        try:
            annotation = _episode_eef_apex(
                annotations,
                object_name=args.object,
                episode=episode,
            )
        except ValueError as exc:
            print(f"[apex] skipping episode {episode.root.name}: {exc}")
            continue
        replayable_episodes.append(episode)
        apex_by_episode[episode.root.name] = annotation
    if not replayable_episodes:
        raise RuntimeError(
            f"no replayable {args.object} episode has a valid EEF apex annotation"
        )

    ranked = rank_by_object_pose(
        current_object_robot,
        replayable_episodes,
        translation_scale_m=args.retrieval_translation_scale_m,
        rotation_scale_rad=args.retrieval_rotation_scale_rad,
    )
    print(f"[retrieval] candidate ranking by frame-0 {args.object} pose:")
    for rank, match in enumerate(ranked, start=1):
        print(
            f"  {rank}. episode {match.episode.root.name}: "
            f"translation={match.translation_error_m:.4f} m, "
            f"rotation={np.rad2deg(match.rotation_error_rad):.1f} deg, "
            f"score={match.score:.3f}"
        )

    selected = ranked[0]
    episode = selected.episode
    apex_annotation = apex_by_episode[episode.root.name]
    apex_arm_frame = int(apex_annotation["arm_frame_index"])
    apex_review = str(apex_annotation.get("manual_review", "unknown"))
    if apex_review != "not_required":
        print(
            f"[apex] warning: {args.object}/{episode.root.name} annotation "
            f"manual_review={apex_review}"
        )
    transformed_arm_poses = core.relative_arm_actions(
        selected.source_object_robot, current_object_robot, episode.arm_poses
    )
    transformed_hand_actions = core._zoh_resample(
        episode.hand_times, episode.hand_commands, episode.arm_times
    )
    replay_arm_poses = transformed_arm_poses[: apex_arm_frame + 1]
    replay_hand_actions = transformed_hand_actions[: apex_arm_frame + 1]
    replay_arm_times = episode.arm_times[: apex_arm_frame + 1]

    if arm is None:
        live_arm_pose, live_arm_qpos, live_hand_qpos = _read_live_robot_state()
    else:
        live_arm_pose, live_arm_qpos, live_hand_qpos = core._read_live_robot_state(
            arm, hand
        )
    apex_trajectory = core._compose_replay_trajectory(
        args,
        live_arm_pose=live_arm_pose,
        live_hand_qpos=live_hand_qpos,
        episode_arm_poses=replay_arm_poses,
        episode_hand_actions=replay_hand_actions,
        episode_arm_times=replay_arm_times,
    )
    fingertip_midpoint_eef = allegro_thumb_index_midpoint_eef(
        replay_hand_actions[-1],
        robot_urdf=args.robot_urdf,
    )
    box_transfer = append_box_pink_transfer_and_release(
        apex_trajectory,
        box_point_robot=box_point_robot,
        fingertip_midpoint_eef=fingertip_midpoint_eef,
        clearance_m=args.box_pink_clearance_m,
        linear_speed_mps=args.dish_transfer_linear_speed_mps,
        max_translation_m=args.dish_transfer_max_distance_m,
        min_seconds=args.dish_transfer_min_seconds,
        rate_hz=args.dish_transfer_rate_hz,
        release_seconds=args.hand_open_seconds,
    )
    trajectory = box_transfer.trajectory

    output_dir = Path(args.replay_output_dir).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)
    plan_path = output_dir / "box_pink_replay_plan.npz"
    object_delta = current_object_robot @ np.linalg.inv(selected.source_object_robot)
    box_point = np.asarray(box_point_robot, dtype=np.float64).reshape(3)
    np.savez_compressed(
        plan_path,
        arm_action=trajectory.arm_poses,
        arm_time=trajectory.times,
        hand_action=trajectory.hand_actions,
        episode_arm_action=replay_arm_poses,
        episode_arm_time=replay_arm_times,
        episode_hand_action=replay_hand_actions,
        transition_frame_count=trajectory.transition_frame_count,
        transition_seconds=trajectory.transition_seconds,
        episode_start_frame_in_plan=trajectory.transition_frame_count - 1,
        selected_episode=episode.root.name,
        selected_frame=0,
        apex_arm_frame=apex_arm_frame,
        apex_annotation_path=str(annotation_path),
        apex_manual_review=apex_review,
        source_object_robot=selected.source_object_robot,
        current_object_robot=current_object_robot,
        object_delta=object_delta,
        source_c2r=episode.source_c2r,
        current_c2r=current_c2r,
        box_pink_point_robot=box_point,
        box_pink_clearance_m=args.box_pink_clearance_m,
        box_pink_target_eef_robot=box_transfer.target_eef_robot,
        apex_fingertip_midpoint_eef=box_transfer.fingertip_midpoint_eef,
        box_pink_target_fingertip_midpoint_robot=(
            box_transfer.target_fingertip_midpoint_robot
        ),
        box_transfer_frame_count=box_transfer.transfer_frame_count,
        box_transfer_seconds=box_transfer.transfer_seconds,
        box_vertical_frame_count=box_transfer.vertical_frame_count,
        box_vertical_seconds=box_transfer.vertical_seconds,
        box_horizontal_frame_count=box_transfer.horizontal_frame_count,
        box_horizontal_seconds=box_transfer.horizontal_seconds,
        hand_open_frame_count=box_transfer.release_frame_count,
        hand_open_seconds=box_transfer.release_seconds,
        initial_arm_qpos=live_arm_qpos,
        return_joint_speed_rps=args.return_joint_speed_rps,
        return_min_seconds=args.return_min_seconds,
        return_rate_hz=args.return_rate_hz,
        retrieval_translation_error_m=selected.translation_error_m,
        retrieval_rotation_error_rad=selected.rotation_error_rad,
        retrieval_score=selected.score,
        candidate_episode=np.asarray([match.episode.root.name for match in ranked]),
        candidate_translation_error_m=np.asarray(
            [match.translation_error_m for match in ranked]
        ),
        candidate_rotation_error_rad=np.asarray(
            [match.rotation_error_rad for match in ranked]
        ),
        candidate_score=np.asarray([match.score for match in ranked]),
        live_wrist_robot=live_arm_pose,
    )
    print(
        f"[retrieval] selected episode {episode.root.name}; replaying arm frames "
        f"0..{apex_arm_frame} (annotated EEF apex)"
    )
    print(f"[plan] saved {plan_path}")
    print(
        f"[box-pink] thumb-index midpoint target xyz "
        f"{np.round(box_transfer.target_fingertip_midpoint_robot, 4).tolist()} "
        f"-> compensated EEF xyz "
        f"{np.round(box_transfer.target_eef_robot[:3, 3], 4).tolist()}: "
        f"vertical {box_transfer.vertical_frame_count} frames/"
        f"{box_transfer.vertical_seconds:.2f}s, then XY "
        f"{box_transfer.horizontal_frame_count} frames/"
        f"{box_transfer.horizontal_seconds:.2f}s; then opening the hand over "
        f"{box_transfer.release_seconds:.2f}s"
    )

    preview_match = core.EpisodeMatch(
        episode=episode,
        frame_index=0,
        grasp_frame_index=min(apex_arm_frame, len(episode.arm_poses) - 1),
        wrist_object_distance_m=0.0,
        distance_delta_m=0.0,
        position_error_m=selected.translation_error_m,
        rotation_error_rad=selected.rotation_error_rad,
        score=selected.score,
    )
    if args.replay_preview:
        core._preview_replay(
            args,
            preview_match,
            trajectory.arm_poses,
            trajectory.hand_actions,
            trajectory.times,
            trajectory.transition_frame_count,
            current_object_robot,
            output_dir,
            live_arm_pose=live_arm_pose,
            live_arm_qpos=live_arm_qpos,
            live_hand_qpos=live_hand_qpos,
            return_arm_qpos=live_arm_qpos,
        )
    if not args.execute:
        print("[plan] complete; no robot or hand motion commands were sent.")
        return plan_path
    if not _execution_confirmed(args):
        print("[execute] cancelled; no robot commands sent.")
        return plan_path
    core._execute(
        args,
        trajectory.arm_poses,
        trajectory.hand_actions,
        trajectory.times,
        trajectory.transition_frame_count,
        arm=arm,
        hand=hand,
        return_arm_qpos=live_arm_qpos,
    )
    return plan_path


def replay_closest_episode(
    args: argparse.Namespace,
    *,
    current_object_robot: np.ndarray,
    current_c2r: np.ndarray,
    dish_point_robot: np.ndarray | None = None,
    arm: Any | None = None,
    hand: Any | None = None,
    episodes: Iterable[core.Episode] | None = None,
) -> Path:
    """Build, preview, and optionally execute an annotated banana replay."""

    if (arm is None) != (hand is None):
        raise ValueError("arm and hand controllers must be supplied together")

    if episodes is None:
        candidate_episodes = load_candidate_episodes(
            args.episode_root, args.candidate_episodes
        )
    else:
        candidate_episodes = list(episodes)
    ranked = rank_by_object_pose(
        current_object_robot,
        candidate_episodes,
        translation_scale_m=args.retrieval_translation_scale_m,
        rotation_scale_rad=args.retrieval_rotation_scale_rad,
    )
    selected = ranked[0]
    print("[retrieval] candidate ranking by frame-0 banana pose:")
    for rank, match in enumerate(ranked, start=1):
        print(
            f"  {rank}. episode {match.episode.root.name}: "
            f"translation={match.translation_error_m:.4f} m, "
            f"rotation={np.rad2deg(match.rotation_error_rad):.1f} deg, "
            f"score={match.score:.3f}"
        )

    if dish_point_robot is None:
        raise RuntimeError(
            "dish center triangulation is required for the apex interruption"
        )

    episode = selected.episode
    episode_arm_poses = core.relative_arm_actions(
        selected.source_object_robot, current_object_robot, episode.arm_poses
    )
    episode_hand_actions = core._zoh_resample(
        episode.hand_times, episode.hand_commands, episode.arm_times
    )
    _, start_annotations = apex_core._load_start_annotations(args.episode_root)
    _, apex_annotations = apex_core._load_apex_annotations(args.episode_root)
    _, release_annotations = apex_core._load_release_annotations(args.episode_root)
    episode_id = int(episode.root.name)
    if episode_id not in start_annotations:
        raise ValueError(
            f"selected episode {episode_id} has no start-position annotation"
        )
    if episode_id not in apex_annotations:
        raise ValueError(f"selected episode {episode_id} has no apex annotation")
    if episode_id not in release_annotations:
        raise ValueError(
            f"selected episode {episode_id} has no right-before-release annotation"
        )
    start_video_frame = start_annotations[episode_id]
    apex_video_frame = apex_annotations[episode_id]
    release_video_frame = release_annotations[episode_id]
    if not start_video_frame < apex_video_frame < release_video_frame:
        raise ValueError(
            f"selected episode {episode_id} annotation frames must satisfy start < "
            f"apex < release, got {start_video_frame}, {apex_video_frame}, "
            f"{release_video_frame}"
        )
    start_arm_frame, start_camera_timestamp = apex_core._map_video_frame_to_arm_index(
        episode,
        start_video_frame,
        label="start",
    )
    apex_arm_frame, apex_camera_timestamp = apex_core._map_video_frame_to_arm_index(
        episode,
        apex_video_frame,
        label="apex",
    )
    release_arm_frame, release_camera_timestamp = (
        apex_core._map_video_frame_to_arm_index(
            episode,
            release_video_frame,
            label="right-before-release",
        )
    )
    if not start_arm_frame <= apex_arm_frame < release_arm_frame:
        raise ValueError(
            f"selected episode {episode_id} synchronized arm frames must satisfy "
            f"start <= apex < release, got {start_arm_frame}, {apex_arm_frame}, "
            f"{release_arm_frame}"
        )
    start_arm_timestamp = float(episode.arm_times[start_arm_frame])
    apex_arm_timestamp = float(episode.arm_times[apex_arm_frame])
    release_arm_timestamp = float(episode.arm_times[release_arm_frame])
    start_sync_error = _check_camera_arm_sync(
        label="start",
        camera_timestamp=start_camera_timestamp,
        arm_timestamp=start_arm_timestamp,
        execute=args.execute,
    )
    apex_sync_error = _check_camera_arm_sync(
        label="apex",
        camera_timestamp=apex_camera_timestamp,
        arm_timestamp=apex_arm_timestamp,
        execute=args.execute,
    )
    release_sync_error = _check_camera_arm_sync(
        label="right-before-release",
        camera_timestamp=release_camera_timestamp,
        arm_timestamp=release_arm_timestamp,
        execute=args.execute,
    )
    tracking_path, tracked_world_poses = load_gotrack_world_poses(
        episode.root,
        (apex_video_frame, release_video_frame),
    )
    source_apex_object_world = tracked_world_poses[apex_video_frame]
    source_release_object_world = tracked_world_poses[release_video_frame]
    robot_from_source_world = np.linalg.inv(episode.source_c2r)
    source_apex_object_robot = robot_from_source_world @ source_apex_object_world
    source_release_object_robot = robot_from_source_world @ source_release_object_world
    object_delta = current_object_robot @ np.linalg.inv(selected.source_object_robot)
    mapped_apex_object_robot = object_delta @ source_apex_object_robot
    mapped_release_object_robot = object_delta @ source_release_object_robot

    if arm is None:
        live_arm_pose, live_arm_qpos, live_hand_qpos = _read_live_robot_state()
    else:
        live_arm_pose, live_arm_qpos, live_hand_qpos = core._read_live_robot_state(
            arm, hand
        )
    replay_arm_poses = episode_arm_poses[start_arm_frame:]
    replay_hand_actions = episode_hand_actions[start_arm_frame:]
    replay_arm_times = episode.arm_times[start_arm_frame:]
    trajectory_before_dish = core._compose_replay_trajectory(
        args,
        live_arm_pose=live_arm_pose,
        live_hand_qpos=live_hand_qpos,
        episode_arm_poses=replay_arm_poses,
        episode_hand_actions=replay_hand_actions,
        episode_arm_times=replay_arm_times,
    )
    apex_replay_frame = apex_arm_frame - start_arm_frame
    interleave = interleave_dish_transfer(
        trajectory_before_dish,
        episode_apex_frame=apex_replay_frame,
        mapped_apex_object_robot=mapped_apex_object_robot,
        mapped_release_object_robot=mapped_release_object_robot,
        dish_point_robot=dish_point_robot,
        clearance_m=args.dish_clearance_m,
        linear_speed_mps=args.dish_transfer_linear_speed_mps,
        max_translation_m=args.dish_transfer_max_distance_m,
        min_seconds=args.dish_transfer_min_seconds,
        rate_hz=args.dish_transfer_rate_hz,
    )
    trajectory = interleave.trajectory
    output_dir = Path(args.replay_output_dir).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)
    plan_path = output_dir / "relative_replay_plan.npz"
    dish = (
        np.asarray(dish_point_robot, dtype=np.float64).reshape(3)
        if dish_point_robot is not None
        else np.full(3, np.nan, dtype=np.float64)
    )
    np.savez_compressed(
        plan_path,
        arm_action=trajectory.arm_poses,
        arm_time=trajectory.times,
        hand_action=trajectory.hand_actions,
        episode_arm_action=replay_arm_poses,
        episode_arm_time=replay_arm_times,
        episode_hand_action=replay_hand_actions,
        transition_frame_count=trajectory.transition_frame_count,
        transition_seconds=trajectory.transition_seconds,
        episode_start_frame_in_plan=trajectory.transition_frame_count - 1,
        selected_episode=episode.root.name,
        selected_frame=start_arm_frame,
        source_object_robot=selected.source_object_robot,
        current_object_robot=current_object_robot,
        object_delta=object_delta,
        source_c2r=episode.source_c2r,
        current_c2r=current_c2r,
        dish_point_robot=dish,
        dish_clearance_m=args.dish_clearance_m,
        start_video_frame=start_video_frame,
        start_camera_timestamp=start_camera_timestamp,
        start_arm_frame=start_arm_frame,
        start_arm_timestamp=start_arm_timestamp,
        start_sync_error_seconds=start_sync_error,
        apex_video_frame=apex_video_frame,
        apex_camera_timestamp=apex_camera_timestamp,
        apex_arm_frame=apex_arm_frame,
        apex_arm_timestamp=apex_arm_timestamp,
        apex_sync_error_seconds=apex_sync_error,
        apex_frame_in_replay=apex_replay_frame,
        apex_plan_frame=interleave.apex_plan_frame,
        release_video_frame=release_video_frame,
        release_camera_timestamp=release_camera_timestamp,
        release_arm_frame=release_arm_frame,
        release_arm_timestamp=release_arm_timestamp,
        release_sync_error_seconds=release_sync_error,
        release_frame_in_replay=release_arm_frame - start_arm_frame,
        gotrack_pose_records_path=str(tracking_path),
        source_apex_object_world=source_apex_object_world,
        source_apex_object_robot=source_apex_object_robot,
        mapped_apex_object_robot=mapped_apex_object_robot,
        shifted_apex_object_robot=interleave.shifted_apex_object_robot,
        source_release_object_world=source_release_object_world,
        source_release_object_robot=source_release_object_robot,
        mapped_release_object_robot=mapped_release_object_robot,
        release_object_target_robot=interleave.release_object_target_robot,
        predicted_release_object_robot=interleave.predicted_release_object_robot,
        release_position_error_m=interleave.release_position_error_m,
        dish_target_eef_robot=interleave.target_eef_robot,
        dish_suffix_correction=interleave.suffix_correction,
        dish_transfer_frame_count=interleave.transfer_frame_count,
        dish_transfer_seconds=interleave.transfer_seconds,
        retrieval_translation_error_m=selected.translation_error_m,
        retrieval_rotation_error_rad=selected.rotation_error_rad,
        retrieval_score=selected.score,
        candidate_episode=np.asarray([match.episode.root.name for match in ranked]),
        candidate_translation_error_m=np.asarray(
            [match.translation_error_m for match in ranked]
        ),
        candidate_rotation_error_rad=np.asarray(
            [match.rotation_error_rad for match in ranked]
        ),
        candidate_score=np.asarray([match.score for match in ranked]),
        live_wrist_robot=live_arm_pose,
    )
    print(
        f"[retrieval] selected episode {episode.root.name} from annotated arm "
        f"frame {start_arm_frame}"
    )
    print(f"[plan] saved {plan_path}")
    print(
        f"[plan] current state -> episode arm frame {start_arm_frame}: "
        f"{trajectory.transition_frame_count} frames over "
        f"{trajectory.transition_seconds:.2f}s"
    )
    print(
        f"[sync] start video frame {start_video_frame} -> camera timestamp "
        f"{start_camera_timestamp:.6f} -> arm frame {start_arm_frame} "
        f"(error {start_sync_error * 1000.0:.2f} ms)"
    )
    print(
        f"[sync] apex video frame {apex_video_frame} -> camera timestamp "
        f"{apex_camera_timestamp:.6f} -> arm frame {apex_arm_frame} "
        f"(error {apex_sync_error * 1000.0:.2f} ms)"
    )
    print(
        f"[sync] release video frame {release_video_frame} -> camera timestamp "
        f"{release_camera_timestamp:.6f} -> arm frame {release_arm_frame} "
        f"(error {release_sync_error * 1000.0:.2f} ms)"
    )
    print(
        f"[dish] apex plan frame {interleave.apex_plan_frame} -> "
        f"EEF target xyz "
        f"{np.round(interleave.target_eef_robot[:3, 3], 4).tolist()}: "
        f"{interleave.transfer_frame_count} frames over "
        f"{interleave.transfer_seconds:.2f}s; predicted banana release xyz "
        f"{np.round(interleave.predicted_release_object_robot[:3, 3], 4).tolist()} "
        f"(error {interleave.release_position_error_m * 1000.0:.3f} mm)"
    )

    preview_match = core.EpisodeMatch(
        episode=episode,
        frame_index=start_arm_frame,
        grasp_frame_index=int(
            np.argmin(
                np.linalg.norm(
                    (
                        np.linalg.inv(selected.source_object_robot)[None]
                        @ episode.arm_poses
                    )[:, :3, 3],
                    axis=1,
                )
            )
        ),
        wrist_object_distance_m=0.0,
        distance_delta_m=0.0,
        position_error_m=selected.translation_error_m,
        rotation_error_rad=selected.rotation_error_rad,
        score=selected.score,
    )
    if args.replay_preview:
        core._preview_replay(
            args,
            preview_match,
            trajectory.arm_poses,
            trajectory.hand_actions,
            trajectory.times,
            trajectory.transition_frame_count,
            current_object_robot,
            output_dir,
            live_arm_pose=live_arm_pose,
            live_arm_qpos=live_arm_qpos,
            live_hand_qpos=live_hand_qpos,
            object_pose_markers={
                "mapped_apex": mapped_apex_object_robot,
                "shifted_apex": interleave.shifted_apex_object_robot,
                "mapped_release": mapped_release_object_robot,
                "release_goal": interleave.release_object_target_robot,
            },
        )
    if not args.execute:
        print("[plan] complete; no robot or hand motion commands were sent.")
        return plan_path
    if not _execution_confirmed(args):
        print("[execute] cancelled; no robot commands sent.")
        return plan_path
    core._execute(
        args,
        trajectory.arm_poses,
        trajectory.hand_actions,
        trajectory.times,
        trajectory.transition_frame_count,
        arm=arm,
        hand=hand,
    )
    return plan_path
