"""Retrieve a banana episode and replay it with a release-relative dish transfer."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from paradex.retargetor.experiment import replay_object_relative as core
from paradex.retargetor.experiment import (
    replay_object_relative_first_frame as apex_core,
)


DEFAULT_EPISODE_ROOT = Path("/home/temp_id/shared_data/capture/0825_test_4/banana")
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
