"""Match and replay an annotated banana grasp from the initial live state.

This entry point deliberately has no VIVE/MANUS teleoperation and no replay
key.  It indexes only ``target_episodes`` from the fixed banana dataset's
``apex_annotations.json``, compares each candidate at episode frame 0, and
replays the selected trajectory from frame 0.  At the selected episode's
annotated apex video frame, the arm and hand hold their pose for one second
before continuing.  Motion is disabled unless ``--execute`` is supplied and
``PLAY`` is confirmed after planning.
"""

from __future__ import annotations

import argparse
from datetime import datetime
import json
from pathlib import Path
import sys

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from paradex.retargetor.experiment import replay_object_relative as core
from paradex.utils.path import shared_dir


OBJECT_NAME = "banana"
ROBOT_NAME = "allegro_v5"
BANANA_DATASET_ROOT = Path(shared_dir) / "capture" / "0825_test_4" / OBJECT_NAME
APEX_PAUSE_SECONDS = 1.0


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mesh-root-dir", type=Path, default=core.DEFAULT_MESH_ROOT)
    parser.add_argument("--robot-urdf", type=Path, default=core.DEFAULT_ROBOT_URDF)
    parser.add_argument(
        "--current-object-pose",
        default=None,
        help="skip capture/RPC and load this pose",
    )
    parser.add_argument(
        "--current-pose-frame", choices=("world", "robot"), default="world"
    )
    parser.add_argument("--current-c2r-path", default=None)
    parser.add_argument("--rcc-entry", default="image_main.py")
    parser.add_argument("--rpc-addr", default="tcp://192.168.0.14:5570")
    parser.add_argument("--rpc-timeout-ms", type=int, default=300000)
    parser.add_argument(
        "--debug-image-mode", choices=("none", "save", "popup"), default="save"
    )
    parser.add_argument("--preview", action="store_true")
    parser.add_argument("--preview-max-frames", type=int, default=150)
    parser.add_argument("--preview-ik-max-nfev", type=int, default=50)
    parser.add_argument("--preview-position-scale", type=float, default=0.05)
    parser.add_argument("--preview-rotation-scale", type=float, default=0.5)
    parser.add_argument("--no-viser-object-align", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--match-position-scale-m", type=float, default=0.05)
    parser.add_argument("--match-rotation-scale-rad", type=float, default=0.5)
    parser.add_argument("--rate-scale", type=float, default=1.0)
    parser.add_argument("--approach-linear-speed-mps", type=float, default=0.05)
    parser.add_argument("--approach-angular-speed-rps", type=float, default=0.5)
    parser.add_argument("--approach-min-seconds", type=float, default=5.0)
    parser.add_argument("--approach-rate-hz", type=float, default=50.0)
    parser.add_argument("--settle-seconds", type=float, default=1.0)
    parser.add_argument(
        "--execute", action="store_true", help="allow real arm/hand commands after PLAY"
    )
    parser.set_defaults(object=OBJECT_NAME, robot=ROBOT_NAME, mesh_name=OBJECT_NAME)
    return parser


def _validate_args(parser: argparse.ArgumentParser, args: argparse.Namespace) -> None:
    positive = {
        "--rate-scale": args.rate_scale,
        "--rpc-timeout-ms": args.rpc_timeout_ms,
        "--preview-max-frames": args.preview_max_frames,
        "--preview-ik-max-nfev": args.preview_ik_max_nfev,
        "--preview-position-scale": args.preview_position_scale,
        "--preview-rotation-scale": args.preview_rotation_scale,
        "--match-position-scale-m": args.match_position_scale_m,
        "--match-rotation-scale-rad": args.match_rotation_scale_rad,
        "--approach-linear-speed-mps": args.approach_linear_speed_mps,
        "--approach-angular-speed-rps": args.approach_angular_speed_rps,
        "--approach-rate-hz": args.approach_rate_hz,
    }
    for name, value in positive.items():
        if value <= 0:
            parser.error(f"{name} must be positive")
    if args.approach_min_seconds < 0:
        parser.error("--approach-min-seconds must be non-negative")


def _read_first_frame_robot_state() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Read one live state without creating teleoperation command controllers."""

    arm_pose, arm_qpos = core._live_xarm_preview_state()
    hand_qpos = core._live_allegro_v5_preview_qpos()
    return arm_pose, arm_qpos, hand_qpos


def _load_frame_annotations(
    dataset_root: Path,
    filename: str,
    *,
    label: str,
) -> tuple[list[int], dict[int, int]]:
    """Load zero-based camera-frame annotations for the target episodes."""

    annotation_path = dataset_root / filename
    payload = json.loads(annotation_path.read_text(encoding="utf-8"))
    target_episodes = payload.get("target_episodes")
    raw_annotations = payload.get("annotations")
    if not isinstance(target_episodes, list) or not target_episodes:
        raise ValueError(f"target_episodes is empty or invalid in {annotation_path}")
    if not isinstance(raw_annotations, dict):
        raise ValueError(f"annotations is invalid in {annotation_path}")

    episode_ids = [int(value) for value in target_episodes]
    if len(set(episode_ids)) != len(episode_ids):
        raise ValueError(f"target_episodes contains duplicates in {annotation_path}")
    annotations: dict[int, int] = {}
    for episode_id in episode_ids:
        key = str(episode_id)
        if key not in raw_annotations:
            raise ValueError(
                f"episode {episode_id} has no {label} annotation in {annotation_path}"
            )
        frame_index = int(raw_annotations[key])
        if frame_index < 0:
            raise ValueError(f"episode {episode_id} has a negative {label} frame")
        annotations[episode_id] = frame_index
    return episode_ids, annotations


def _load_apex_annotations(dataset_root: Path) -> tuple[list[int], dict[int, int]]:
    return _load_frame_annotations(
        dataset_root,
        "apex_annotations.json",
        label="apex",
    )


def _load_start_annotations(dataset_root: Path) -> tuple[list[int], dict[int, int]]:
    return _load_frame_annotations(
        dataset_root,
        "start_position.json",
        label="start position",
    )


def _load_release_annotations(dataset_root: Path) -> tuple[list[int], dict[int, int]]:
    return _load_frame_annotations(
        dataset_root,
        "right_before_release.json",
        label="right-before-release",
    )


def _load_banana_episode(root: Path) -> core.Episode:
    raw = root / "raw"
    arm_poses = core._load_array(raw / "arm" / "action.npy", allow_pickle=True).astype(
        np.float64
    )
    arm_times = core._load_array(raw / "arm" / "time.npy", allow_pickle=True).astype(
        np.float64
    )
    if arm_poses.shape != (len(arm_poses), 4, 4) or len(arm_poses) == 0:
        raise ValueError(f"arm action is empty or malformed: {arm_poses.shape}")
    if len(arm_poses) != len(arm_times):
        count = min(len(arm_poses), len(arm_times))
        arm_poses, arm_times = arm_poses[:count], arm_times[:count]

    hand_commands = core._as_allegro_v5_actions(
        core._load_array(raw / "hand" / "action.npy"),
        label=f"episode {root.name} Allegro V5 hand action",
    )
    hand_times = core._load_array(raw / "hand" / "time.npy").astype(np.float64)
    pose_paths = sorted(
        (root / "object_tracking_foundpose_gotrack").glob(
            "**/foundpose_init/init_pose_world.npy"
        )
    )
    preferred_pose_paths = [
        path for path in pose_paths if "_foundpose_init_" in path.as_posix()
    ]
    if len(preferred_pose_paths) == 1:
        pose_path = preferred_pose_paths[0]
    elif len(pose_paths) == 1:
        pose_path = pose_paths[0]
    else:
        raise ValueError(
            "could not uniquely select the *_foundpose_init_* init_pose_world.npy "
            f"from {len(pose_paths)} result(s)"
        )
    source_object_world = core._as_transform(np.load(pose_path), label=str(pose_path))
    source_c2r = core._as_transform(
        core._load_array(root / "C2R.npy"), label=f"episode {root.name} C2R"
    )
    return core.Episode(
        root,
        arm_poses,
        arm_times,
        hand_commands,
        hand_times,
        source_object_world,
        source_c2r,
    )


def _load_banana_candidates(
    dataset_root: Path,
) -> tuple[list[core.Episode], dict[int, int]]:
    episode_ids, annotations = _load_apex_annotations(dataset_root)
    episodes: list[core.Episode] = []
    for episode_id in episode_ids:
        root = dataset_root / str(episode_id)
        try:
            if not root.is_dir():
                raise FileNotFoundError(root)
            episodes.append(_load_banana_episode(root))
        except (FileNotFoundError, ValueError) as exc:
            print(f"[dataset] skipping target episode {episode_id}: {exc}")
    if not episodes:
        raise RuntimeError(f"no replayable target episode found in {dataset_root}")
    loaded_ids = {int(episode.root.name) for episode in episodes}
    loaded_annotations = {
        episode_id: frame
        for episode_id, frame in annotations.items()
        if episode_id in loaded_ids
    }
    print(
        f"[dataset] loaded {len(episodes)}/{len(episode_ids)} target episode(s) "
        f"from {dataset_root}"
    )
    return episodes, loaded_annotations


def _map_video_frame_to_arm_index(
    episode: core.Episode,
    video_frame_index: int,
    *,
    label: str = "annotation",
) -> tuple[int, float]:
    timestamp_root = episode.root / "raw" / "timestamps"
    timestamp_path = timestamp_root / "timestamp.npy"
    frame_id_path = timestamp_root / "frame_id.npy"
    camera_times = core._load_array(timestamp_path).astype(np.float64).reshape(-1)
    camera_frame_ids = core._load_array(frame_id_path).astype(np.int64).reshape(-1)
    if len(camera_times) != len(camera_frame_ids) or len(camera_times) == 0:
        raise ValueError(
            f"camera timestamp/frame-id arrays differ or are empty: "
            f"{len(camera_times)} != {len(camera_frame_ids)}"
        )
    if video_frame_index < 0:
        raise IndexError(
            f"episode {episode.root.name} has negative {label} video frame "
            f"{video_frame_index}"
        )
    # Frame annotation JSON files are zero-based in the encoded video, while
    # the capture master timeline stores one-based frame IDs. Resolve through
    # the explicit frame_id array so a dropped timestamp entry does not shift
    # every later annotation.
    requested_frame_id = video_frame_index + 1
    timestamp_index = int(np.argmin(np.abs(camera_frame_ids - requested_frame_id)))
    resolved_frame_id = int(camera_frame_ids[timestamp_index])
    if resolved_frame_id != requested_frame_id:
        print(
            f"[sync] episode {episode.root.name} camera frame id "
            f"{requested_frame_id} is missing; using nearest id {resolved_frame_id}"
        )
    annotation_time = float(camera_times[timestamp_index])
    arm_index = int(np.argmin(np.abs(episode.arm_times - annotation_time)))
    return arm_index, annotation_time


def _map_apex_video_frame_to_arm_index(
    episode: core.Episode, video_frame_index: int
) -> tuple[int, float]:
    """Backward-compatible apex-specific synchronization wrapper."""

    return _map_video_frame_to_arm_index(
        episode,
        video_frame_index,
        label="apex",
    )


def _insert_trajectory_pause(
    trajectory: core.ReplayTrajectory,
    *,
    episode_frame_index: int,
    pause_seconds: float,
) -> tuple[core.ReplayTrajectory, int, int]:
    """Insert repeated arm/hand commands after an episode frame."""

    if pause_seconds <= 0:
        raise ValueError("pause_seconds must be positive")
    plan_index = trajectory.transition_frame_count - 1 + episode_frame_index
    if not 0 <= plan_index < len(trajectory.times):
        raise IndexError(
            f"episode pause frame {episode_frame_index} maps outside replay plan"
        )
    positive_deltas = np.diff(trajectory.times)
    positive_deltas = positive_deltas[positive_deltas > 0]
    if len(positive_deltas) == 0:
        raise ValueError("trajectory has no positive time step")
    sample_period = float(np.median(positive_deltas))
    hold_frame_count = max(1, int(np.ceil(pause_seconds / sample_period)))
    hold_offsets = np.linspace(
        pause_seconds / hold_frame_count,
        pause_seconds,
        hold_frame_count,
        dtype=np.float64,
    )

    insert_at = plan_index + 1
    held_arm = np.repeat(
        trajectory.arm_poses[plan_index][None], hold_frame_count, axis=0
    )
    held_hand = np.repeat(
        trajectory.hand_actions[plan_index][None], hold_frame_count, axis=0
    )
    shifted_suffix_times = trajectory.times[insert_at:] + pause_seconds
    times = np.concatenate(
        (
            trajectory.times[:insert_at],
            trajectory.times[plan_index] + hold_offsets,
            shifted_suffix_times,
        )
    )
    paused = core.ReplayTrajectory(
        arm_poses=np.concatenate(
            (
                trajectory.arm_poses[:insert_at],
                held_arm,
                trajectory.arm_poses[insert_at:],
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
        times=times,
        transition_frame_count=trajectory.transition_frame_count,
        transition_seconds=trajectory.transition_seconds,
    )
    return paused, plan_index, hold_frame_count


def _rank_episode_frame_zero_matches(
    current_object_robot: np.ndarray,
    current_wrist_robot: np.ndarray,
    episodes: list[core.Episode],
    *,
    position_scale_m: float,
    rotation_scale_rad: float,
) -> list[core.EpisodeMatch]:
    """Rank episodes using only each episode's frame-0 wrist pose."""

    if not episodes:
        raise ValueError("at least one candidate episode is required")
    if position_scale_m <= 0 or rotation_scale_rad <= 0:
        raise ValueError("matching position/rotation scales must be positive")
    current_object_robot = core._as_transform(
        current_object_robot, label="current object pose"
    )
    current_wrist_robot = core._as_transform(
        current_wrist_robot, label="current wrist pose"
    )
    current_relative = np.linalg.inv(current_object_robot) @ current_wrist_robot
    current_distance = float(np.linalg.norm(current_relative[:3, 3]))

    matches: list[core.EpisodeMatch] = []
    for episode in episodes:
        if len(episode.arm_poses) == 0:
            print(f"[match] skipping empty episode {episode.root.name}")
            continue
        source_object_robot = (
            np.linalg.inv(episode.source_c2r) @ episode.source_object_world
        )
        relative_poses = np.linalg.inv(source_object_robot)[None] @ episode.arm_poses
        distances = np.linalg.norm(relative_poses[:, :3, 3], axis=1)
        candidate = relative_poses[0]
        position_error = float(
            np.linalg.norm(candidate[:3, 3] - current_relative[:3, 3])
        )
        rotation_error = core._rotation_error_rad(
            candidate[:3, :3], current_relative[:3, :3]
        )
        matches.append(
            core.EpisodeMatch(
                episode=episode,
                frame_index=0,
                grasp_frame_index=int(np.argmin(distances)),
                wrist_object_distance_m=float(distances[0]),
                distance_delta_m=float(abs(distances[0] - current_distance)),
                position_error_m=position_error,
                rotation_error_rad=rotation_error,
                score=float(
                    np.hypot(
                        position_error / position_scale_m,
                        rotation_error / rotation_scale_rad,
                    )
                ),
            )
        )
    if not matches:
        raise RuntimeError("no episode contains a frame-0 arm pose")
    return sorted(matches, key=lambda match: match.score)


def _print_ranking(ranked_matches: list[core.EpisodeMatch]) -> None:
    print("[match] candidate ranking using episode frame 0 only:")
    for rank, match in enumerate(ranked_matches, start=1):
        print(
            f"  {rank}. episode {match.episode.root.name}, frame 0 "
            f"(distance minimum {match.grasp_frame_index}): "
            f"position={match.position_error_m:.4f} m, "
            f"rotation={np.rad2deg(match.rotation_error_rad):.1f} deg, "
            f"distance_delta={match.distance_delta_m:.4f} m, score={match.score:.3f}"
        )


def _save_plan(
    path: Path,
    *,
    trajectory: core.ReplayTrajectory,
    episode_arm_poses: np.ndarray,
    episode_arm_times: np.ndarray,
    episode_hand_actions: np.ndarray,
    source_object_robot: np.ndarray,
    current_object_robot: np.ndarray,
    current_c2r: np.ndarray,
    match: core.EpisodeMatch,
    ranked_matches: list[core.EpisodeMatch],
    live_arm_pose: np.ndarray,
    apex_video_frame: int,
    apex_arm_frame: int,
    apex_plan_frame: int,
    apex_hold_frame_count: int,
) -> None:
    episode = match.episode
    np.savez_compressed(
        path,
        arm_action=trajectory.arm_poses,
        arm_time=trajectory.times,
        hand_action=trajectory.hand_actions,
        episode_arm_action=episode_arm_poses,
        episode_arm_time=episode_arm_times,
        episode_hand_action=episode_hand_actions,
        transition_frame_count=trajectory.transition_frame_count,
        transition_seconds=trajectory.transition_seconds,
        episode_start_frame_in_plan=trajectory.transition_frame_count - 1,
        source_object_robot=source_object_robot,
        current_object_robot=current_object_robot,
        object_delta=current_object_robot @ np.linalg.inv(source_object_robot),
        source_c2r=episode.source_c2r,
        current_c2r=current_c2r,
        selected_episode=episode.root.name,
        selected_frame=match.frame_index,
        grasp_distance_minimum_frame=match.grasp_frame_index,
        matched_wrist_object_distance_m=match.wrist_object_distance_m,
        match_distance_delta_m=match.distance_delta_m,
        match_position_error_m=match.position_error_m,
        match_rotation_error_rad=match.rotation_error_rad,
        match_score=match.score,
        live_wrist_robot=live_arm_pose,
        candidate_episode=np.asarray(
            [item.episode.root.name for item in ranked_matches]
        ),
        candidate_frame=np.asarray(
            [item.frame_index for item in ranked_matches], dtype=np.int64
        ),
        candidate_grasp_distance_minimum_frame=np.asarray(
            [item.grasp_frame_index for item in ranked_matches], dtype=np.int64
        ),
        candidate_position_error_m=np.asarray(
            [item.position_error_m for item in ranked_matches]
        ),
        candidate_rotation_error_rad=np.asarray(
            [item.rotation_error_rad for item in ranked_matches]
        ),
        candidate_score=np.asarray([item.score for item in ranked_matches]),
        apex_video_frame=apex_video_frame,
        apex_arm_frame=apex_arm_frame,
        apex_plan_frame=apex_plan_frame,
        apex_pause_seconds=APEX_PAUSE_SECONDS,
        apex_hold_frame_count=apex_hold_frame_count,
    )


def main() -> None:
    parser = _parser()
    args = parser.parse_args()
    _validate_args(parser, args)
    core._resolve_capture_object6d_mesh(args.mesh_name, args.mesh_root_dir)
    if not args.robot_urdf.is_file():
        raise FileNotFoundError(f"preview robot URDF not found: {args.robot_urdf}")

    episodes, apex_annotations = _load_banana_candidates(BANANA_DATASET_ROOT)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = args.output_dir or (
        Path(shared_dir)
        / "capture"
        / "eccv2026"
        / "relative_replay"
        / args.robot
        / args.object
        / f"first_frame_match_{timestamp}"
    )
    output_dir = output_dir.expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)

    current_object_robot, current_c2r = core._current_object_robot_pose(
        args, output_dir / "initial_object_capture"
    )
    try:
        print(
            "[first-frame] reading xArm and Allegro state; no teleoperation is started"
        )
        live_arm_pose, live_arm_qpos, live_hand_qpos = _read_first_frame_robot_state()
        ranked_matches = _rank_episode_frame_zero_matches(
            current_object_robot,
            live_arm_pose,
            episodes,
            position_scale_m=args.match_position_scale_m,
            rotation_scale_rad=args.match_rotation_scale_rad,
        )
        match = ranked_matches[0]
        _print_ranking(ranked_matches)

        episode = match.episode
        source_arm_poses, episode_hand_actions, episode_arm_times = (
            core._episode_remainder(episode, 0)
        )
        source_object_robot = (
            np.linalg.inv(episode.source_c2r) @ episode.source_object_world
        )
        episode_arm_poses = core.relative_arm_actions(
            source_object_robot, current_object_robot, source_arm_poses
        )
        trajectory_without_pause = core._compose_replay_trajectory(
            args,
            live_arm_pose=live_arm_pose,
            live_hand_qpos=live_hand_qpos,
            episode_arm_poses=episode_arm_poses,
            episode_hand_actions=episode_hand_actions,
            episode_arm_times=episode_arm_times,
        )
        apex_video_frame = apex_annotations[int(episode.root.name)]
        apex_arm_frame, apex_timestamp = _map_apex_video_frame_to_arm_index(
            episode, apex_video_frame
        )
        trajectory, apex_plan_frame, apex_hold_frame_count = _insert_trajectory_pause(
            trajectory_without_pause,
            episode_frame_index=apex_arm_frame,
            pause_seconds=APEX_PAUSE_SECONDS,
        )

        plan_path = output_dir / "relative_replay_plan.npz"
        _save_plan(
            plan_path,
            trajectory=trajectory,
            episode_arm_poses=episode_arm_poses,
            episode_arm_times=episode_arm_times,
            episode_hand_actions=episode_hand_actions,
            source_object_robot=source_object_robot,
            current_object_robot=current_object_robot,
            current_c2r=current_c2r,
            match=match,
            ranked_matches=ranked_matches,
            live_arm_pose=live_arm_pose,
            apex_video_frame=apex_video_frame,
            apex_arm_frame=apex_arm_frame,
            apex_plan_frame=apex_plan_frame,
            apex_hold_frame_count=apex_hold_frame_count,
        )
        print(
            f"[match] selected episode {episode.root.name}, frame 0; "
            f"distance minimum/grasp frame is {match.grasp_frame_index}"
        )
        print(f"[plan] saved {plan_path}")
        print(
            f"[plan] apex annotation: video frame {apex_video_frame} "
            f"(timestamp {apex_timestamp:.6f}) -> arm frame {apex_arm_frame} -> "
            f"plan frame {apex_plan_frame}; hold {APEX_PAUSE_SECONDS:.2f}s "
            f"with {apex_hold_frame_count} repeated command frame(s)"
        )
        print(
            f"[plan] initial live state -> selected episode frame 0: "
            f"{trajectory.transition_frame_count} frames over {trajectory.transition_seconds:.2f}s"
        )

        if args.preview:
            core._preview_replay(
                args,
                match,
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
            return

        prompt = (
            "Preview complete. Type PLAY to execute the robot trajectory: "
            if args.preview
            else "Trajectory ready. Type PLAY to execute the robot trajectory: "
        )
        if input(prompt).strip() != "PLAY":
            print("[execute] cancelled; no robot commands sent.")
            return
        core._execute(
            args,
            trajectory.arm_poses,
            trajectory.hand_actions,
            trajectory.times,
            trajectory.transition_frame_count,
        )
    except KeyboardInterrupt:
        print("\n[execute] interrupted; no new robot command will be sent.")


if __name__ == "__main__":
    main()
