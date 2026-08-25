"""Replay one explicitly selected ECCV v0 episode relative to the current object.

Unlike ``replay_object_relative.py``, this entry point does not run teleoperation
or search for a matching episode/frame.  It always replays the selected episode
from frame 0 after transforming the recorded wrist trajectory into the current
object frame.
"""

from __future__ import annotations

import argparse
from dataclasses import replace
from datetime import datetime
from pathlib import Path
import sys

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:
    from paradex.retargetor.experiment import replay_object_relative as core
except ImportError:
    # Compatibility with branches where the original explicit replay still
    # lives under src/dataset_acquisition/hri.
    from src.dataset_acquisition.hri import replay_object_relative as core
from paradex.utils.path import shared_dir


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--robot", default="allegro_v5", choices=("allegro_v5",))
    parser.add_argument("--object", required=True, help="object/mesh name")
    parser.add_argument("--episode", required=True, type=int, help="numeric episode directory")
    parser.add_argument("--capture-root", type=Path, default=core.DEFAULT_CAPTURE_ROOT)
    parser.add_argument("--mesh-name", default=None, help="tracker mesh name; defaults to --object")
    parser.add_argument("--source-object-pose", default=None, help="override the episode frame-0 object pose")
    parser.add_argument("--mesh-root-dir", type=Path, default=core.DEFAULT_MESH_ROOT)
    parser.add_argument("--robot-urdf", type=Path, default=core.DEFAULT_ROBOT_URDF)
    parser.add_argument("--current-object-pose", default=None, help="skip capture/RPC and load this pose")
    parser.add_argument("--current-pose-frame", choices=("world", "robot"), default="world")
    parser.add_argument("--current-c2r-path", default=None)
    parser.add_argument("--rcc-entry", default="image_main.py")
    parser.add_argument("--rpc-addr", default="tcp://192.168.0.14:5570")
    parser.add_argument("--rpc-timeout-ms", type=int, default=300000)
    parser.add_argument("--debug-image-mode", choices=("none", "save", "popup"), default="save")
    parser.add_argument("--preview", action="store_true")
    parser.add_argument("--preview-max-frames", type=int, default=150)
    parser.add_argument("--preview-ik-max-nfev", type=int, default=50)
    parser.add_argument("--preview-position-scale", type=float, default=0.05)
    parser.add_argument("--preview-rotation-scale", type=float, default=0.5)
    parser.add_argument("--no-viser-object-align", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--rate-scale", type=float, default=1.0)
    parser.add_argument("--approach-linear-speed-mps", type=float, default=0.05)
    parser.add_argument("--approach-angular-speed-rps", type=float, default=0.5)
    parser.add_argument("--approach-min-seconds", type=float, default=5.0)
    parser.add_argument("--approach-rate-hz", type=float, default=50.0)
    parser.add_argument("--settle-seconds", type=float, default=1.0)
    parser.add_argument(
        "--plan-only",
        action="store_true",
        help="generate/save/preview only; this is also the default without --execute",
    )
    parser.add_argument("--execute", action="store_true", help="allow real arm/hand commands after PLAY")
    return parser


def _validate_args(parser: argparse.ArgumentParser, args: argparse.Namespace) -> None:
    if args.plan_only and args.execute:
        parser.error("--plan-only and --execute are mutually exclusive")
    if args.episode < 0:
        parser.error("--episode must be non-negative")
    if args.rate_scale <= 0:
        parser.error("--rate-scale must be positive")
    if args.rpc_timeout_ms <= 0:
        parser.error("--rpc-timeout-ms must be positive")
    if args.preview_max_frames <= 0 or args.preview_ik_max_nfev <= 0:
        parser.error("preview frame/evaluation limits must be positive")
    if args.preview_position_scale <= 0 or args.preview_rotation_scale <= 0:
        parser.error("preview residual scales must be positive")
    if args.approach_linear_speed_mps <= 0 or args.approach_angular_speed_rps <= 0:
        parser.error("approach speeds must be positive")
    if args.approach_min_seconds < 0 or args.approach_rate_hz <= 0:
        parser.error("approach duration/rate must be non-negative/positive")


def _load_selected_episode(args: argparse.Namespace) -> core.Episode:
    root = Path(args.capture_root).expanduser() / args.robot / args.object / str(args.episode)
    if not root.is_dir():
        raise FileNotFoundError(f"selected episode directory not found: {root}")
    episode = core._load_episode_root(args, root)
    if args.source_object_pose is not None:
        episode = replace(
            episode,
            source_object_world=core._load_pose_file(
                Path(args.source_object_pose).expanduser(), frame="world"
            ),
        )
    return episode


def _explicit_match_for_preview(
    episode: core.Episode,
    source_object_robot: np.ndarray,
    current_object_robot: np.ndarray,
    live_arm_pose: np.ndarray,
) -> core.EpisodeMatch:
    source_relative = np.linalg.inv(source_object_robot)[None] @ episode.arm_poses
    distances = np.linalg.norm(source_relative[:, :3, 3], axis=1)
    grasp_frame = int(np.argmin(distances))
    current_relative = np.linalg.inv(current_object_robot) @ live_arm_pose
    position_error = float(np.linalg.norm(source_relative[0, :3, 3] - current_relative[:3, 3]))
    rotation_error = core._rotation_error_rad(
        source_relative[0, :3, :3], current_relative[:3, :3]
    )
    return core.EpisodeMatch(
        episode=episode,
        frame_index=0,
        grasp_frame_index=grasp_frame,
        wrist_object_distance_m=float(distances[0]),
        distance_delta_m=float(abs(distances[0] - np.linalg.norm(current_relative[:3, 3]))),
        position_error_m=position_error,
        rotation_error_rad=rotation_error,
        score=0.0,
    )


def _read_preview_robot_state() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Read live state through temporary read-only ROS clients before Viser starts."""

    arm_pose, arm_qpos = core._live_xarm_preview_state()
    hand_qpos = core._live_allegro_v5_preview_qpos()
    return arm_pose, arm_qpos, hand_qpos


def main() -> None:
    if not hasattr(core, "_compose_replay_trajectory"):
        core.main()
        return

    parser = _parser()
    args = parser.parse_args()
    _validate_args(parser, args)
    args.mesh_name = args.mesh_name or args.object
    core._resolve_capture_object6d_mesh(args.mesh_name, args.mesh_root_dir)
    if not args.robot_urdf.is_file():
        raise FileNotFoundError(f"preview robot URDF not found: {args.robot_urdf}")

    episode = _load_selected_episode(args)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = args.output_dir or (
        Path(shared_dir)
        / "capture"
        / "eccv2026"
        / "relative_replay"
        / args.robot
        / args.object
        / f"{args.episode}_{timestamp}"
    )
    output_dir = output_dir.expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)

    current_object_robot, current_c2r = core._current_object_robot_pose(
        args, output_dir / "initial_object_capture"
    )
    try:
        # Do not keep rclpy-backed command controllers alive while Viser uses
        # Ctrl+C as its return mechanism.  rclpy's SIGINT handler invalidates
        # their shared context.  Execution controllers are opened only after
        # the preview has returned and PLAY has been confirmed.
        live_arm_pose, live_arm_qpos, live_hand_qpos = _read_preview_robot_state()
        source_object_robot = np.linalg.inv(episode.source_c2r) @ episode.source_object_world
        episode_arm_poses = core.relative_arm_actions(
            source_object_robot, current_object_robot, episode.arm_poses
        )
        episode_hand_actions = core._zoh_resample(
            episode.hand_times, episode.hand_commands, episode.arm_times
        )
        trajectory = core._compose_replay_trajectory(
            args,
            live_arm_pose=live_arm_pose,
            live_hand_qpos=live_hand_qpos,
            episode_arm_poses=episode_arm_poses,
            episode_hand_actions=episode_hand_actions,
            episode_arm_times=episode.arm_times,
        )
        match = _explicit_match_for_preview(
            episode, source_object_robot, current_object_robot, live_arm_pose
        )

        plan_path = output_dir / "relative_replay_plan.npz"
        np.savez_compressed(
            plan_path,
            arm_action=trajectory.arm_poses,
            arm_time=trajectory.times,
            hand_action=trajectory.hand_actions,
            episode_arm_action=episode_arm_poses,
            episode_arm_time=episode.arm_times,
            episode_hand_action=episode_hand_actions,
            transition_frame_count=trajectory.transition_frame_count,
            transition_seconds=trajectory.transition_seconds,
            episode_start_frame_in_plan=trajectory.transition_frame_count - 1,
            source_object_robot=source_object_robot,
            current_object_robot=current_object_robot,
            object_delta=current_object_robot @ np.linalg.inv(source_object_robot),
            source_c2r=episode.source_c2r,
            current_c2r=current_c2r,
            selected_episode=str(args.episode),
            selected_frame=0,
            live_wrist_robot=live_arm_pose,
        )
        print(f"[episode] explicitly selected {args.object}/{args.episode} from frame 0")
        print(f"[plan] saved {plan_path}")
        print(
            f"[plan] current state -> episode frame 0: "
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
