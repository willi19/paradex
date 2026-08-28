#!/usr/bin/env python3
"""Visualize live grasp indexing at each banana episode's cached grasp frame.

The banana pose is estimated once from a multiview Object6D capture.  VIVE +
MANUS teleoperation then runs without camera recording while Viser displays the
live xArm/Allegro state together with every candidate episode at its minimum
wrist-to-object-distance frame, transformed into the detected banana frame.
Those frame indices are persisted and reused.  The candidate closest to the
live wrist pose is highlighted and changes as the robot moves.
"""

from __future__ import annotations

import argparse
import datetime
import json
import select
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from threading import Event, Lock
from typing import Any, Iterable

import numpy as np
import trimesh
from scipy.spatial.transform import Rotation


PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from paradex.retargetor.experiment import replay_object_relative as replay_core
from paradex.retargetor.experiment import (
    replay_object_relative_first_frame as apex_core,
)
from paradex.retargetor.experiment.replay_pose_retrieval import (
    DEFAULT_CANDIDATE_EPISODES,
    DEFAULT_EPISODE_ROOT,
    load_candidate_episodes,
    parse_episode_ids,
)
from paradex.utils.path import shared_dir
from paradex.visualization.visualizer.viser import ViserViewer
from src.object6d.capture_sil import (
    capture_once,
    extract_pose,
    send_rpc_once,
    to_4x4,
    to_shared_data_path,
)


DEFAULT_HAND_URDF = (
    PROJECT_ROOT / "rsc" / "robot" / "allegro_v5" / "allegro_right_A.urdf"
)
LIVE_COLOR = (0.95, 0.95, 0.95, 1.0)
SELECTED_COLOR = (0.15, 0.95, 0.25, 0.9)
CANDIDATE_COLOR = (0.25, 0.55, 1.0, 0.16)


@dataclass(frozen=True)
class CandidateVisualization:
    episode: replay_core.Episode
    scene_name: str
    eef_robot: np.ndarray
    hand_pose_robot: np.ndarray
    hand_qpos: np.ndarray
    frame_index: int
    hand_mesh: trimesh.Trimesh


@dataclass(frozen=True)
class PreloadedEpisodeHand:
    episode: replay_core.Episode
    frame_index: int
    hand_qpos: np.ndarray
    hand_mesh: trimesh.Trimesh


@dataclass(frozen=True)
class ReplaySelection:
    """Top candidate and measured robot state frozen by the PLAY request."""

    episode_id: str
    frame_index: int
    live_arm_pose: np.ndarray
    live_hand_qpos: np.ndarray
    trigger: str = "unknown"


class MiddlePedalPlayTrigger:
    """Convert the bimanual middle-pedal hold state into one PLAY press edge."""

    def __init__(self, pedal_state: Any | None = None) -> None:
        if pedal_state is None:
            from paradex.io.streamdeck_pedal import MiddlePedalState

            pedal_state = MiddlePedalState()
        self._pedal_state = pedal_state
        self._was_pressed = self._read_pressed()

    def _read_pressed(self) -> bool:
        return int(self._pedal_state.get_state()) == 0

    def poll_pressed(self) -> bool:
        pressed = self._read_pressed()
        triggered = pressed and not self._was_pressed
        self._was_pressed = pressed
        return triggered

    def close(self) -> None:
        self._pedal_state.close()


def discover_episode_ids(episode_root: Path) -> tuple[int, ...]:
    """Discover every numeric episode directory, regardless of grasp label."""

    root = Path(episode_root).expanduser()
    if not root.is_dir():
        raise FileNotFoundError(f"banana episode directory not found: {root}")
    episode_ids = tuple(
        sorted(
            int(path.name)
            for path in root.iterdir()
            if path.is_dir() and path.name.isdigit()
        )
    )
    if not episode_ids:
        raise FileNotFoundError(f"no numeric episode directories found under {root}")
    return episode_ids


def load_all_valid_episodes(
    episode_root: Path,
    episode_ids: Iterable[int],
) -> list[replay_core.Episode]:
    """Load all usable requested episodes without hiding the remaining scene."""

    episodes: list[replay_core.Episode] = []
    for episode_id in episode_ids:
        try:
            episode = load_candidate_episodes(episode_root, (episode_id,))[0]
        except (FileNotFoundError, ValueError) as exc:
            print(f"[preload] skipping episode {episode_id}: {exc}")
            continue
        if len(episode.arm_poses) == 0 or len(episode.hand_commands) == 0:
            print(f"[preload] skipping empty episode {episode_id}")
            continue
        episodes.append(episode)
    if not episodes:
        raise RuntimeError(f"no valid banana episode found under {episode_root}")
    return episodes


def resolve_candidate_episode_ids(
    episode_root: Path,
    configured_episode_ids: tuple[int, ...] | None,
) -> tuple[int, ...]:
    """Use capture_sil's validated banana candidates for its default dataset."""

    if configured_episode_ids is not None:
        return configured_episode_ids
    root = Path(episode_root).expanduser()
    if root == DEFAULT_EPISODE_ROOT.expanduser():
        return DEFAULT_CANDIDATE_EPISODES
    return discover_episode_ids(root)


def load_apex_arm_frames(
    episode_root: Path,
    episodes: Iterable[replay_core.Episode],
) -> tuple[dict[str, int], dict[str, int]]:
    """Map apex_annotations.json video frames onto each episode's arm timeline."""

    _, video_annotations = apex_core._load_apex_annotations(
        Path(episode_root).expanduser()
    )
    arm_frames: dict[str, int] = {}
    video_frames: dict[str, int] = {}
    for episode in episodes:
        episode_id = int(episode.root.name)
        if episode_id not in video_annotations:
            raise ValueError(
                f"episode {episode_id} has no apex annotation in "
                f"{Path(episode_root) / 'apex_annotations.json'}"
            )
        video_frame = int(video_annotations[episode_id])
        arm_frame, _ = apex_core._map_video_frame_to_arm_index(
            episode,
            video_frame,
            label="apex",
        )
        arm_frames[episode.root.name] = arm_frame
        video_frames[episode.root.name] = video_frame
        print(
            f"[apex] episode {episode.root.name}: video frame {video_frame} -> "
            f"arm frame {arm_frame}"
        )
    return arm_frames, video_frames


def _poll_terminal_command() -> str | None:
    """Read one complete terminal line without creating a competing input thread."""

    try:
        readable, _, _ = select.select([sys.stdin], [], [], 0.0)
    except (OSError, ValueError):
        return None
    if not readable:
        return None
    value = sys.stdin.readline()
    if value == "":
        return None
    return value.strip()


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--name", "--mesh-name", dest="mesh_name", default="banana")
    parser.add_argument("--save-path", default="object_6d/grasp_indexing")
    parser.add_argument("--rpc-addr", default="tcp://192.168.0.3:5570")
    parser.add_argument("--rpc-timeout-ms", type=int, default=300_000)
    parser.add_argument("--rcc-entry", default="image_main.py")
    parser.add_argument(
        "--mesh-root-dir",
        type=Path,
        default=Path(shared_dir) / "mesh_blender",
    )
    parser.add_argument(
        "--episode-root",
        type=Path,
        default=DEFAULT_EPISODE_ROOT,
        help="directory containing numeric banana episode directories",
    )
    parser.add_argument(
        "--candidate-episodes",
        type=parse_episode_ids,
        default=None,
        help="comma-separated episode IDs; defaults to automatic discovery",
    )
    parser.add_argument(
        "--device", choices=("xsens", "occulus", "vive"), default="vive"
    )
    parser.add_argument(
        "--no-teleop",
        action="store_true",
        help=(
            "do not start VIVE/MANUS teleoperation; continuously read xArm and "
            "Allegro state without publishing commands"
        ),
    )
    parser.add_argument("--allegro-command-rate-hz", type=float, default=30.0)
    parser.add_argument("--visualization-rate-hz", type=float, default=20.0)
    parser.add_argument(
        "--execute",
        action="store_true",
        help="allow the saved top-candidate replay to command the real arm/hand after PLAY",
    )
    parser.add_argument(
        "--auto-execute",
        action="store_true",
        help="with --execute, skip the terminal PLAY confirmation and execute immediately",
    )
    pedal_play = parser.add_mutually_exclusive_group()
    pedal_play.add_argument(
        "--pedal-play",
        dest="pedal_play",
        action="store_true",
        default=True,
        help="allow the Stream Deck Pedal middle button to select and PLAY the top candidate",
    )
    pedal_play.add_argument(
        "--no-pedal-play",
        dest="pedal_play",
        action="store_false",
        help="do not open the Stream Deck Pedal",
    )
    parser.add_argument(
        "--replay-output-dir",
        type=Path,
        default=None,
        help="replay plan directory; defaults to the initial Object6D capture directory",
    )
    parser.add_argument("--rate-scale", type=float, default=1.0)
    parser.add_argument("--approach-linear-speed-mps", type=float, default=0.5)
    parser.add_argument("--approach-angular-speed-rps", type=float, default=2.0)
    parser.add_argument("--approach-min-seconds", type=float, default=1.0)
    parser.add_argument("--approach-rate-hz", type=float, default=50.0)
    parser.add_argument("--settle-seconds", type=float, default=1.0)
    replay_preview = parser.add_mutually_exclusive_group()
    replay_preview.add_argument(
        "--replay-preview",
        dest="replay_preview",
        action="store_true",
        default=True,
        help="open the generated top-candidate trajectory in Viser before PLAY",
    )
    replay_preview.add_argument(
        "--no-replay-preview",
        dest="replay_preview",
        action="store_false",
        help="skip the generated top-candidate Viser trajectory preview",
    )
    parser.add_argument("--preview-max-frames", type=int, default=150)
    parser.add_argument("--preview-ik-max-nfev", type=int, default=50)
    parser.add_argument("--preview-position-scale", type=float, default=0.05)
    parser.add_argument("--preview-rotation-scale", type=float, default=0.5)
    parser.add_argument(
        "--robot-urdf", type=Path, default=replay_core.DEFAULT_ROBOT_URDF
    )
    parser.add_argument(
        "--hand-urdf",
        type=Path,
        default=DEFAULT_HAND_URDF,
        help="standalone Allegro V5 URDF used for hand-only visualization",
    )
    parser.add_argument(
        "--preview-robot-link-max-faces",
        type=int,
        default=500,
        help="maximum faces per link for each overlaid robot; 0 disables simplification",
    )
    parser.add_argument(
        "--preview-object-max-faces",
        type=int,
        default=3_000,
    )
    parser.add_argument(
        "--grasp-frame-cache",
        type=Path,
        default=None,
        help=(
            "cached closest-approach arm-frame JSON; defaults to "
            "<episode-root>/grasp_index_frames.json"
        ),
    )
    parser.add_argument(
        "--rebuild-grasp-frame-cache",
        action="store_true",
        help="recompute every closest-approach frame and overwrite the cache",
    )
    parser.add_argument("--no-viser-object-align", action="store_true")
    return parser


def _validate_args(parser: argparse.ArgumentParser, args: argparse.Namespace) -> None:
    positive = {
        "--rpc-timeout-ms": args.rpc_timeout_ms,
        "--allegro-command-rate-hz": args.allegro_command_rate_hz,
        "--visualization-rate-hz": args.visualization_rate_hz,
        "--rate-scale": args.rate_scale,
        "--approach-linear-speed-mps": args.approach_linear_speed_mps,
        "--approach-angular-speed-rps": args.approach_angular_speed_rps,
        "--approach-rate-hz": args.approach_rate_hz,
        "--preview-position-scale": args.preview_position_scale,
        "--preview-rotation-scale": args.preview_rotation_scale,
    }
    invalid = [name for name, value in positive.items() if value <= 0]
    if invalid:
        parser.error(f"positive value required for: {', '.join(invalid)}")
    if args.preview_robot_link_max_faces < 0 or args.preview_object_max_faces < 0:
        parser.error("preview mesh face limits must be non-negative")
    if args.preview_max_frames <= 0 or args.preview_ik_max_nfev <= 0:
        parser.error("preview frame/evaluation limits must be positive")
    if args.approach_min_seconds < 0 or args.settle_seconds < 0:
        parser.error("approach/settle durations must be non-negative")
    if args.mesh_name != "banana":
        parser.error("grasp indexing visualization currently supports banana only")
    if not args.robot_urdf.is_file():
        parser.error(f"combined robot URDF not found: {args.robot_urdf}")
    if not args.hand_urdf.is_file():
        parser.error(f"Allegro hand URDF not found: {args.hand_urdf}")


def rank_episode_grasp_frame_matches(
    current_object_robot: np.ndarray,
    current_wrist_robot: np.ndarray,
    episodes: Iterable[replay_core.Episode],
    *,
    candidate_frame_indices: dict[str, int] | None = None,
) -> list[replay_core.EpisodeMatch]:
    """Rank cached grasp frames using only object-relative wrist rotation."""
    current_object = replay_core._as_transform(
        current_object_robot, label="current object pose"
    )
    current_wrist = replay_core._as_transform(
        current_wrist_robot, label="current wrist pose"
    )
    current_relative = np.linalg.inv(current_object) @ current_wrist
    current_distance = float(np.linalg.norm(current_relative[:3, 3]))

    matches: list[replay_core.EpisodeMatch] = []
    for episode in episodes:
        if len(episode.arm_poses) == 0:
            continue
        source_object = np.linalg.inv(episode.source_c2r) @ episode.source_object_world
        relative_poses = np.linalg.inv(source_object)[None] @ episode.arm_poses
        distances = np.linalg.norm(relative_poses[:, :3, 3], axis=1)
        if candidate_frame_indices is None:
            frame_index = 0
        else:
            if episode.root.name not in candidate_frame_indices:
                raise KeyError(
                    f"cached grasp frame is missing for episode {episode.root.name}"
                )
            frame_index = candidate_frame_indices[episode.root.name]
        if not 0 <= frame_index < len(relative_poses):
            raise IndexError(
                f"candidate frame {frame_index} is outside episode "
                f"{episode.root.name} ({len(relative_poses)} frames)"
            )
        candidate = relative_poses[frame_index]
        position_error = float(
            np.linalg.norm(candidate[:3, 3] - current_relative[:3, 3])
        )
        rotation_error = replay_core._rotation_error_rad(
            candidate[:3, :3], current_relative[:3, :3]
        )
        matches.append(
            replay_core.EpisodeMatch(
                episode=episode,
                frame_index=frame_index,
                grasp_frame_index=int(np.argmin(distances)),
                wrist_object_distance_m=float(distances[frame_index]),
                distance_delta_m=float(abs(distances[frame_index] - current_distance)),
                position_error_m=position_error,
                rotation_error_rad=rotation_error,
                score=rotation_error,
            )
        )
    if not matches:
        raise RuntimeError("no candidate episode contains a usable grasp frame")
    return sorted(
        matches,
        key=lambda match: (match.score, int(match.episode.root.name)),
    )


def _capture_initial_object_pose(
    args: argparse.Namespace,
) -> tuple[np.ndarray, np.ndarray, Path]:
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    relative_capture = Path(args.save_path) / timestamp
    capture_root = Path(shared_dir) / relative_capture
    remote_path = (Path("shared_data") / relative_capture / "raw").as_posix()

    capture_once(capture_root, remote_path, args.rcc_entry)
    response = send_rpc_once(
        args.rpc_addr,
        {
            "command": "infer",
            "image_path": to_shared_data_path(capture_root),
            "mesh_name": args.mesh_name,
            "save_projection_grids": True,
            "wooden_object_triangulation": False,
        },
        args.rpc_timeout_ms,
    )
    pose = extract_pose(response)
    c2r = to_4x4(np.load(capture_root / "C2R.npy"))
    pose_robot = np.linalg.inv(c2r) @ to_4x4(pose["pose_world"])
    pose["C2R"] = c2r.astype(float).tolist()
    pose["pose_robot"] = pose_robot.astype(float).tolist()
    result_path = capture_root / "object_6d.json"
    result_path.write_text(
        json.dumps(pose, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"[object6d] saved initial banana pose: {result_path}")
    return pose_robot, c2r, capture_root


def hand_pose_from_eef(
    eef_robot: np.ndarray,
    palm_eef: np.ndarray,
) -> np.ndarray:
    """Place the standalone hand's palm using the combined URDF mount."""

    eef = replay_core._as_transform(eef_robot, label="episode EEF pose")
    palm_mount = replay_core._as_transform(palm_eef, label="Allegro palm pose in EEF")
    return eef @ palm_mount


def closest_approach_frame(episode: replay_core.Episode) -> tuple[int, float]:
    """Select the recorded wrist frame closest to the episode's object."""

    source_object_robot = (
        np.linalg.inv(episode.source_c2r) @ episode.source_object_world
    )
    relative_poses = np.linalg.inv(source_object_robot)[None] @ episode.arm_poses
    distances = np.linalg.norm(relative_poses[:, :3, 3], axis=1)
    frame_index = int(np.argmin(distances))
    return frame_index, float(distances[frame_index])


def load_or_create_grasp_frame_cache(
    episodes: list[replay_core.Episode],
    cache_path: Path,
    *,
    rebuild: bool = False,
) -> dict[str, int]:
    """Load valid cached grasp frames and compute only missing/stale entries."""

    path = Path(cache_path).expanduser()
    payload: dict[str, Any] = {}
    if path.is_file() and not rebuild:
        try:
            loaded = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(loaded, dict):
                payload = loaded
        except (OSError, json.JSONDecodeError) as exc:
            print(f"[grasp-frame] ignoring invalid cache {path}: {exc}")
    records = payload.get("episodes")
    if not isinstance(records, dict) or rebuild:
        records = {}

    selected: dict[str, int] = {}
    changed = rebuild or not path.is_file()
    for episode in episodes:
        episode_id = episode.root.name
        record = records.get(episode_id)
        frame_count = len(episode.arm_poses)
        valid = (
            isinstance(record, dict)
            and record.get("arm_frame_count") == frame_count
            and isinstance(record.get("frame_index"), int)
            and 0 <= record["frame_index"] < frame_count
        )
        if valid:
            selected[episode_id] = int(record["frame_index"])
            continue
        frame_index, distance_m = closest_approach_frame(episode)
        records[episode_id] = {
            "frame_index": frame_index,
            "arm_frame_count": frame_count,
            "arm_time": float(episode.arm_times[frame_index]),
            "wrist_object_distance_m": distance_m,
        }
        selected[episode_id] = frame_index
        changed = True
        print(
            f"[grasp-frame] episode {episode_id}: frame {frame_index}/"
            f"{frame_count - 1}, distance={distance_m:.4f} m"
        )

    if changed:
        output = {
            "version": 1,
            "method": "minimum_wrist_object_distance",
            "description": "Arm frame closest to the object for grasp visualization.",
            "episodes": records,
        }
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_text(
            json.dumps(output, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        temporary.replace(path)
        print(f"[grasp-frame] saved cache: {path}")
    else:
        print(f"[grasp-frame] loaded cached frames: {path}")
    return selected


def _preload_hand_link_meshes(
    hand: Any,
    max_faces_per_link: int,
) -> list[tuple[str, trimesh.Trimesh]]:
    """Simplify each link once so candidate poses never repeat decimation."""

    from paradex.visualization.robot import simplify_mesh

    return [
        (link_name, simplify_mesh(mesh, max_faces_per_link))
        for link_name, mesh in hand.scene.geometry.items()
    ]


def _combine_configured_hand_mesh(
    hand: Any,
    link_meshes: list[tuple[str, trimesh.Trimesh]],
) -> trimesh.Trimesh:
    """Apply current FK transforms and combine all preloaded hand links."""

    transformed = []
    for link_name, mesh in link_meshes:
        link_mesh = mesh.copy()
        link_mesh.apply_transform(hand.scene.graph.get(link_name)[0])
        transformed.append(link_mesh)
    return trimesh.util.concatenate(transformed)


def _preload_episode_hands(
    args: argparse.Namespace,
    episodes: list[replay_core.Episode],
    frame_indices: dict[str, int],
) -> tuple[np.ndarray, list[PreloadedEpisodeHand]]:
    """Build the cached grasp-frame hand meshes with one shared URDF instance."""

    from paradex.visualization.robot import RobotModule

    started = time.perf_counter()
    combined = RobotModule(str(args.robot_urdf))
    palm_eef = combined.urdf.get_transform("palm_link", "link6")
    hand = RobotModule(str(args.hand_urdf))
    hand_joint_names = tuple(hand.get_joint_names())
    if len(hand_joint_names) != 16:
        raise ValueError(
            f"hand-only URDF must expose 16 joints, got {len(hand_joint_names)}"
        )
    hand_joint_limits = hand.get_joint_limits()
    source_faces_per_hand = sum(
        len(mesh.faces) for mesh in hand.scene.geometry.values()
    )
    link_meshes = _preload_hand_link_meshes(hand, args.preview_robot_link_max_faces)
    preview_faces_per_hand = sum(len(mesh.faces) for _, mesh in link_meshes)
    preloaded: list[PreloadedEpisodeHand] = []
    for episode in episodes:
        frame_index = frame_indices[episode.root.name]
        hand_action = replay_core._zoh_resample(
            episode.hand_times,
            episode.hand_commands,
            episode.arm_times[frame_index : frame_index + 1],
        )[0]
        hand_qpos = replay_core._allegro_v5_preview_qpos(
            hand_action[None],
            urdf_hand_joint_names=hand_joint_names,
            joint_limits=hand_joint_limits,
        )[0]
        hand.update_cfg(hand_qpos)
        preloaded.append(
            PreloadedEpisodeHand(
                episode=episode,
                frame_index=frame_index,
                hand_qpos=hand_qpos,
                hand_mesh=_combine_configured_hand_mesh(hand, link_meshes),
            )
        )
    print(
        f"[preload] {len(preloaded)} cached grasp-frame hand mesh(es) ready in "
        f"{time.perf_counter() - started:.3f}s; "
        f"faces/hand={source_faces_per_hand}->{preview_faces_per_hand}"
    )
    return palm_eef, preloaded


def _prepare_candidates(
    args: argparse.Namespace,
    current_object_robot: np.ndarray,
    episodes: list[replay_core.Episode],
    frame_indices: dict[str, int] | None = None,
) -> list[CandidateVisualization]:
    if frame_indices is None:
        frame_indices = {
            episode.root.name: closest_approach_frame(episode)[0]
            for episode in episodes
        }
    palm_eef, preloaded_hands = _preload_episode_hands(args, episodes, frame_indices)
    return _place_preloaded_hands(
        current_object_robot,
        palm_eef,
        preloaded_hands,
    )


def _place_preloaded_hands(
    current_object_robot: np.ndarray,
    palm_eef: np.ndarray,
    preloaded_hands: list[PreloadedEpisodeHand],
) -> list[CandidateVisualization]:
    """Place prebuilt grasp-frame hand meshes in the detected object scene."""

    candidates: list[CandidateVisualization] = []
    for preloaded in preloaded_hands:
        episode = preloaded.episode
        source_object_robot = (
            np.linalg.inv(episode.source_c2r) @ episode.source_object_world
        )
        frame_index = preloaded.frame_index
        eef_robot = replay_core.relative_arm_actions(
            source_object_robot,
            current_object_robot,
            episode.arm_poses[frame_index : frame_index + 1],
        )[0]
        candidates.append(
            CandidateVisualization(
                episode=episode,
                scene_name=f"episode_{episode.root.name}",
                eef_robot=eef_robot,
                hand_pose_robot=hand_pose_from_eef(eef_robot, palm_eef),
                hand_qpos=preloaded.hand_qpos,
                frame_index=frame_index,
                hand_mesh=preloaded.hand_mesh,
            )
        )
    return candidates


class LiveGraspIndexingVisualization:
    """Own the Viser scene and update it from an active CaptureSession."""

    def __init__(
        self,
        args: argparse.Namespace,
        current_object_robot: np.ndarray,
        episodes: list[replay_core.Episode],
        candidates: list[CandidateVisualization],
    ) -> None:
        self.args = args
        self.current_object_robot = current_object_robot
        self.episodes = episodes
        self.candidates = candidates
        first_candidate = candidates[0]
        self.palm_eef = (
            np.linalg.inv(first_candidate.eef_robot) @ first_candidate.hand_pose_robot
        )
        self.candidate_frame_indices = {
            candidate.episode.root.name: candidate.frame_index
            for candidate in candidates
        }
        self.selected_episode: str | None = None
        self.replay_event = Event()
        self._replay_lock = Lock()
        self._replay_selection: ReplaySelection | None = None
        self._latest_arm_pose: np.ndarray | None = None
        self._latest_hand_qpos: np.ndarray | None = None
        self.next_update_time = 0.0
        self.update_count = 0

        mesh_path = replay_core._resolve_capture_object6d_mesh(
            args.mesh_name, args.mesh_root_dir
        )
        object_pose_for_viser = replay_core._viser_object_pose(
            current_object_robot,
            mesh_path,
            apply_mesh_alignment=not args.no_viser_object_align,
        )
        object_mesh = replay_core._load_preview_mesh(
            mesh_path, args.preview_object_max_faces
        )

        self.viewer = ViserViewer(
            scene_title="Live banana grasp indexing",
            show_player=False,
        )
        with self.viewer.server.gui.add_folder(
            "Live grasp indexing", expand_by_default=True
        ):
            self.episode_text = self.viewer.server.gui.add_text(
                "Top episode", initial_value="waiting for robot state", disabled=True
            )
            self.frame_text = self.viewer.server.gui.add_text(
                "Top grasp frame", initial_value="-", disabled=True
            )
            self.score_text = self.viewer.server.gui.add_text(
                "Score", initial_value="-", disabled=True
            )
            self.position_text = self.viewer.server.gui.add_text(
                "Position error", initial_value="-", disabled=True
            )
            self.rotation_text = self.viewer.server.gui.add_text(
                "Rotation error", initial_value="-", disabled=True
            )
            self.hand_text = self.viewer.server.gui.add_text(
                "Hand RMS error", initial_value="-", disabled=True
            )
            self.top3_text = self.viewer.server.gui.add_text(
                "Top 3 scores", initial_value="-", disabled=True
            )
            self.update_count_text = self.viewer.server.gui.add_text(
                "State updates", initial_value="0", disabled=True
            )
            self.play_button = self.viewer.server.gui.add_button(
                "PLAY top candidate", disabled=True
            )
            self.viewer.server.gui.add_text(
                "Candidates",
                initial_value=str(len(candidates)),
                disabled=True,
            )
            self.viewer.server.gui.add_text(
                "Replayable through apex",
                initial_value=str(len(episodes)),
                disabled=True,
            )

        @self.play_button.on_click
        def _(_) -> None:
            self.request_replay(trigger="viser")

        self.viewer.add_floor(height=0.0)
        self.viewer.add_frame("robot_origin", np.eye(4, dtype=np.float64))
        self.viewer.add_object(args.mesh_name, object_mesh, object_pose_for_viser)
        self.viewer.add_robot(
            "live",
            str(args.hand_urdf),
            pose=hand_pose_from_eef(np.eye(4), self.palm_eef),
            include_arm_meshes=True,
            max_mesh_faces=args.preview_robot_link_max_faces,
        )
        self.viewer.change_color("live", LIVE_COLOR)
        live_hand_robot = self.viewer.robot_dict["live"]
        self.hand_joint_names = tuple(live_hand_robot.urdf.get_joint_names())
        self.hand_joint_limits = live_hand_robot.urdf.get_joint_limits()

        self.candidate_mesh_handles: dict[str, Any] = {}
        self.candidate_mesh_specs: dict[str, tuple[str, trimesh.Trimesh]] = {}
        self.candidate_mesh_revisions: dict[str, int] = {}
        for candidate in candidates:
            episode_id = candidate.episode.root.name
            root_name = f"/candidate_hands/{candidate.scene_name}"
            pose = candidate.hand_pose_robot
            self.viewer.server.scene.add_frame(
                root_name,
                position=pose[:3, 3],
                wxyz=Rotation.from_matrix(pose[:3, :3]).as_quat()[[3, 0, 1, 2]],
                show_axes=False,
            )
            mesh_handle = self.viewer.server.scene.add_mesh_simple(
                f"{root_name}/mesh_0",
                vertices=np.asarray(candidate.hand_mesh.vertices),
                faces=np.asarray(candidate.hand_mesh.faces),
                color=self._rgb255(CANDIDATE_COLOR),
                opacity=CANDIDATE_COLOR[3],
            )
            self.candidate_mesh_handles[episode_id] = mesh_handle
            self.candidate_mesh_specs[episode_id] = (root_name, candidate.hand_mesh)
            self.candidate_mesh_revisions[episode_id] = 0
            self.viewer.server.scene.add_label(
                f"{root_name}/label",
                f"episode {episode_id} · frame {candidate.frame_index}",
            )

    @staticmethod
    def _rgb255(color: tuple[float, float, float, float]) -> tuple[int, int, int]:
        return tuple(int(round(channel * 255)) for channel in color[:3])

    def _set_candidate_selected(
        self,
        episode_id: str,
        selected: bool,
    ) -> None:
        self.candidate_mesh_handles[episode_id].remove()
        root_name, mesh = self.candidate_mesh_specs[episode_id]
        revision = self.candidate_mesh_revisions[episode_id] + 1
        self.candidate_mesh_revisions[episode_id] = revision
        color = SELECTED_COLOR if selected else CANDIDATE_COLOR
        self.candidate_mesh_handles[episode_id] = (
            self.viewer.server.scene.add_mesh_simple(
                f"{root_name}/mesh_{revision}",
                vertices=np.asarray(mesh.vertices),
                faces=np.asarray(mesh.faces),
                color=self._rgb255(color),
                opacity=color[3],
            )
        )

    def request_replay(self, *, trigger: str = "unknown") -> bool:
        """Freeze the current winner and measured state for replay planning."""

        with self._replay_lock:
            if (
                self.selected_episode is None
                or self._latest_arm_pose is None
                or self._latest_hand_qpos is None
            ):
                print("[play] waiting for the first valid robot-state update")
                return False
            self._replay_selection = ReplaySelection(
                episode_id=self.selected_episode,
                frame_index=self.candidate_frame_indices[self.selected_episode],
                live_arm_pose=self._latest_arm_pose.copy(),
                live_hand_qpos=self._latest_hand_qpos.copy(),
                trigger=trigger,
            )
            self.play_button.disabled = True
            self.replay_event.set()
        print(
            f"[play] locked episode {self._replay_selection.episode_id} "
            f"frame {self._replay_selection.frame_index}"
        )
        return True

    def reset_replay_request(self) -> None:
        """Re-arm candidate selection after a replay/teleoperation cycle."""

        with self._replay_lock:
            self._replay_selection = None
            self._latest_arm_pose = None
            self._latest_hand_qpos = None
            self.replay_event.clear()
            self.play_button.disabled = True
        self.next_update_time = 0.0

    def replay_selection(self) -> ReplaySelection | None:
        with self._replay_lock:
            return self._replay_selection

    def update_robot_state(
        self,
        live_arm_pose: np.ndarray,
        live_hand_qpos: np.ndarray,
    ) -> None:
        """Update the live mesh and top candidate from one measured robot state."""

        live_arm_pose = replay_core._as_transform(live_arm_pose, label="live xArm pose")
        live_hand_qpos = np.asarray(live_hand_qpos, dtype=np.float64).reshape(-1)
        if live_hand_qpos.shape != (16,) or not np.all(np.isfinite(live_hand_qpos)):
            return

        live_hand_pose = hand_pose_from_eef(live_arm_pose, self.palm_eef)
        live_robot = self.viewer.robot_dict["live"]
        live_robot._visual_root_frame.position = live_hand_pose[:3, 3]
        live_robot._visual_root_frame.wxyz = Rotation.from_matrix(
            live_hand_pose[:3, :3]
        ).as_quat()[[3, 0, 1, 2]]
        display_hand_qpos = replay_core._allegro_v5_preview_qpos(
            live_hand_qpos[None],
            urdf_hand_joint_names=self.hand_joint_names,
            joint_limits=self.hand_joint_limits,
        )
        live_robot.update_cfg(display_hand_qpos[0])
        ranked = rank_episode_grasp_frame_matches(
            self.current_object_robot,
            live_arm_pose,
            self.episodes,
            candidate_frame_indices=self.candidate_frame_indices,
        )
        selected = ranked[0]
        selected_id = selected.episode.root.name
        selected_frame = self.candidate_frame_indices[selected_id]
        selected_hand = replay_core._zoh_resample(
            selected.episode.hand_times,
            selected.episode.hand_commands,
            selected.episode.arm_times[selected_frame : selected_frame + 1],
        )[0]
        hand_error_rad = float(
            np.sqrt(np.mean(np.square(selected_hand - live_hand_qpos)))
        )

        self.update_count += 1
        self.episode_text.value = selected_id
        self.frame_text.value = str(selected.frame_index)
        self.score_text.value = f"{selected.score:.3f}"
        self.position_text.value = f"{selected.position_error_m * 100.0:.1f} cm"
        self.rotation_text.value = f"{np.rad2deg(selected.rotation_error_rad):.1f} deg"
        self.hand_text.value = f"{hand_error_rad:.3f} rad"
        self.top3_text.value = " | ".join(
            f"{match.episode.root.name}[f{match.frame_index}]: {match.score:.3f}"
            for match in ranked[:3]
        )
        self.update_count_text.value = str(self.update_count)
        previous_selected = self.selected_episode
        with self._replay_lock:
            self.selected_episode = selected_id
            self._latest_arm_pose = live_arm_pose.copy()
            self._latest_hand_qpos = live_hand_qpos.copy()
            if not self.replay_event.is_set():
                self.play_button.disabled = False
        if selected_id == previous_selected:
            return

        if previous_selected is not None:
            self._set_candidate_selected(previous_selected, False)
        self._set_candidate_selected(selected_id, True)
        self.viewer.server.flush()
        print(
            f"[match] top episode {selected_id} frame {selected.frame_index}: "
            f"position={selected.position_error_m:.4f} m, "
            f"rotation={np.rad2deg(selected.rotation_error_rad):.1f} deg, "
            f"hand_rms={hand_error_rad:.3f} rad, "
            f"score={selected.score:.3f}"
        )

    def update(self, session: Any) -> None:
        """CaptureSession callback used by the live teleoperation path."""

        now = time.monotonic()
        if now < self.next_update_time:
            return
        self.next_update_time = now + 1.0 / self.args.visualization_rate_hz

        arm_state = session.arm.get_data()
        hand_state = session.hand.get_data()
        self.update_robot_state(
            arm_state.get("position"),
            hand_state.get("qpos"),
        )

    def close(self) -> None:
        replay_core._stop_viser_server(self.viewer.server)


class ReadOnlyRobotStateMonitor:
    """Keep xArm and Allegro state subscriptions alive without command publishers."""

    def __init__(self, *, timeout_seconds: float = 3.0) -> None:
        from rclpy.executors import SingleThreadedExecutor
        from rclpy.node import Node
        from xarm_msgs.srv import GetFloat32List

        try:
            from xarm_msgs.msg import RobotMsg
        except ImportError:
            RobotMsg = None

        from paradex.io.robot_controller import get_hand
        from paradex.utils.system import network_info

        self._latest_arm_pose: np.ndarray | None = None
        self._hand = get_hand(
            "allegro_v5",
            hand_side="right",
            command_enabled=False,
        )
        namespace = str(network_info["xarm"]["param"].get("namespace", "")).strip("/")
        base = f"/{namespace}/xarm" if namespace else "/xarm"
        self._timeout_seconds = timeout_seconds
        self._arm_node = Node(f"grasp_indexing_state_{time.time_ns()}")
        self._arm_subscription = None
        self._use_arm_position_service = RobotMsg is None
        if RobotMsg is not None:
            self._arm_subscription = self._arm_node.create_subscription(
                RobotMsg,
                f"{base}/robot_states",
                self._on_arm_state,
                10,
            )
        self._arm_position_client = self._arm_node.create_client(
            GetFloat32List,
            f"{base}/get_position",
        )
        self._arm_executor = SingleThreadedExecutor()
        self._arm_executor.add_node(self._arm_node)

        if not self._hand.connection_event.wait(timeout_seconds):
            self.close()
            raise RuntimeError("timed out waiting for Allegro V5 joint state")
        if self._arm_subscription is not None:
            deadline = time.monotonic() + timeout_seconds
            while self._latest_arm_pose is None and time.monotonic() < deadline:
                self._arm_executor.spin_once(timeout_sec=0.05)
        if self._latest_arm_pose is None:
            try:
                self._latest_arm_pose = self._read_arm_position_service()
                self._use_arm_position_service = True
            except RuntimeError:
                self.close()
                raise RuntimeError(
                    f"timed out waiting for xArm state from {base}"
                ) from None

    def _on_arm_state(self, message: Any) -> None:
        try:
            self._latest_arm_pose = replay_core._xarm_position_to_transform(
                message.pose[:6]
            )
        except (TypeError, ValueError):
            return

    def _read_arm_position_service(self) -> np.ndarray:
        from xarm_msgs.srv import GetFloat32List

        if not self._arm_position_client.wait_for_service(
            timeout_sec=self._timeout_seconds
        ):
            raise RuntimeError("xArm get_position service is unavailable")
        future = self._arm_position_client.call_async(GetFloat32List.Request())
        self._arm_executor.spin_until_future_complete(
            future,
            timeout_sec=self._timeout_seconds,
        )
        if not future.done():
            raise RuntimeError("timed out reading xArm get_position")
        response = future.result()
        if response is None or response.ret != 0 or len(response.datas) < 6:
            raise RuntimeError("xArm get_position failed")
        return replay_core._xarm_position_to_transform(response.datas[:6])

    def get_state(self) -> tuple[np.ndarray, np.ndarray]:
        if self._use_arm_position_service:
            self._latest_arm_pose = self._read_arm_position_service()
        else:
            self._arm_executor.spin_once(timeout_sec=0.0)
        if self._latest_arm_pose is None:
            raise RuntimeError("xArm state is not available")
        hand_data = self._hand.get_data()
        hand_qpos = np.asarray(hand_data.get("qpos"), dtype=np.float64).reshape(-1)
        if hand_qpos.shape != (16,) or not np.all(np.isfinite(hand_qpos)):
            raise RuntimeError("Allegro V5 state is not a finite 16-vector")
        return self._latest_arm_pose.copy(), hand_qpos

    def close(self) -> None:
        arm_executor = getattr(self, "_arm_executor", None)
        arm_node = getattr(self, "_arm_node", None)
        if arm_executor is not None and arm_node is not None:
            try:
                arm_executor.remove_node(arm_node)
            except Exception:
                pass
            arm_executor.shutdown()
            arm_node.destroy_node()
            self._arm_executor = None
            self._arm_node = None
        hand = getattr(self, "_hand", None)
        if hand is not None:
            hand.end()
            self._hand = None


def _build_top_candidate_replay(
    args: argparse.Namespace,
    *,
    current_object_robot: np.ndarray,
    current_c2r: np.ndarray,
    episodes: Iterable[replay_core.Episode],
    selection: ReplaySelection,
    apex_frame_indices: dict[str, int],
    apex_video_frames: dict[str, int],
    output_dir: Path,
    persist_plan: bool = True,
) -> tuple[Path | None, replay_core.ReplayTrajectory]:
    """Build and persist current-state approach plus the selected replay suffix."""

    episodes_by_id = {episode.root.name: episode for episode in episodes}
    try:
        episode = episodes_by_id[selection.episode_id]
    except KeyError as exc:
        raise KeyError(
            f"selected episode {selection.episode_id} is no longer loaded"
        ) from exc
    frame_index = selection.frame_index
    if not 0 <= frame_index < len(episode.arm_poses):
        raise IndexError(
            f"selected frame {frame_index} is outside episode "
            f"{selection.episode_id} ({len(episode.arm_poses)} frames)"
        )
    if selection.episode_id not in apex_frame_indices:
        raise KeyError(
            f"apex arm frame is missing for episode {selection.episode_id}"
        )
    end_frame = int(apex_frame_indices[selection.episode_id])
    if not frame_index <= end_frame < len(episode.arm_poses):
        raise ValueError(
            f"episode {selection.episode_id} apex arm frame {end_frame} must be "
            f"between selected frame {frame_index} and trajectory end "
            f"{len(episode.arm_poses) - 1}"
        )
    if selection.episode_id not in apex_video_frames:
        raise KeyError(
            f"apex video frame is missing for episode {selection.episode_id}"
        )
    source_object_robot = (
        np.linalg.inv(episode.source_c2r) @ episode.source_object_world
    )
    replay_slice = slice(frame_index, end_frame + 1)
    episode_arm_times = episode.arm_times[replay_slice]
    episode_arm_poses = replay_core.relative_arm_actions(
        source_object_robot,
        current_object_robot,
        episode.arm_poses[replay_slice],
    )
    episode_hand_actions = replay_core._zoh_resample(
        episode.hand_times,
        episode.hand_commands,
        episode_arm_times,
    )
    trajectory = replay_core._compose_replay_trajectory(
        args,
        live_arm_pose=selection.live_arm_pose,
        live_hand_qpos=selection.live_hand_qpos,
        episode_arm_poses=episode_arm_poses,
        episode_hand_actions=episode_hand_actions,
        episode_arm_times=episode_arm_times,
    )

    plan_path = None
    if persist_plan:
        output_dir = Path(output_dir).expanduser()
        output_dir.mkdir(parents=True, exist_ok=True)
        plan_path = output_dir / "top_candidate_replay_plan.npz"
        np.savez_compressed(
            plan_path,
            arm_action=trajectory.arm_poses,
            arm_time=trajectory.times,
            hand_action=trajectory.hand_actions,
            episode_arm_action=episode_arm_poses,
            episode_arm_time=episode_arm_times,
            episode_hand_action=episode_hand_actions,
            transition_frame_count=trajectory.transition_frame_count,
            transition_seconds=trajectory.transition_seconds,
            episode_start_frame_in_plan=trajectory.transition_frame_count - 1,
            replay_speed_scale=args.rate_scale,
            selected_episode=episode.root.name,
            selected_frame=frame_index,
            apex_video_frame=apex_video_frames[selection.episode_id],
            apex_arm_frame=end_frame,
            source_object_robot=source_object_robot,
            current_object_robot=current_object_robot,
            object_delta=current_object_robot @ np.linalg.inv(source_object_robot),
            source_c2r=episode.source_c2r,
            current_c2r=current_c2r,
            live_wrist_robot=selection.live_arm_pose,
            live_hand_qpos=selection.live_hand_qpos,
        )
    print(
        f"[plan] episode {episode.root.name} frame {frame_index} -> "
        f"annotated apex arm frame {end_frame} "
        f"({len(episode_arm_poses)} source frames)"
    )
    print(
        f"[plan] current state -> selected scene: "
        f"{trajectory.transition_frame_count} frames over "
        f"{trajectory.transition_seconds:.2f}s"
    )
    if plan_path is not None:
        print(f"[plan] saved {plan_path}")
    return plan_path, trajectory


def _execute_replay_if_confirmed(
    args: argparse.Namespace,
    trajectory: replay_core.ReplayTrajectory,
    pedal_trigger: MiddlePedalPlayTrigger | None = None,
    *,
    arm: Any | None = None,
    hand: Any | None = None,
    immediate: bool = False,
) -> bool:
    """Require the established exact PLAY confirmation before real motion."""

    if not args.execute:
        print("[plan] complete; no robot or hand motion commands were sent.")
        return False
    if immediate:
        print("[execute] pedal PLAY; executing immediately without preview/confirmation.")
    elif getattr(args, "auto_execute", False):
        print("[execute] --auto-execute enabled; skipping PLAY confirmation.")
    else:
        readiness = (
            "Preview complete."
            if getattr(args, "replay_preview", False)
            else "Trajectory ready."
        )
        try:
            if pedal_trigger is None:
                answer = input(
                    f"{readiness} Type PLAY to execute the robot trajectory: "
                ).strip()
                confirmed = answer == "PLAY"
            else:
                print(
                    f"{readiness} Release, then press the middle pedal again "
                    "to execute; alternatively type PLAY and Enter."
                )
                confirmed = False
                while True:
                    if pedal_trigger.poll_pressed():
                        print("[pedal] middle button confirmed robot execution")
                        confirmed = True
                        break
                    command = _poll_terminal_command()
                    if command is not None:
                        confirmed = command == "PLAY"
                        break
                    time.sleep(0.02)
        except (EOFError, KeyboardInterrupt):
            print("\n[execute] cancelled; no robot commands sent.")
            return False
        if not confirmed:
            print("[execute] cancelled; no robot commands sent.")
            return False
    controller_kwargs = {}
    if arm is not None or hand is not None:
        controller_kwargs = {"arm": arm, "hand": hand}
    replay_core._execute(
        args,
        trajectory.arm_poses,
        trajectory.hand_actions,
        trajectory.times,
        trajectory.transition_frame_count,
        **controller_kwargs,
    )
    return True


def _preview_top_candidate_replay(
    args: argparse.Namespace,
    *,
    current_object_robot: np.ndarray,
    episodes: Iterable[replay_core.Episode],
    candidate_frame_indices: dict[str, int],
    selection: ReplaySelection,
    trajectory: replay_core.ReplayTrajectory,
    output_dir: Path,
) -> None:
    """Open the exact saved top-candidate plan in the existing Viser player."""

    matches = rank_episode_grasp_frame_matches(
        current_object_robot,
        selection.live_arm_pose,
        episodes,
        candidate_frame_indices=candidate_frame_indices,
    )
    try:
        match = next(
            candidate
            for candidate in matches
            if candidate.episode.root.name == selection.episode_id
        )
    except StopIteration as exc:
        raise RuntimeError(
            f"selected episode {selection.episode_id} is absent from preview ranking"
        ) from exc

    _, live_arm_qpos = replay_core._live_xarm_preview_state()
    print(
        f"[preview] opening episode {selection.episode_id} from cached frame "
        f"{selection.frame_index}; use Resume/Playback in Viser"
    )
    replay_core._preview_replay(
        args,
        match,
        trajectory.arm_poses,
        trajectory.hand_actions,
        trajectory.times,
        trajectory.transition_frame_count,
        current_object_robot,
        output_dir,
        live_arm_pose=selection.live_arm_pose,
        live_arm_qpos=live_arm_qpos,
        live_hand_qpos=selection.live_hand_qpos,
    )


def _run_teleoperation(
    args: argparse.Namespace,
    visualization: LiveGraspIndexingVisualization,
    pedal_trigger: MiddlePedalPlayTrigger | None = None,
    session: Any | None = None,
) -> tuple[ReplaySelection | None, Any | None]:
    # CaptureSession imports optional hardware/UI dependencies, so keep it out
    # of module import and --help paths.
    from paradex.dataset_acqusition.capture import CaptureSession

    exit_event = Event()
    events = {"save": Event(), "stop": Event(), "exit": exit_event}

    def update_and_check_play(session: Any) -> None:
        visualization.update(session)
        if pedal_trigger is not None and pedal_trigger.poll_pressed():
            print("[pedal] middle button pressed")
            visualization.request_replay(trigger="pedal")
        command = _poll_terminal_command()
        if command is not None:
            if command.lower() == "p":
                visualization.request_replay(trigger="terminal")
            elif command.lower() == "q":
                exit_event.set()
        if visualization.replay_event.is_set():
            exit_event.set()

    if session is None:
        session = CaptureSession(
            camera=False,
            realsense=False,
            arm="xarm",
            hand="allegro_v5",
            teleop=args.device,
            hand_side="right",
            events=events,
            timestamp=False,
            arm_kwargs={"servo_api": "cartesian_aa"},
            hand_command_rate_hz=args.allegro_command_rate_hz,
            use_vive=args.device == "vive",
            require_left_control=args.device == "vive",
        )
    try:
        print(
            "[teleop] camera recording is disabled. Teleoperation and live "
            "candidate indexing are active; click PLAY top candidate, press p, "
            "or press the middle pedal to build the replay; enter q to exit."
        )
        state = session.teleop(
            session_events=events,
            state_policy="keyboard_control",
            loop_callback=update_and_check_play,
        )
        print(f"[teleop] finished with state={state!r}")
        selection = visualization.replay_selection()
        if selection is None:
            return None, session

        # Keep the already initialized ROS controllers alive for execution.
        # Destroying them here and constructing a second xArm node in the same
        # process leaves rclpy's shared executor in a stale state.  Stop only
        # the outgoing servo stream while the plan is built and previewed;
        # replay_core._execute() will resume it with its first move().
        session.arm.stop_motion_commands()
        return selection, session
    finally:
        if hasattr(session, "retargetor"):
            session.retargetor.stop()


def _run_visualization_only(
    visualization: LiveGraspIndexingVisualization,
    pedal_trigger: MiddlePedalPlayTrigger | None = None,
) -> ReplaySelection | None:
    """Continuously visualize read-only robot state without VIVE or MANUS."""

    exit_event = Event()
    print(
        "[no-teleop] live read-only xArm + Allegro visualization is active; "
        "VIVE/MANUS, command publishers, and CaptureSession were not started. "
        "Click PLAY top candidate or press p to build the replay; enter q or "
        "press Ctrl+C to exit."
    )
    monitor = ReadOnlyRobotStateMonitor()
    update_period = 1.0 / visualization.args.visualization_rate_hz
    try:
        while not exit_event.is_set():
            started = time.monotonic()
            live_arm_pose, live_hand_qpos = monitor.get_state()
            visualization.update_robot_state(live_arm_pose, live_hand_qpos)
            if pedal_trigger is not None and pedal_trigger.poll_pressed():
                print("[pedal] middle button pressed")
                visualization.request_replay(trigger="pedal")
            command = _poll_terminal_command()
            if command is not None:
                if command.lower() == "p":
                    visualization.request_replay(trigger="terminal")
                elif command.lower() == "q":
                    exit_event.set()
            if visualization.replay_event.is_set():
                break
            remaining = update_period - (time.monotonic() - started)
            exit_event.wait(max(0.0, remaining))
    finally:
        monitor.close()
    return visualization.replay_selection()


def _run_selected_replay_cycle(
    args: argparse.Namespace,
    *,
    current_object_robot: np.ndarray,
    current_c2r: np.ndarray,
    episodes: list[replay_core.Episode],
    candidate_frame_indices: dict[str, int],
    apex_frame_indices: dict[str, int],
    apex_video_frames: dict[str, int],
    selection: ReplaySelection,
    output_dir: Path,
    pedal_trigger: MiddlePedalPlayTrigger | None,
    session: Any | None,
) -> bool:
    """Build and run one selection, using the low-latency pedal fast path."""

    immediate = selection.trigger == "pedal" and args.execute
    started = time.perf_counter()
    _, trajectory = _build_top_candidate_replay(
        args,
        current_object_robot=current_object_robot,
        current_c2r=current_c2r,
        episodes=episodes,
        selection=selection,
        apex_frame_indices=apex_frame_indices,
        apex_video_frames=apex_video_frames,
        output_dir=output_dir,
        persist_plan=not immediate,
    )
    print(
        f"[plan] object-relative trajectory ready in "
        f"{time.perf_counter() - started:.3f}s"
    )
    if args.replay_preview and not immediate:
        _preview_top_candidate_replay(
            args,
            current_object_robot=current_object_robot,
            episodes=episodes,
            candidate_frame_indices=candidate_frame_indices,
            selection=selection,
            trajectory=trajectory,
            output_dir=output_dir,
        )
    return _execute_replay_if_confirmed(
        args,
        trajectory,
        pedal_trigger,
        arm=None if session is None else session.arm,
        hand=None if session is None else session.hand,
        immediate=immediate,
    )


def main() -> None:
    parser = _parser()
    args = parser.parse_args()
    _validate_args(parser, args)
    args.episode_root = args.episode_root.expanduser()
    args.object = args.mesh_name
    args.candidate_episodes = resolve_candidate_episode_ids(
        args.episode_root,
        args.candidate_episodes,
    )

    print(
        f"[preload] loading {len(args.candidate_episodes)} candidate episode(s) "
        f"from {args.episode_root}"
    )
    episodes = load_all_valid_episodes(args.episode_root, args.candidate_episodes)
    grasp_frame_cache = (
        args.grasp_frame_cache.expanduser()
        if args.grasp_frame_cache is not None
        else args.episode_root / "grasp_index_frames.json"
    )
    frame_indices = load_or_create_grasp_frame_cache(
        episodes,
        grasp_frame_cache,
        rebuild=args.rebuild_grasp_frame_cache,
    )
    apex_frame_indices, apex_video_frames = load_apex_arm_frames(
        args.episode_root,
        episodes,
    )
    replayable_episodes = []
    for episode in episodes:
        episode_id = episode.root.name
        if frame_indices[episode_id] > apex_frame_indices[episode_id]:
            print(
                f"[apex] episode {episode_id} remains visible but is excluded "
                f"from ranking: cached frame {frame_indices[episode_id]} is "
                f"after apex arm frame {apex_frame_indices[episode_id]}"
            )
            continue
        replayable_episodes.append(episode)
    if not replayable_episodes:
        raise RuntimeError("no candidate has a cached frame at or before its apex")
    with ThreadPoolExecutor(
        max_workers=1,
        thread_name_prefix="candidate-hand-preload",
    ) as executor:
        preload_future = executor.submit(
            _preload_episode_hands,
            args,
            episodes,
            frame_indices,
        )
        current_object_robot, current_c2r, object_pose_path = (
            _capture_initial_object_pose(args)
        )
        palm_eef, preloaded_hands = preload_future.result()
    candidates = _place_preloaded_hands(
        current_object_robot,
        palm_eef,
        preloaded_hands,
    )
    visualization = LiveGraspIndexingVisualization(
        args, current_object_robot, replayable_episodes, candidates
    )
    pedal_trigger = None
    if args.pedal_play:
        try:
            pedal_trigger = MiddlePedalPlayTrigger()
            print("[pedal] middle button PLAY trigger is active")
        except Exception as exc:
            print(f"[pedal] unavailable; continuing without pedal: {exc}")
    replay_session: Any | None = None
    replay_output_dir = (
        args.replay_output_dir.expanduser()
        if args.replay_output_dir is not None
        else object_pose_path.parent
    )
    try:
        if args.no_teleop:
            selection = _run_visualization_only(visualization, pedal_trigger)
            if selection is None:
                print("[play] exited without selecting a replay candidate.")
                return
            _run_selected_replay_cycle(
                args,
                current_object_robot=current_object_robot,
                current_c2r=current_c2r,
                episodes=replayable_episodes,
                candidate_frame_indices=frame_indices,
                apex_frame_indices=apex_frame_indices,
                apex_video_frames=apex_video_frames,
                selection=selection,
                output_dir=replay_output_dir,
                pedal_trigger=pedal_trigger,
                session=None,
            )
        else:
            while True:
                selection, replay_session = _run_teleoperation(
                    args,
                    visualization,
                    pedal_trigger,
                    session=replay_session,
                )
                if selection is None:
                    print("[play] teleoperation exited.")
                    break
                replay_executed = _run_selected_replay_cycle(
                    args,
                    current_object_robot=current_object_robot,
                    current_c2r=current_c2r,
                    episodes=replayable_episodes,
                    candidate_frame_indices=frame_indices,
                    apex_frame_indices=apex_frame_indices,
                    apex_video_frames=apex_video_frames,
                    selection=selection,
                    output_dir=replay_output_dir,
                    pedal_trigger=pedal_trigger,
                    session=replay_session,
                )
                replay_session.arm.stop_motion_commands()
                if replay_executed:
                    replay_session.set_hand_teleoperation_enabled(False)
                    print(
                        "[teleop] Allegro teleoperation disabled; holding the "
                        "final replay hand pose"
                    )
                visualization.reset_replay_request()
                print("[teleop] replay finished; resuming from the current robot pose")
    except KeyboardInterrupt:
        print("\n[teleop] interrupted; shutting down.")
    finally:
        if replay_session is not None:
            replay_session.arm.stop_motion_commands()
            replay_session.end()
        visualization.close()
        if pedal_trigger is not None:
            pedal_trigger.close()


if __name__ == "__main__":
    main()
