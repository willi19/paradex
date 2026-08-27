"""Select and replay an ECCV v0 grasp from a live object-relative wrist pose.

The object is localized once before teleoperation.  When the replay key is
pressed, each candidate grasp first selects the frame whose scalar
wrist-to-object distance is closest to the live distance.  Those selected
frames are compared by their full object-relative 6D wrist pose, and the
remainder of the best grasp is transformed into the current object frame::

    T_current_eef[t] = T_current_object @ inv(T_source_object) @ T_source_eef[t]

Thus a 5 cm translation of the object produces the same 5 cm translation of
the wrist, while the saved Allegro V5 hand command is replayed at the original
time.

This is intentionally separate from ``capture_robot.py``: it uses the same
remote-camera/robot-controller conventions but never changes the recorder.
For a live run, its one-frame 6D estimate follows
``h2r/capture_object6d.py``: capture a synchronized image set, request one
RPC pose inference, and convert the returned world pose into robot frame.
Real motion requires the explicit ``--execute`` flag.
"""

from __future__ import annotations

import argparse
import json
import select
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping

import numpy as np

# Allow direct execution from any working directory.
PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from paradex.calibration.utils import load_current_C2R, save_current_camparam
from paradex.io.camera_system.remote_camera_controller import remote_camera_controller
from paradex.retargetor.allegro_alignment import (
    ALLEGRO_URDF_JOINT_NAMES,
    ALLEGRO_V5_DRIVER_JOINT_NAMES,
    retargeter_action_to_urdf_qpos,
)
from paradex.utils.path import shared_dir


DEFAULT_CAPTURE_ROOT = Path(shared_dir) / "capture" / "eccv2026" / "v0"
DEFAULT_MESH_ROOT = Path(shared_dir) / "mesh_blender"
DEFAULT_ROBOT_URDF = PROJECT_ROOT / "rsc" / "robot" / "xarm_allegro_v5.urdf"
@dataclass(frozen=True)
class Episode:
    root: Path
    arm_poses: np.ndarray
    arm_times: np.ndarray
    hand_commands: np.ndarray
    hand_times: np.ndarray
    source_object_world: np.ndarray
    source_c2r: np.ndarray


@dataclass(frozen=True)
class EpisodeMatch:
    episode: Episode
    frame_index: int
    grasp_frame_index: int
    wrist_object_distance_m: float
    distance_delta_m: float
    position_error_m: float
    rotation_error_rad: float
    score: float


@dataclass(frozen=True)
class ReplayTrajectory:
    arm_poses: np.ndarray
    hand_actions: np.ndarray
    times: np.ndarray
    transition_frame_count: int
    transition_seconds: float


def _as_transform(value: Any, *, label: str) -> np.ndarray:
    pose = np.asarray(value, dtype=np.float64)
    if pose.shape == (3, 4):
        pose = np.vstack((pose, np.array([0.0, 0.0, 0.0, 1.0])))
    if pose.shape != (4, 4):
        raise ValueError(f"{label} must be 4x4 or 3x4, got {pose.shape}")
    if not np.all(np.isfinite(pose)) or not np.allclose(pose[3], [0, 0, 0, 1]):
        raise ValueError(f"{label} is not a finite homogeneous transform")
    return pose


def relative_arm_actions(
    source_object_robot: np.ndarray,
    current_object_robot: np.ndarray,
    source_arm_actions: np.ndarray,
) -> np.ndarray:
    """Apply the object-frame transform to each 4x4 end-effector action."""

    source_object_robot = _as_transform(source_object_robot, label="source object pose")
    current_object_robot = _as_transform(current_object_robot, label="current object pose")
    source_arm_actions = np.asarray(source_arm_actions, dtype=np.float64)
    if source_arm_actions.ndim != 3 or source_arm_actions.shape[1:] != (4, 4):
        raise ValueError("source arm action.npy must have shape (N, 4, 4)")
    delta = current_object_robot @ np.linalg.inv(source_object_robot)
    return delta[None, :, :] @ source_arm_actions


def _load_array(path: Path, *, allow_pickle: bool = False) -> np.ndarray:
    if not path.exists():
        raise FileNotFoundError(path)
    return np.asarray(np.load(path, allow_pickle=allow_pickle))


def _load_pose_file(path: Path, *, frame: str) -> np.ndarray:
    """Load a pose from txt/npy/npz or the JSON emitted by the pose tools."""

    suffix = path.suffix.lower()
    if suffix == ".txt":
        return _as_transform(np.loadtxt(path), label=str(path))
    if suffix == ".npy":
        return _as_transform(np.load(path), label=str(path))
    if suffix == ".npz":
        with np.load(path) as values:
            key = "frame_0" if "frame_0" in values else sorted(values.files)[0]
            return _as_transform(values[key], label=f"{path}:{key}")
    if suffix == ".json":
        payload = json.loads(path.read_text(encoding="utf-8"))
        keys = ("pose_robot", "pose") if frame == "robot" else ("pose_world", "refined_pose_world", "pose")
        for key in keys:
            if payload.get(key) is not None:
                return _as_transform(payload[key], label=f"{path}:{key}")
        nested = payload.get("object_6d")
        if isinstance(nested, Mapping):
            for key in keys:
                if nested.get(key) is not None:
                    return _as_transform(nested[key], label=f"{path}:object_6d.{key}")
        raise ValueError(f"No {frame}-frame pose in {path}")
    raise ValueError(f"Unsupported pose file: {path}")


def _zoh_resample(times: np.ndarray, values: np.ndarray, target_times: np.ndarray) -> np.ndarray:
    """Zero-order hold matches how the robot controller holds its latest command."""

    times = np.asarray(times, dtype=np.float64).reshape(-1)
    values = np.asarray(values, dtype=np.float64)
    target_times = np.asarray(target_times, dtype=np.float64).reshape(-1)
    if len(times) != len(values) or len(times) == 0:
        raise ValueError("hand command/time arrays are empty or have different lengths")
    order = np.argsort(times)
    indices = np.searchsorted(times[order], target_times, side="right") - 1
    indices = np.clip(indices, 0, len(order) - 1)
    return values[order][indices]


def _as_allegro_v5_actions(values: np.ndarray, *, label: str) -> np.ndarray:
    """Validate the recorded controller order: ``joint_0_0`` through ``joint_15_0``."""

    actions = np.asarray(values, dtype=np.float64)
    if actions.ndim != 2 or actions.shape[1] != 16:
        raise ValueError(f"{label} must have shape (N, 16), got {actions.shape}")
    return actions


def _as_allegro_v5_joint_vector(values: np.ndarray, *, label: str) -> np.ndarray:
    joints = np.asarray(values, dtype=np.float64).reshape(-1)
    if joints.shape != (16,) or not np.all(np.isfinite(joints)):
        raise ValueError(f"{label} must be a finite 16-joint vector, got {joints.shape}")
    return joints


def _allegro_v5_preview_qpos(
    actions: np.ndarray,
    *,
    urdf_hand_joint_names: tuple[str, ...],
    joint_limits: Mapping[str, tuple[float, float]],
) -> np.ndarray:
    """Map recorded v5 actions by joint name and clamp the rendered URDF pose.

    Captures and the v5 driver use index/middle/ring/thumb blocks.  Some
    preview URDFs expose those native ``joint_<n>_0`` names, while older
    combined URDFs expose semantic thumb/index/middle/ring names.  Mapping by
    name keeps both conventions correct.  Clamping is display-only: historical
    captures contain requested commands outside the physical joint limits,
    especially ``joint_13_0=-2.1``, which otherwise makes the thumb animation
    look unrelated to the real hand.
    """

    actions = _as_allegro_v5_actions(actions, label="Allegro V5 preview hand action")
    names = tuple(str(name) for name in urdf_hand_joint_names)
    if len(names) != 16 or len(set(names)) != 16:
        raise ValueError(f"preview URDF must expose 16 unique Allegro joints, got {names}")

    if set(names) == set(ALLEGRO_V5_DRIVER_JOINT_NAMES):
        mapped = np.asarray(
            [
                [dict(zip(ALLEGRO_V5_DRIVER_JOINT_NAMES, action))[name] for name in names]
                for action in actions
            ],
            dtype=np.float64,
        )
    elif set(names) == set(ALLEGRO_URDF_JOINT_NAMES):
        mapped = np.asarray(
            [
                [retargeter_action_to_urdf_qpos(action)[name] for name in names]
                for action in actions
            ],
            dtype=np.float64,
        )
    else:
        raise ValueError(
            "preview URDF Allegro joints match neither the v5 driver nor semantic "
            f"convention: {names}"
        )

    missing_limits = [name for name in names if name not in joint_limits]
    if missing_limits:
        raise ValueError(f"preview URDF is missing Allegro joint limits: {missing_limits}")
    lower = np.asarray([joint_limits[name][0] for name in names], dtype=np.float64)
    upper = np.asarray([joint_limits[name][1] for name in names], dtype=np.float64)
    return np.clip(mapped, lower, upper)


def _episode_source_pose_path(root: Path, override: str | None) -> Path:
    if override:
        return Path(override).expanduser()
    # ``object_6d_pose_v2.npz`` uses the same object-frame convention as the
    # current one-shot FoundationPose convention. Prefer it when an episode has both files;
    # the v1 pose would otherwise introduce a spurious rigid rotation.
    for candidate in (
        root / "object_6d" / "pose_000000.txt",
        root / "object_6d_pose_v2.npz",
        root / "object_6d_pose.npz",
    ):
        if candidate.is_file():
            return candidate
    # FoundPose+GoTrack captures keep the frame-0 initialization below a
    # run/attempt directory instead of materializing the legacy pose files at
    # the episode root. GoTrack runs can contain a copied FoundPose
    # initialization; always prefer the dedicated ``*_foundpose_init_*`` run
    # so retrieval uses the pose produced by the initialization pipeline.
    foundpose_candidates = sorted(
        root.glob(
            "object_tracking_foundpose_gotrack/*/attempt_*/"
            "foundpose_init/init_pose_world.npy"
        )
    )
    preferred_candidates = [
        path
        for path in foundpose_candidates
        if "_foundpose_init_" in path.parents[2].name
    ]
    if len(preferred_candidates) == 1:
        return preferred_candidates[0]
    if len(preferred_candidates) > 1:
        raise ValueError(
            "multiple dedicated FoundPose initialization poses found; select one "
            f"with --source-object-pose: {preferred_candidates}"
        )
    if len(foundpose_candidates) == 1:
        return foundpose_candidates[0]
    if len(foundpose_candidates) > 1:
        raise ValueError(
            "multiple FoundPose initialization poses found, but none belongs to a "
            f"*_foundpose_init_* run: {foundpose_candidates}"
        )
    raise FileNotFoundError(
        "source object frame-0 pose not found; expected "
        f"{root / 'object_6d' / 'pose_000000.txt'}, "
        f"{root / 'object_6d_pose_v2.npz'}, {root / 'object_6d_pose.npz'}, "
        "or one FoundPose initialization under object_tracking_foundpose_gotrack"
    )


def _candidate_episode_roots(capture_root: Path, robot: str, object_name: str) -> list[Path]:
    object_root = Path(capture_root).expanduser() / robot / object_name
    if not object_root.is_dir():
        raise FileNotFoundError(f"object capture directory not found: {object_root}")
    roots = sorted(
        (path for path in object_root.iterdir() if path.is_dir() and path.name.isdigit()),
        key=lambda path: int(path.name),
    )
    if not roots:
        raise FileNotFoundError(f"no numeric grasp episodes found under {object_root}")

    successful: list[Path] = []
    for root in roots:
        result_path = root / "grasp_result.json"
        if not result_path.is_file():
            continue
        try:
            result = json.loads(result_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            print(f"[match] ignoring unreadable grasp result {result_path}: {exc}")
            continue
        if result.get("grasp_success") is True:
            successful.append(root)
    if successful:
        print(f"[match] using {len(successful)} successful grasp episode(s) from {object_root}")
        return successful
    print(f"[match] no successful grasp labels found; using all {len(roots)} episode(s)")
    return roots


def _load_episode_root(args: argparse.Namespace, root: Path) -> Episode:
    raw = root / "raw"
    # Existing ECCV v0 Allegro captures store arm arrays with ``object`` dtype.
    # The capture root is an explicitly selected local dataset, so load this
    # legacy representation and immediately convert it to numeric arrays.
    arm_poses = _load_array(raw / "arm" / "action.npy", allow_pickle=True).astype(np.float64)
    arm_times = _load_array(raw / "arm" / "time.npy", allow_pickle=True).astype(np.float64)
    if arm_poses.shape[0] != arm_times.shape[0]:
        count = min(arm_poses.shape[0], arm_times.shape[0])
        arm_poses, arm_times = arm_poses[:count], arm_times[:count]

    if args.robot != "allegro_v5":
        raise ValueError("This replay preview/executor currently supports --robot allegro_v5 only")
    hand_dir = raw / "hand"
    command_path = hand_dir / "action.npy"
    command_time_path = hand_dir / "time.npy"
    if not command_path.exists():
        raise FileNotFoundError(f"Allegro V5 action file not found: {command_path}")
    hand_commands = _as_allegro_v5_actions(_load_array(command_path), label="Allegro V5 hand action")
    hand_times = _load_array(command_time_path).astype(np.float64)

    pose_path = _episode_source_pose_path(root, None)
    source_object_world = _load_pose_file(pose_path, frame="world")
    source_c2r = _load_array(root / "C2R.npy")
    return Episode(root, arm_poses, arm_times, hand_commands, hand_times, source_object_world, _as_transform(source_c2r, label="source C2R"))


def _load_candidate_episodes(args: argparse.Namespace) -> list[Episode]:
    episodes: list[Episode] = []
    for root in _candidate_episode_roots(args.capture_root, args.robot, args.object):
        try:
            episodes.append(_load_episode_root(args, root))
        except (FileNotFoundError, ValueError) as exc:
            print(f"[match] skipping invalid episode {root.name}: {exc}")
    if not episodes:
        raise RuntimeError(f"no valid grasp episodes available for {args.object}")
    return episodes


def _rotation_error_rad(rotation_a: np.ndarray, rotation_b: np.ndarray) -> float:
    relative = np.asarray(rotation_a, dtype=np.float64).T @ np.asarray(rotation_b, dtype=np.float64)
    cosine = np.clip((np.trace(relative) - 1.0) / 2.0, -1.0, 1.0)
    return float(np.arccos(cosine))


def _rank_episode_matches(
    current_object_robot: np.ndarray,
    current_wrist_robot: np.ndarray,
    episodes: list[Episode],
    *,
    position_scale_m: float,
    rotation_scale_rad: float,
) -> list[EpisodeMatch]:
    """Match only before each episode's first minimum wrist-object distance."""

    if not episodes:
        raise ValueError("at least one candidate episode is required")
    if position_scale_m <= 0 or rotation_scale_rad <= 0:
        raise ValueError("matching position/rotation scales must be positive")
    current_object_robot = _as_transform(current_object_robot, label="current object pose")
    current_wrist_robot = _as_transform(current_wrist_robot, label="current wrist pose")
    current_relative = np.linalg.inv(current_object_robot) @ current_wrist_robot
    current_distance = float(np.linalg.norm(current_relative[:3, 3]))

    matches: list[EpisodeMatch] = []
    for episode in episodes:
        source_object_robot = np.linalg.inv(episode.source_c2r) @ episode.source_object_world
        relative_poses = np.linalg.inv(source_object_robot)[None] @ episode.arm_poses
        distances = np.linalg.norm(relative_poses[:, :3, 3], axis=1)
        grasp_frame_index = int(np.argmin(distances))
        if grasp_frame_index == 0:
            print(
                f"[match] skipping episode {episode.root.name}: "
                "minimum wrist-object distance is already at frame 0"
            )
            continue
        pregrasp_distances = distances[:grasp_frame_index]
        frame_index = int(np.argmin(np.abs(pregrasp_distances - current_distance)))
        candidate = relative_poses[frame_index]
        position_error = float(np.linalg.norm(candidate[:3, 3] - current_relative[:3, 3]))
        rotation_error = _rotation_error_rad(candidate[:3, :3], current_relative[:3, :3])
        matches.append(
            EpisodeMatch(
                episode=episode,
                frame_index=frame_index,
                grasp_frame_index=grasp_frame_index,
                wrist_object_distance_m=float(distances[frame_index]),
                distance_delta_m=float(abs(distances[frame_index] - current_distance)),
                position_error_m=position_error,
                rotation_error_rad=rotation_error,
                score=float(np.hypot(position_error / position_scale_m, rotation_error / rotation_scale_rad)),
            )
        )
    if not matches:
        raise RuntimeError("no episode contains frames before its minimum wrist-object distance")
    return sorted(matches, key=lambda match: match.score)


def _select_episode_match(
    current_object_robot: np.ndarray,
    current_wrist_robot: np.ndarray,
    episodes: list[Episode],
    *,
    position_scale_m: float,
    rotation_scale_rad: float,
) -> EpisodeMatch:
    return _rank_episode_matches(
        current_object_robot,
        current_wrist_robot,
        episodes,
        position_scale_m=position_scale_m,
        rotation_scale_rad=rotation_scale_rad,
    )[0]


def _episode_remainder(episode: Episode, frame_index: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if not 0 <= frame_index < len(episode.arm_poses):
        raise IndexError(f"matched frame {frame_index} outside episode of length {len(episode.arm_poses)}")
    hand_actions = _zoh_resample(episode.hand_times, episode.hand_commands, episode.arm_times)
    arm_poses = episode.arm_poses[frame_index:].copy()
    hand_actions = hand_actions[frame_index:].copy()
    arm_times = episode.arm_times[frame_index:].copy()
    arm_times -= arm_times[0]
    return arm_poses, hand_actions, arm_times


def _shared_rel_path(path: Path) -> str:
    path = path.resolve()
    shared = Path(shared_dir).resolve()
    try:
        return f"shared_data/{path.relative_to(shared).as_posix()}"
    except ValueError as exc:
        raise ValueError(f"capture path must live under {shared}: {path}") from exc


def _resolve_capture_object6d_mesh(mesh_name: str, mesh_root_dir: Path) -> Path:
    """Use the same viser-aligned mesh existence check as capture_object6d."""

    mesh_path = mesh_root_dir.expanduser() / mesh_name / f"{mesh_name}_viser.obj"
    if not mesh_path.is_file():
        raise FileNotFoundError(f"mesh not found: {mesh_path}")
    return mesh_path


def _capture_one_frame(save_dir: Path, rcc_entry: str) -> None:
    """Mirror the single-image capture section of ``capture_object6d.py``."""

    save_dir.mkdir(parents=True, exist_ok=True)
    save_current_camparam(str(save_dir))
    rcc = remote_camera_controller(rcc_entry)
    try:
        rcc.start("image", False, f"{_shared_rel_path(save_dir)}/raw")
        rcc.stop()
    finally:
        rcc.end()


def _send_rpc_once(addr: str, request: dict[str, Any], timeout_ms: int) -> dict[str, Any]:
    """Same request/reply semantics as ``capture_object6d._send_rpc_once``."""

    import zmq

    context = zmq.Context()
    socket = context.socket(zmq.REQ)
    socket.setsockopt(zmq.RCVTIMEO, timeout_ms)
    socket.setsockopt(zmq.SNDTIMEO, timeout_ms)
    socket.setsockopt(zmq.LINGER, 0)
    try:
        socket.connect(addr)
        socket.send_json(request)
        return socket.recv_json()
    finally:
        socket.close()
        context.term()


def _extract_object_6d_response(response: Mapping[str, Any]) -> dict[str, Any]:
    """Extract the pose payload in the capture_object6d RPC response shape."""

    object_6d = response.get("object_6d", response)
    if not isinstance(object_6d, Mapping) or object_6d.get("pose_world") is None:
        raise RuntimeError(f"invalid 6D response, pose_world missing: {response}")
    return {
        "pose_world": object_6d["pose_world"],
        "pose_left_cam": object_6d.get("pose_left_cam"),
        "R_world": object_6d.get("R_world"),
        "t_world": object_6d.get("t_world"),
    }


def _extract_rpc_timing(response: Mapping[str, Any]) -> dict[str, Any] | None:
    """Read timing metadata from either the response root or object payload."""

    timing = response.get("timing_seconds")
    if timing is None:
        object_6d = response.get("object_6d")
        if isinstance(object_6d, Mapping):
            timing = object_6d.get("timing_seconds")
    if timing is None:
        return None
    if not isinstance(timing, Mapping):
        print(f"[object6d timing] ignored invalid timing_seconds: {timing!r}")
        return None
    return {str(name): seconds for name, seconds in timing.items()}


def _print_rpc_timing(timing: Mapping[str, Any]) -> None:
    print("[object6d timing] remote inference breakdown:")
    for name, seconds in timing.items():
        try:
            value = f"{float(seconds):.3f} s"
        except (TypeError, ValueError):
            value = str(seconds)
        print(f"  {name}: {value}")


def _open_rpc_debug_images(debug_dir: Path) -> int:
    """Open the projection/overlay images written by the 6D RPC server."""

    paths = sorted((*debug_dir.glob("*.png"), *debug_dir.glob("*.jpg"), *debug_dir.glob("*.jpeg")))
    if not paths:
        print(f"[debug] RPC produced no overlay images in {debug_dir}")
        return 0
    for path in paths:
        subprocess.Popen(["xdg-open", str(path)], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    print(f"[debug] opened {len(paths)} RPC overlay image(s) from {debug_dir}")
    return len(paths)


def _debug_projection_camera_id(
    pose_world: np.ndarray,
    pose_left_cam: np.ndarray | None,
    camera_from_world: Mapping[str, np.ndarray],
) -> str:
    """Choose the captured camera whose world pose matches ``pose_left_cam``."""

    if not camera_from_world:
        raise ValueError("camera extrinsics are empty")
    pose_world = _as_transform(pose_world, label="debug pose_world")
    if pose_left_cam is None:
        return next(iter(camera_from_world))
    pose_left_cam = _as_transform(pose_left_cam, label="debug pose_left_cam")
    return min(
        camera_from_world,
        key=lambda camera_id: np.linalg.norm(
            camera_from_world[camera_id] @ pose_world - pose_left_cam
        ),
    )


def _write_local_projection_debug_image(
    save_dir: Path,
    *,
    mesh_name: str,
    mesh_root_dir: Path,
    pose_out: Mapping[str, Any],
) -> Path | None:
    """Render a replay-owned mesh projection when the pose RPC emits no PNG."""

    import cv2
    import trimesh

    intrinsics_path = save_dir / "cam_param" / "intrinsics.json"
    extrinsics_path = save_dir / "cam_param" / "extrinsics.json"
    mesh_path = mesh_root_dir / mesh_name / f"{mesh_name}.obj"
    if not intrinsics_path.is_file() or not extrinsics_path.is_file() or not mesh_path.is_file():
        print(
            "[debug] local projection skipped; missing camera calibration or mesh: "
            f"{intrinsics_path}, {extrinsics_path}, {mesh_path}"
        )
        return None

    intrinsics = json.loads(intrinsics_path.read_text(encoding="utf-8"))
    raw_extrinsics = json.loads(extrinsics_path.read_text(encoding="utf-8"))
    camera_from_world = {
        str(camera_id): _as_transform(value, label=f"camera extrinsic {camera_id}")
        for camera_id, value in raw_extrinsics.items()
    }
    pose_world = _as_transform(pose_out["pose_world"], label="debug pose_world")
    pose_left_cam = (
        _as_transform(pose_out["pose_left_cam"], label="debug pose_left_cam")
        if pose_out.get("pose_left_cam") is not None
        else None
    )
    camera_id = _debug_projection_camera_id(
        pose_world, pose_left_cam, camera_from_world
    )
    camera = intrinsics.get(camera_id)
    image_path = save_dir / "raw" / "images" / f"{camera_id}.png"
    if camera is None or not image_path.is_file():
        print(
            f"[debug] local projection skipped; no image/calibration for camera {camera_id}"
        )
        return None

    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        print(f"[debug] local projection skipped; cannot read {image_path}")
        return None
    mesh = trimesh.load(mesh_path, force="mesh", process=False)
    vertices = np.asarray(mesh.vertices, dtype=np.float64)
    faces = np.asarray(mesh.faces, dtype=np.int64)
    pose_camera = camera_from_world[camera_id] @ pose_world
    rvec, _ = cv2.Rodrigues(pose_camera[:3, :3])
    points_2d, _ = cv2.projectPoints(
        vertices,
        rvec,
        pose_camera[:3, 3],
        np.asarray(camera["original_intrinsics"], dtype=np.float64),
        np.asarray(camera.get("dist_params", ()), dtype=np.float64),
    )
    points_2d = np.rint(points_2d.reshape(-1, 2)).astype(np.int32)
    depth = vertices @ pose_camera[:3, :3].T + pose_camera[:3, 3]
    if len(faces) > 3000:
        faces = faces[np.linspace(0, len(faces) - 1, 3000, dtype=np.int64)]
    for face in faces:
        if np.all(depth[face, 2] > 0.01):
            cv2.polylines(image, [points_2d[face]], True, (0, 255, 255), 1, cv2.LINE_AA)

    axis = np.array([[0.0, 0.0, 0.0], [0.06, 0.0, 0.0], [0.0, 0.06, 0.0], [0.0, 0.0, 0.06]])
    axis_2d, _ = cv2.projectPoints(
        axis,
        rvec,
        pose_camera[:3, 3],
        np.asarray(camera["original_intrinsics"], dtype=np.float64),
        np.asarray(camera.get("dist_params", ()), dtype=np.float64),
    )
    origin, x_axis, y_axis, z_axis = np.rint(axis_2d.reshape(-1, 2)).astype(np.int32)
    for endpoint, color in ((x_axis, (0, 0, 255)), (y_axis, (0, 255, 0)), (z_axis, (255, 0, 0))):
        cv2.line(image, origin, endpoint, color, 3, cv2.LINE_AA)

    debug_dir = save_dir / "debug"
    debug_dir.mkdir(exist_ok=True)
    output_path = debug_dir / f"local_projection_{mesh_name}_{camera_id}.png"
    if not cv2.imwrite(str(output_path), image):
        raise RuntimeError(f"failed to save debug projection: {output_path}")
    print(f"[debug] wrote local object projection: {output_path}")
    return output_path


def _infer_capture_object6d_rpc(args: argparse.Namespace, save_dir: Path) -> np.ndarray:
    """Request one remote 6D inference and save the returned pose."""

    request = {
        "command": "infer",
        "image_path": _shared_rel_path(save_dir),
        "mesh_name": args.mesh_name,
    }
    response = _send_rpc_once(args.rpc_addr, request, timeout_ms=args.rpc_timeout_ms)
    pose_out = _extract_object_6d_response(response)
    timing = _extract_rpc_timing(response)
    if timing is not None:
        pose_out["timing_seconds"] = timing
        _print_rpc_timing(timing)
    (save_dir / "object_6d.json").write_text(
        json.dumps(pose_out, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    debug_dir = save_dir / "debug"
    if args.debug_image_mode != "none":
        projection_path = _write_local_projection_debug_image(
            save_dir,
            mesh_name=args.mesh_name,
            mesh_root_dir=Path(args.mesh_root_dir),
            pose_out=pose_out,
        )
        if projection_path is None:
            print(f"[debug] projection image was not generated; see messages above: {debug_dir}")
        elif args.debug_image_mode == "popup":
            _open_rpc_debug_images(debug_dir)
    else:
        print("[debug] image generation disabled (--debug-image-mode none)")
    return _as_transform(pose_out["pose_world"], label="capture_object6d RPC pose_world")


def _current_object_robot_pose(args: argparse.Namespace, save_dir: Path) -> tuple[np.ndarray, np.ndarray]:
    if args.current_object_pose:
        pose = _load_pose_file(Path(args.current_object_pose).expanduser(), frame=args.current_pose_frame)
        c2r = _as_transform(np.load(args.current_c2r_path), label="current C2R") if args.current_c2r_path else _as_transform(load_current_C2R(), label="current C2R")
        return (pose if args.current_pose_frame == "robot" else np.linalg.inv(c2r) @ pose), c2r

    _capture_one_frame(save_dir, args.rcc_entry)
    world_pose = _infer_capture_object6d_rpc(args, save_dir)
    c2r = _as_transform(load_current_C2R(), label="current C2R")
    np.save(save_dir / "C2R.npy", c2r)
    np.save(save_dir / "object_pose_world.npy", world_pose)
    return np.linalg.inv(c2r) @ world_pose, c2r


def _preview_indices(frame_count: int, max_frames: int) -> np.ndarray:
    if frame_count < 1:
        raise ValueError("Cannot preview an empty trajectory")
    if max_frames < 1:
        raise ValueError("--preview-max-frames must be positive")
    if frame_count <= max_frames:
        return np.arange(frame_count)
    return np.unique(np.linspace(0, frame_count - 1, max_frames, dtype=np.int64))


def _cartesian_approach_trajectory(
    current_pose: np.ndarray,
    target_pose: np.ndarray,
    *,
    linear_speed_mps: float,
    angular_speed_rps: float,
    min_seconds: float,
    rate_hz: float,
) -> tuple[np.ndarray, float]:
    """Generate a bounded-speed SE(3) path from the live arm pose to replay start."""

    from scipy.spatial.transform import Rotation, Slerp

    current_pose = _as_transform(current_pose, label="current xArm pose")
    target_pose = _as_transform(target_pose, label="replay start pose")
    translation_distance = float(np.linalg.norm(target_pose[:3, 3] - current_pose[:3, 3]))
    rotation_distance = float(
        Rotation.from_matrix(current_pose[:3, :3].T @ target_pose[:3, :3]).magnitude()
    )
    if translation_distance < 1e-8 and rotation_distance < 1e-8:
        return target_pose[None], 0.0

    seconds = max(
        min_seconds,
        translation_distance / linear_speed_mps,
        rotation_distance / angular_speed_rps,
    )
    segments = max(1, int(np.ceil(seconds * rate_hz)))
    alphas = np.linspace(0.0, 1.0, segments + 1, dtype=np.float64)[1:]
    rotations = Slerp(
        [0.0, 1.0], Rotation.from_matrix(np.stack((current_pose[:3, :3], target_pose[:3, :3])))
    )(alphas).as_matrix()
    poses = np.tile(np.eye(4, dtype=np.float64), (len(alphas), 1, 1))
    poses[:, :3, :3] = rotations
    poses[:, :3, 3] = (
        current_pose[:3, 3][None] * (1.0 - alphas[:, None])
        + target_pose[:3, 3][None] * alphas[:, None]
    )
    return poses, seconds


def _hand_approach_trajectory(
    current_qpos: np.ndarray,
    target_qpos: np.ndarray,
    *,
    frame_count: int,
) -> np.ndarray:
    """Linearly blend Allegro V5 joints over the xArm approach frames."""

    if frame_count < 1:
        raise ValueError("hand approach needs at least one frame")
    current_qpos = _as_allegro_v5_joint_vector(current_qpos, label="current Allegro V5 pose")
    target_qpos = _as_allegro_v5_joint_vector(target_qpos, label="replay-start Allegro V5 pose")
    alphas = np.linspace(0.0, 1.0, frame_count + 1, dtype=np.float64)[1:]
    return current_qpos[None] * (1.0 - alphas[:, None]) + target_qpos[None] * alphas[:, None]


def _hand_approach_slew_rate(
    current_qpos: np.ndarray,
    target_qpos: np.ndarray,
    *,
    seconds: float,
) -> np.ndarray:
    """Per-joint output-speed limit that reaches the replay start in ``seconds``."""

    if seconds <= 0.0:
        raise ValueError("hand approach duration must be positive")
    current_qpos = _as_allegro_v5_joint_vector(current_qpos, label="current Allegro V5 pose")
    target_qpos = _as_allegro_v5_joint_vector(target_qpos, label="replay-start Allegro V5 pose")
    return np.abs(target_qpos - current_qpos) / seconds


def _extend_stationary_approach(
    approach_poses: np.ndarray,
    *,
    required_frames: int,
) -> np.ndarray:
    """Keep the arm at replay start while the hand finishes its slow approach."""

    if required_frames <= len(approach_poses):
        return approach_poses
    return np.repeat(approach_poses[-1:], required_frames, axis=0)


def _compose_replay_trajectory(
    args: argparse.Namespace,
    *,
    live_arm_pose: np.ndarray,
    live_hand_qpos: np.ndarray,
    episode_arm_poses: np.ndarray,
    episode_hand_actions: np.ndarray,
    episode_arm_times: np.ndarray,
) -> ReplayTrajectory:
    """Prepend a timed current-state-to-selected-frame SE(3) transition."""

    live_arm_pose = _as_transform(live_arm_pose, label="current xArm pose")
    live_hand_qpos = _as_allegro_v5_joint_vector(
        live_hand_qpos, label="current Allegro V5 pose"
    )
    episode_arm_poses = np.asarray(episode_arm_poses, dtype=np.float64)
    episode_hand_actions = _as_allegro_v5_actions(
        episode_hand_actions, label="episode remainder hand action"
    )
    episode_arm_times = np.asarray(episode_arm_times, dtype=np.float64).reshape(-1)
    if len(episode_arm_poses) < 1:
        raise ValueError("episode remainder is empty")
    if len(episode_arm_poses) != len(episode_hand_actions) or len(episode_arm_poses) != len(episode_arm_times):
        raise ValueError("episode remainder arm/hand/time lengths differ")

    transition_targets, transition_seconds = _cartesian_approach_trajectory(
        live_arm_pose,
        episode_arm_poses[0],
        linear_speed_mps=args.approach_linear_speed_mps,
        angular_speed_rps=args.approach_angular_speed_rps,
        min_seconds=args.approach_min_seconds,
        rate_hz=args.approach_rate_hz,
    )
    hand_delta = float(np.max(np.abs(live_hand_qpos - episode_hand_actions[0])))
    if hand_delta > 1e-8:
        required_frames = max(
            1, int(np.ceil(args.approach_min_seconds * args.approach_rate_hz))
        )
        transition_targets = _extend_stationary_approach(
            transition_targets, required_frames=required_frames
        )
    transition_hand_targets = _hand_approach_trajectory(
        live_hand_qpos,
        episode_hand_actions[0],
        frame_count=len(transition_targets),
    )
    transition_seconds = max(
        transition_seconds,
        len(transition_targets) / args.approach_rate_hz,
    )
    transition_arm = np.concatenate((live_arm_pose[None], transition_targets), axis=0)
    transition_hand = np.concatenate(
        (live_hand_qpos[None], transition_hand_targets), axis=0
    )
    transition_times = np.linspace(
        0.0, transition_seconds, len(transition_arm), dtype=np.float64
    )

    scaled_episode_times = (episode_arm_times - episode_arm_times[0]) / args.rate_scale
    arm_poses = np.concatenate((transition_arm, episode_arm_poses[1:]), axis=0)
    hand_actions = np.concatenate((transition_hand, episode_hand_actions[1:]), axis=0)
    times = np.concatenate(
        (transition_times, transition_seconds + scaled_episode_times[1:]), axis=0
    )
    if np.any(np.diff(times) < 0):
        raise ValueError("composed replay timestamps are not monotonic")
    return ReplayTrajectory(
        arm_poses=arm_poses,
        hand_actions=hand_actions,
        times=times,
        transition_frame_count=len(transition_arm),
        transition_seconds=float(transition_seconds),
    )


def _prepend_preview_start_state(
    joint_trajectory: np.ndarray,
    arm_qpos: np.ndarray,
    hand_qpos: np.ndarray,
) -> np.ndarray:
    """Make Viser frame 0 equal to the real robot state before interpolation."""

    joint_trajectory = np.asarray(joint_trajectory, dtype=np.float64)
    if joint_trajectory.ndim != 2 or joint_trajectory.shape[1] != 22:
        raise ValueError(f"preview joint trajectory must have shape (N, 22), got {joint_trajectory.shape}")
    arm_qpos = np.asarray(arm_qpos, dtype=np.float64).reshape(-1)
    hand_qpos = _as_allegro_v5_joint_vector(hand_qpos, label="preview-start Allegro V5 pose")
    if arm_qpos.shape != (6,) or not np.all(np.isfinite(arm_qpos)):
        raise ValueError(f"preview-start xArm pose must have shape (6,), got {arm_qpos.shape}")
    return np.vstack((np.concatenate((arm_qpos, hand_qpos)), joint_trajectory))


def _preview_joint_trajectory(
    args: argparse.Namespace,
    episode: Episode,
    arm_poses: np.ndarray,
    hand_actions: np.ndarray,
    *,
    initial_arm_qpos: np.ndarray | None = None,
    label: str = "replay",
) -> np.ndarray:
    """Solve a downsampled, continuous IK path for mesh-only Viser preview."""

    from scipy.optimize import least_squares
    from scipy.spatial.transform import Rotation

    from paradex.visualization.robot import RobotModule

    robot = RobotModule(str(args.robot_urdf))
    joint_names = robot.get_joint_names()
    arm_joint_names = tuple(f"joint{i}" for i in range(1, 7))
    if tuple(joint_names[:6]) != arm_joint_names:
        raise ValueError(
            f"Preview URDF must begin with {arm_joint_names}, got {tuple(joint_names[:6])}: {args.robot_urdf}"
        )
    if robot.get_num_joints() != 22:
        raise ValueError(f"Preview URDF must expose 22 xArm+Allegro V5 joints, got {robot.get_num_joints()}")

    indices = _preview_indices(len(arm_poses), args.preview_max_frames)
    limits = robot.get_joint_limits()
    hand_joint_names = tuple(joint_names[6:])
    raw_hand_qpos = _as_allegro_v5_actions(
        hand_actions[indices], label="Allegro V5 preview hand action"
    )
    hand_qpos = _allegro_v5_preview_qpos(
        raw_hand_qpos,
        urdf_hand_joint_names=hand_joint_names,
        joint_limits=limits,
    )
    clipped_values = int(np.count_nonzero(~np.isclose(hand_qpos, raw_hand_qpos)))
    if clipped_values:
        print(
            f"[preview] clamped {clipped_values} Allegro value(s) to URDF physical limits"
        )
    lower = np.array([limits[name][0] for name in arm_joint_names], dtype=np.float64)
    upper = np.array([limits[name][1] for name in arm_joint_names], dtype=np.float64)
    if initial_arm_qpos is None:
        source_qpos = _load_array(
            episode.root / "raw" / "arm" / "position.npy", allow_pickle=True
        ).astype(np.float64)
        if source_qpos.ndim != 2 or source_qpos.shape[1] < 6:
            raise ValueError(f"Expected source arm positions with shape (N, >=6), got {source_qpos.shape}")
        if source_qpos.shape[0] != arm_poses.shape[0]:
            source_qpos = _zoh_resample(episode.arm_times, source_qpos, episode.arm_times)
        initial_arm_qpos = source_qpos[indices[0], :6]
    initial_arm_qpos = np.asarray(initial_arm_qpos, dtype=np.float64).reshape(-1)
    if initial_arm_qpos.shape != (6,) or not np.all(np.isfinite(initial_arm_qpos)):
        raise ValueError(f"Expected finite initial xArm joint state with shape (6,), got {initial_arm_qpos.shape}")
    last_qpos = np.clip(initial_arm_qpos, lower, upper)
    result = np.zeros((len(indices), 22), dtype=np.float64)
    failures = 0

    def residual(qpos: np.ndarray, target: np.ndarray) -> np.ndarray:
        robot.update_cfg(np.concatenate((qpos, np.zeros(16))))
        actual = robot.urdf.get_transform("link6", robot.urdf.base_link)
        position_error = (actual[:3, 3] - target[:3, 3]) / args.preview_position_scale
        rotation_error = Rotation.from_matrix(actual[:3, :3].T @ target[:3, :3]).as_rotvec() / args.preview_rotation_scale
        return np.concatenate((position_error, rotation_error))

    for preview_index, source_index in enumerate(indices):
        target = arm_poses[source_index]
        # Warm-start from the prior result to avoid IK branch flips.
        seed = last_qpos
        solution = least_squares(
            residual,
            seed,
            bounds=(lower, upper),
            args=(target,),
            max_nfev=args.preview_ik_max_nfev,
        )
        last_qpos = solution.x
        if np.linalg.norm(residual(last_qpos, target)) > 1.0:
            failures += 1
        result[preview_index, :6] = last_qpos
        result[preview_index, 6:] = hand_qpos[preview_index]

    print(
        f"[preview] {label} IK mesh trajectory: {len(indices)}/{len(arm_poses)} frames; "
        f"{failures} frame(s) exceed preview tolerance"
    )
    return result


def _load_preview_mesh(mesh_path: Path):
    import trimesh

    mesh = trimesh.load(mesh_path, force="mesh", process=False)
    if isinstance(mesh, trimesh.Trimesh):
        return mesh
    if isinstance(mesh, trimesh.Scene):
        return trimesh.util.concatenate(tuple(mesh.geometry.values()))
    raise ValueError(f"Unsupported object mesh: {mesh_path}")


def _xarm_position_to_transform(position: Any) -> np.ndarray:
    """Convert xArm ``get_position`` XYZ-mm/RPY-rad output to a transform."""

    from scipy.spatial.transform import Rotation

    position = np.asarray(position, dtype=np.float64).reshape(-1)
    if position.shape != (6,) or not np.all(np.isfinite(position)):
        raise ValueError(f"xArm position must have shape (6,), got {position.shape}")
    pose = np.eye(4, dtype=np.float64)
    pose[:3, 3] = position[:3] / 1000.0
    pose[:3, :3] = Rotation.from_euler("xyz", position[3:]).as_matrix()
    return pose


def _live_xarm_preview_state(*, timeout_seconds: float = 3.0) -> tuple[np.ndarray, np.ndarray]:
    """Read xArm state through ROS without changing its mode or sending a command."""

    import rclpy
    from rclpy.node import Node
    from sensor_msgs.msg import JointState
    from xarm_msgs.srv import GetFloat32List

    from paradex.utils.system import network_info

    namespace = str(network_info["xarm"]["param"].get("namespace", "")).strip("/")
    base = f"/{namespace}/xarm" if namespace else "/xarm"
    owns_rclpy = not rclpy.ok()
    if owns_rclpy:
        rclpy.init()
    node = Node(f"replay_object_relative_preview_{time.time_ns()}")
    latest_qpos: np.ndarray | None = None

    def on_joint_state(message: JointState) -> None:
        nonlocal latest_qpos
        qpos = np.asarray(message.position[:6], dtype=np.float64)
        if qpos.shape == (6,) and np.all(np.isfinite(qpos)):
            latest_qpos = qpos

    try:
        node.create_subscription(JointState, f"{base}/joint_states", on_joint_state, 10)
        position_client = node.create_client(GetFloat32List, f"{base}/get_position")
        deadline = time.monotonic() + timeout_seconds
        while not position_client.wait_for_service(timeout_sec=0.1):
            if time.monotonic() >= deadline:
                raise RuntimeError(f"timed out waiting for xArm service: {base}/get_position")
        future = position_client.call_async(GetFloat32List.Request())
        while time.monotonic() < deadline and (not future.done() or latest_qpos is None):
            rclpy.spin_once(node, timeout_sec=min(0.1, max(0.0, deadline - time.monotonic())))
        if not future.done():
            raise RuntimeError("timed out reading live xArm Cartesian pose for preview")
        response = future.result()
        if response is None or response.ret != 0 or len(response.datas) < 6:
            raise RuntimeError("xArm get_position failed while preparing preview")
        if latest_qpos is None:
            raise RuntimeError("timed out reading live xArm joint state for preview")
        return _xarm_position_to_transform(response.datas[:6]), latest_qpos
    finally:
        node.destroy_node()
        if owns_rclpy and rclpy.ok():
            rclpy.shutdown()


def _live_allegro_v5_qpos(hand: Any, *, timeout_seconds: float = 3.0) -> np.ndarray:
    if not hand.connection_event.wait(timeout_seconds):
        raise RuntimeError("timed out waiting for Allegro V5 joint state")
    return _as_allegro_v5_joint_vector(hand.get_data()["qpos"], label="live Allegro V5 pose")


def _live_allegro_v5_preview_qpos() -> np.ndarray:
    """Read the live hand state without publishing a hand command."""

    from paradex.io.robot_controller import get_hand

    hand = get_hand("allegro_v5", hand_side="right", command_enabled=False)
    try:
        return _live_allegro_v5_qpos(hand)
    finally:
        hand.end()


def _viser_object_pose(
    object_pose_robot: np.ndarray,
    viser_mesh_path: Path,
    *,
    apply_mesh_alignment: bool,
) -> np.ndarray:
    """Convert an original-mesh pose to the ``*_viser.obj`` mesh frame.

    FoundationPose returns the original object-mesh frame. ``*_viser.obj`` is
    the same mesh after ``*_viser_align.npy`` has been applied, so its local
    geometry needs the inverse transform when it is placed with that pose.
    This is display-only: it must not alter the object-relative arm plan.
    """

    object_pose_robot = _as_transform(object_pose_robot, label="object pose for Viser")
    if not apply_mesh_alignment:
        return object_pose_robot
    align_path = viser_mesh_path.with_name(f"{viser_mesh_path.stem}_align.npy")
    if not align_path.is_file():
        print(f"[preview] no Viser mesh alignment at {align_path}; using FoundationPose pose directly")
        return object_pose_robot
    viser_from_original = _as_transform(np.load(align_path), label=str(align_path))
    print(f"[preview] applying display-only inverse mesh alignment: {align_path}")
    return object_pose_robot @ np.linalg.inv(viser_from_original)


def _preview_replay(
    args: argparse.Namespace,
    match: EpisodeMatch,
    arm_poses: np.ndarray,
    hand_actions: np.ndarray,
    arm_times: np.ndarray,
    transition_frame_count: int,
    current_object_robot: np.ndarray,
    output_dir: Path,
    *,
    live_arm_pose: np.ndarray | None = None,
    live_arm_qpos: np.ndarray | None = None,
    live_hand_qpos: np.ndarray | None = None,
    object_pose_markers: dict[str, np.ndarray] | None = None,
) -> None:
    """Show the planned xArm+Allegro V5 mesh trajectory and current object in Viser."""

    from paradex.visualization.visualizer.viser import ViserViewer

    episode = match.episode

    if live_arm_pose is None or live_arm_qpos is None:
        live_arm_pose, live_arm_qpos = _live_xarm_preview_state()
    if live_hand_qpos is None:
        live_hand_qpos = _live_allegro_v5_preview_qpos()
    if not 2 <= transition_frame_count <= len(arm_poses):
        raise ValueError(
            f"invalid transition frame count {transition_frame_count} for {len(arm_poses)} poses"
        )
    transition_end = transition_frame_count - 1
    approach_poses = arm_poses[:transition_frame_count]
    approach_hand_actions = hand_actions[:transition_frame_count]
    replay_poses = arm_poses[transition_end:]
    replay_hand_actions = hand_actions[transition_end:]
    hand_delta = np.max(np.abs(approach_hand_actions[0] - approach_hand_actions[-1]))
    print(
        f"[preview] hand approach: {len(approach_hand_actions)} frames, "
        f"max joint delta={hand_delta:.3f} rad"
    )
    approach_joint_trajectory = _preview_joint_trajectory(
        args,
        episode,
        approach_poses,
        approach_hand_actions,
        initial_arm_qpos=live_arm_qpos,
        label=f"live arm+hand transition ({arm_times[transition_end]:.2f}s)",
    )
    replay_joint_trajectory = _preview_joint_trajectory(
        args,
        episode,
        replay_poses,
        replay_hand_actions,
        initial_arm_qpos=approach_joint_trajectory[-1, :6],
    )
    rendered_approach_hand = approach_joint_trajectory[:, 6:]
    np.savez_compressed(
        output_dir / "viser_preview_trajectory.npz",
        approach_joint_trajectory=approach_joint_trajectory,
        replay_joint_trajectory=replay_joint_trajectory,
        approach_hand_action=rendered_approach_hand,
        transition_frame_count=transition_frame_count,
        transition_seconds=arm_times[transition_end],
    )
    mesh_path = _resolve_capture_object6d_mesh(args.mesh_name, args.mesh_root_dir)
    object_pose_for_viser = _viser_object_pose(
        current_object_robot,
        mesh_path,
        apply_mesh_alignment=not args.no_viser_object_align,
    )
    viewer = ViserViewer(scene_title=f"Object-relative replay: {args.object}/{episode.root.name}")
    explicit_selection = hasattr(args, "episode")
    selection_folder = "Explicit episode" if explicit_selection else "Matched grasp"
    with viewer.server.gui.add_folder(selection_folder, expand_by_default=True):
        viewer.server.gui.add_text("Episode", initial_value=episode.root.name, disabled=True)
        viewer.server.gui.add_text("Start frame", initial_value=str(match.frame_index), disabled=True)
        viewer.server.gui.add_text(
            "Grasp-distance minimum frame",
            initial_value=str(match.grasp_frame_index),
            disabled=True,
        )
        viewer.server.gui.add_text(
            "Position error",
            initial_value=f"{match.position_error_m * 100.0:.1f} cm",
            disabled=True,
        )
        viewer.server.gui.add_text(
            "Rotation error",
            initial_value=f"{np.rad2deg(match.rotation_error_rad):.1f} deg",
            disabled=True,
        )
        if not explicit_selection:
            viewer.server.gui.add_text("Match score", initial_value=f"{match.score:.3f}", disabled=True)
        viewer.server.gui.add_text(
            "Transition",
            initial_value=f"{arm_times[transition_end]:.2f}s / {transition_frame_count} frames",
            disabled=True,
        )
    viewer.add_floor(height=0.0)
    viewer.add_robot("robot", str(args.robot_urdf), include_arm_meshes=True)
    viewer.robot_dict["robot"].update_cfg(approach_joint_trajectory[0])
    preview_mesh = _load_preview_mesh(mesh_path)
    viewer.add_object(args.object, preview_mesh, object_pose_for_viser)
    if object_pose_markers:
        for marker_name, marker_pose in object_pose_markers.items():
            marker_pose_for_viser = _viser_object_pose(
                _as_transform(marker_pose, label=f"preview marker {marker_name}"),
                mesh_path,
                apply_mesh_alignment=not args.no_viser_object_align,
            )
            viewer.add_object(
                f"{args.object}_{marker_name}",
                preview_mesh.copy(),
                marker_pose_for_viser,
                opacity=0.3,
            )
    # Keep them separate in Viser so the user can scrub the entire hand
    # approach before the episode trajectory begins.
    viewer.add_traj("live_arm_hand_approach", robot_traj={"robot": approach_joint_trajectory})
    viewer.add_traj("object_relative_replay", robot_traj={"robot": replay_joint_trajectory})
    # The server begins ticking before the browser connects.  Start paused at
    # frame zero so the live arm/hand state and its approach are inspectable.
    viewer.gui_playing.value = False
    viewer.gui_timestep.value = 0
    print(
        "[preview] Viser preview is paused at frame 0. Frames "
        f"0..{len(approach_joint_trajectory) - 1} are the live arm+hand approach; "
        f"frame {len(approach_joint_trajectory)} starts object-relative replay. "
        "Click Resume (or scrub Playback/Timestep) to inspect it; press Ctrl+C in this terminal to return."
    )
    viewer.start_viewer()


def _execute(
    args: argparse.Namespace,
    arm_poses: np.ndarray,
    hand_actions: np.ndarray,
    arm_times: np.ndarray,
    transition_frame_count: int,
    *,
    arm: Any | None = None,
    hand: Any | None = None,
) -> None:
    from paradex.io.robot_controller import get_arm, get_hand

    if (arm is None) != (hand is None):
        raise ValueError("arm and hand controllers must be supplied together")
    owns_controllers = arm is None
    if owns_controllers:
        arm = get_arm("xarm", servo_api="cartesian_aa")
        hand = get_hand("allegro_v5", hand_side="right")
    try:
        arm_poses = np.asarray(arm_poses, dtype=np.float64)
        hand_actions = _as_allegro_v5_actions(hand_actions, label="combined replay hand action")
        arm_times = np.asarray(arm_times, dtype=np.float64).reshape(-1)
        if len(arm_poses) != len(hand_actions) or len(arm_poses) != len(arm_times):
            raise ValueError("combined arm/hand/time lengths differ")
        if not 2 <= transition_frame_count <= len(arm_poses):
            raise ValueError("invalid transition frame count")
        transition_end = transition_frame_count - 1
        print(
            f"[execute] streaming the previewed plan: {transition_frame_count} transition "
            f"frames over {arm_times[transition_end]:.2f}s, then "
            f"{len(arm_poses) - transition_end} episode frames"
        )
        for i, pose in enumerate(arm_poses):
            arm.move(pose)
            hand.move(hand_actions[i])
            if arm.is_error() or hand.is_error():
                raise RuntimeError(f"controller error at frame {i}")
            if i == transition_end and args.settle_seconds > 0:
                time.sleep(args.settle_seconds)
            if i + 1 < len(arm_poses):
                dt = max(0.0, arm_times[i + 1] - arm_times[i])
                time.sleep(dt)
    finally:
        if owns_controllers:
            arm.end()
            hand.end()


def _teleoperate_until_trigger(args: argparse.Namespace):
    """Run existing VIVE+MANUS teleoperation until the replay key is pressed."""

    from threading import Event

    from paradex.dataset_acqusition.capture import CaptureSession

    trigger_event = Event()
    exit_event = Event()
    events = {"save": trigger_event, "stop": Event(), "exit": exit_event}
    stdin_open = True
    terminal_settings = None
    terminal_fd = None

    def poll_keyboard(_session: Any) -> None:
        nonlocal stdin_open
        if not stdin_open:
            return
        readable, _, _ = select.select([sys.stdin], [], [], 0.0)
        if not readable:
            return
        value = sys.stdin.read(1) if terminal_settings is not None else sys.stdin.readline()
        if value == "":
            stdin_open = False
            return
        key = value.strip().lower()
        if key == args.replay_key:
            trigger_event.set()
        elif key == "q":
            exit_event.set()
        elif key:
            print(f"[teleop] unknown key {key!r}; use {args.replay_key!r} or 'q'")

    session = CaptureSession(
        camera=False,
        realsense=False,
        arm="xarm",
        hand="allegro_v5",
        teleop="vive",
        hand_side="right",
        events=events,
        timestamp=False,
        arm_kwargs={"servo_api": "cartesian_aa"},
        hand_command_rate_hz=args.allegro_command_rate_hz,
        use_vive=True,
        use_manus=True,
        require_left_control=True,
    )
    try:
        if sys.stdin.isatty():
            import termios
            import tty

            terminal_fd = sys.stdin.fileno()
            terminal_settings = termios.tcgetattr(terminal_fd)
            tty.setcbreak(terminal_fd)
        print(
            f"[teleop] VIVE+MANUS teleoperation active. Press {args.replay_key!r} "
            "to match/replay a grasp, or 'q' to exit."
        )
        state = session.teleop(
            session_events=events,
            state_policy="keyboard_control",
            loop_callback=poll_keyboard,
        )
        if hasattr(session, "retargetor"):
            session.retargetor.stop()
        return session, state
    except Exception:
        session.end()
        raise
    finally:
        if terminal_settings is not None and terminal_fd is not None:
            import termios

            termios.tcsetattr(terminal_fd, termios.TCSADRAIN, terminal_settings)


def _read_live_robot_state(arm: Any, hand: Any) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Read state from already-open xArm and Allegro controller instances."""

    arm_state = arm.get_data()
    arm_pose = _as_transform(arm_state["position"], label="current xArm pose")
    arm_qpos = np.asarray(arm_state["qpos"], dtype=np.float64)
    if arm_qpos.shape != (6,) or not np.all(np.isfinite(arm_qpos)):
        raise RuntimeError("current xArm joint state is unavailable")
    hand_qpos = _live_allegro_v5_qpos(hand)
    return arm_pose, arm_qpos, hand_qpos


def _read_live_robot_state_once() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Read xArm and Allegro state without starting VIVE/MANUS teleoperation."""

    from paradex.io.robot_controller import get_arm, get_hand

    arm = None
    hand = None
    try:
        arm = get_arm("xarm", servo_api="cartesian_aa")
        hand = get_hand("allegro_v5", hand_side="right", command_enabled=False)
        return _read_live_robot_state(arm, hand)
    finally:
        if hand is not None:
            hand.end()
        if arm is not None:
            arm.end()


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--robot", default="allegro_v5", choices=("allegro_v5",), help="source robot folder under capture root")
    parser.add_argument("--object", required=True, help="object/mesh name")
    parser.add_argument("--capture-root", type=Path, default=DEFAULT_CAPTURE_ROOT)
    parser.add_argument("--mesh-name", default=None, help="tracker mesh name (defaults to --object)")
    parser.add_argument("--mesh-root-dir", type=Path, default=DEFAULT_MESH_ROOT,
                        help="mesh root containing source .obj and Viser-aligned .obj files")
    parser.add_argument("--robot-urdf", type=Path, default=DEFAULT_ROBOT_URDF,
                        help="combined xArm + Allegro V5 URDF used for preview")
    parser.add_argument("--current-object-pose", default=None, help="skip capture/tracking and load a current pose")
    parser.add_argument("--current-pose-frame", choices=("world", "robot"), default="world")
    parser.add_argument("--current-c2r-path", default=None, help="required only to convert a supplied world pose with non-current calibration")
    parser.add_argument("--rcc-entry", default="image_main.py")
    parser.add_argument("--rpc-addr", default="tcp://192.168.0.14:5570",
                        help="remote 6D pose service used by capture_object6d.py")
    parser.add_argument("--rpc-timeout-ms", type=int, default=300000)
    parser.add_argument("--debug-image-mode", choices=("none", "save", "popup"), default="save",
                        help="save a pose projection after inference, optionally opening it")
    parser.add_argument("--preview", action="store_true",
                        help="open a Viser xArm+Allegro V5 mesh preview before returning")
    parser.add_argument("--preview-max-frames", type=int, default=150,
                        help="maximum IK-solved mesh frames sent to Viser")
    parser.add_argument("--preview-ik-max-nfev", type=int, default=50,
                        help="maximum numerical IK evaluations per preview frame")
    parser.add_argument("--preview-position-scale", type=float, default=0.05,
                        help="metres represented by one unit of preview IK position residual")
    parser.add_argument("--preview-rotation-scale", type=float, default=0.5,
                        help="radians represented by one unit of preview IK rotation residual")
    parser.add_argument("--no-viser-object-align", action="store_true",
                        help="show the *_viser.obj mesh with the unadjusted FoundationPose pose (debug only)")
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--replay-key", default="r",
                        help="keyboard key that stops teleoperation and selects a grasp")
    parser.add_argument("--match-position-scale-m", type=float, default=0.05,
                        help="translation normalization used in selected-frame 6D matching")
    parser.add_argument("--match-rotation-scale-rad", type=float, default=0.5,
                        help="rotation normalization used in selected-frame 6D matching")
    parser.add_argument("--allegro-command-rate-hz", type=float, default=30.0,
                        help="Allegro V5 command rate during live teleoperation")
    parser.add_argument("--rate-scale", type=float, default=1.0)
    
    parser.add_argument("--approach-linear-speed-mps", type=float, default=0.5,
                        help="maximum xArm translation speed while approaching replay start")
    parser.add_argument("--approach-angular-speed-rps", type=float, default=2.0,
                        help="maximum xArm angular speed while approaching replay start")
    parser.add_argument("--approach-min-seconds", type=float, default=1.0,
                        help="minimum duration of the initial current-pose-to-replay approach")
    parser.add_argument("--approach-rate-hz", type=float, default=50.0,
                        help="Cartesian command rate for the initial approach")
    parser.add_argument("--settle-seconds", type=float, default=1.0)
    parser.add_argument(
        "--plan-only",
        action="store_true",
        help=(
            "skip VIVE/MANUS teleoperation, read the current robot state once, "
            "save/preview the matched trajectory, and never command motion"
        ),
    )
    parser.add_argument("--execute", action="store_true", help="enable real robot/hand commands (off by default)")
    return parser


def main() -> None:
    parser = _parser()
    args = parser.parse_args()
    if args.plan_only and args.execute:
        parser.error("--plan-only and --execute are mutually exclusive")
    if not args.plan_only and not args.execute:
        parser.error("live teleoperation and grasp replay require the explicit --execute flag")
    if args.rate_scale <= 0:
        raise ValueError("--rate-scale must be positive")
    args.replay_key = args.replay_key.strip().lower()
    if len(args.replay_key) != 1 or args.replay_key == "q":
        raise ValueError("--replay-key must be one character other than 'q'")
    if args.match_position_scale_m <= 0 or args.match_rotation_scale_rad <= 0:
        raise ValueError("matching position/rotation scales must be positive")
    if args.allegro_command_rate_hz <= 0:
        raise ValueError("--allegro-command-rate-hz must be positive")
    if args.rpc_timeout_ms <= 0:
        raise ValueError("--rpc-timeout-ms must be positive")
    if args.preview_position_scale <= 0 or args.preview_rotation_scale <= 0:
        raise ValueError("preview residual scales must be positive")
    if args.approach_linear_speed_mps <= 0 or args.approach_angular_speed_rps <= 0:
        raise ValueError("approach speeds must be positive")
    if args.approach_min_seconds < 0 or args.approach_rate_hz <= 0:
        raise ValueError("approach duration/rate must be non-negative/positive")
    args.mesh_name = args.mesh_name or args.object
    _resolve_capture_object6d_mesh(args.mesh_name, args.mesh_root_dir)
    if not args.robot_urdf.is_file():
        raise FileNotFoundError(f"preview robot URDF not found: {args.robot_urdf}")
    episodes = _load_candidate_episodes(args)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = args.output_dir or (
        Path(shared_dir)
        / "capture"
        / "eccv2026"
        / "relative_replay"
        / args.robot
        / args.object
        / f"matched_{timestamp}"
    )
    output_dir = output_dir.expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)

    current_object_robot, current_c2r = _current_object_robot_pose(args, output_dir / "initial_object_capture")
    session = None
    try:
        if args.plan_only:
            print("[plan-only] reading xArm and Allegro state without VIVE/MANUS")
            live_arm_pose, live_arm_qpos, live_hand_qpos = _read_live_robot_state_once()
        else:
            session, teleop_state = _teleoperate_until_trigger(args)
            if teleop_state == "exit":
                print("[teleop] exited; no replay trajectory generated.")
                return

            arm_state = session.arm.get_data()
            live_arm_pose = _as_transform(arm_state["position"], label="trigger-time xArm pose")
            live_arm_qpos = np.asarray(arm_state["qpos"], dtype=np.float64)
            if live_arm_qpos.shape != (6,) or not np.all(np.isfinite(live_arm_qpos)):
                raise RuntimeError("trigger-time xArm joint state is unavailable")
            live_hand_qpos = _live_allegro_v5_qpos(session.hand)
        ranked_matches = _rank_episode_matches(
            current_object_robot,
            live_arm_pose,
            episodes,
            position_scale_m=args.match_position_scale_m,
            rotation_scale_rad=args.match_rotation_scale_rad,
        )
        match = ranked_matches[0]
        print("[match] candidate ranking:")
        for rank, candidate_match in enumerate(ranked_matches, start=1):
            print(
                f"  {rank}. episode {candidate_match.episode.root.name}, "
                f"frame {candidate_match.frame_index} "
                f"(before distance minimum {candidate_match.grasp_frame_index}): "
                f"position={candidate_match.position_error_m:.4f} m, "
                f"rotation={np.rad2deg(candidate_match.rotation_error_rad):.1f} deg, "
                f"distance_delta={candidate_match.distance_delta_m:.4f} m, "
                f"score={candidate_match.score:.3f}"
            )
        episode = match.episode
        source_arm_poses, episode_hand_actions, episode_arm_times = _episode_remainder(
            episode, match.frame_index
        )
        source_object_robot = np.linalg.inv(episode.source_c2r) @ episode.source_object_world
        episode_arm_poses = relative_arm_actions(
            source_object_robot, current_object_robot, source_arm_poses
        )
        trajectory = _compose_replay_trajectory(
            args,
            live_arm_pose=live_arm_pose,
            live_hand_qpos=live_hand_qpos,
            episode_arm_poses=episode_arm_poses,
            episode_hand_actions=episode_hand_actions,
            episode_arm_times=episode_arm_times,
        )
        arm_poses = trajectory.arm_poses
        hand_actions = trajectory.hand_actions
        arm_times = trajectory.times
        np.savez_compressed(
            output_dir / "relative_replay_plan.npz",
            arm_action=arm_poses,
            arm_time=arm_times,
            hand_action=hand_actions,
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
                [candidate.episode.root.name for candidate in ranked_matches]
            ),
            candidate_frame=np.asarray(
                [candidate.frame_index for candidate in ranked_matches], dtype=np.int64
            ),
            candidate_grasp_distance_minimum_frame=np.asarray(
                [candidate.grasp_frame_index for candidate in ranked_matches], dtype=np.int64
            ),
            candidate_position_error_m=np.asarray(
                [candidate.position_error_m for candidate in ranked_matches]
            ),
            candidate_rotation_error_rad=np.asarray(
                [candidate.rotation_error_rad for candidate in ranked_matches]
            ),
            candidate_score=np.asarray([candidate.score for candidate in ranked_matches]),
        )
        print(
            f"[match] selected episode {episode.root.name}, frame {match.frame_index}; "
            f"distance minimum/grasp frame is {match.grasp_frame_index}"
        )
        print(
            f"[match] distance delta={match.distance_delta_m:.4f} m, "
            f"position error={match.position_error_m:.4f} m, "
            f"rotation error={np.rad2deg(match.rotation_error_rad):.1f} deg, "
            f"score={match.score:.3f}"
        )
        print(f"[plan] saved {output_dir / 'relative_replay_plan.npz'}")
        print(
            f"[plan] current state -> selected frame transition: "
            f"{trajectory.transition_frame_count} frames over "
            f"{trajectory.transition_seconds:.2f}s"
        )
        print("[plan] source object robot xyz:", np.round(source_object_robot[:3, 3], 4).tolist())
        print("[plan] current object robot xyz:", np.round(current_object_robot[:3, 3], 4).tolist())
        print("[plan] delta xyz:", np.round((current_object_robot @ np.linalg.inv(source_object_robot))[:3, 3], 4).tolist())

        if args.preview:
            _preview_replay(
                args,
                match,
                arm_poses,
                hand_actions,
                arm_times,
                trajectory.transition_frame_count,
                current_object_robot,
                output_dir,
                live_arm_pose=live_arm_pose,
                live_arm_qpos=live_arm_qpos,
                live_hand_qpos=live_hand_qpos,
            )
        if args.plan_only:
            print("[plan-only] complete; no robot or hand motion commands were sent.")
            return
        prompt = (
            "Preview complete. Type PLAY to execute the robot trajectory: "
            if args.preview
            else "Trajectory ready. Type PLAY to execute the robot trajectory: "
        )
        answer = input(prompt).strip()
        if answer != "PLAY":
            print("[execute] cancelled; no robot commands sent.")
            return
        _execute(
            args,
            arm_poses,
            hand_actions,
            arm_times,
            trajectory.transition_frame_count,
            arm=session.arm,
            hand=session.hand,
        )
    finally:
        if session is not None:
            session.end()


if __name__ == "__main__":
    main()
