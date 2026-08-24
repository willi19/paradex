"""Replay an ECCV v0 robot episode in the current object's coordinate frame.

The recorded arm action is a Cartesian end-effector transform.  Given the
recorded object pose ``T_source_object`` and a newly captured initial object
pose ``T_current_object``, every arm target is transformed as::

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
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping

import numpy as np

# Allow ``python src/dataset_acquisition/hri/replay_object_relative.py`` from
# any working directory, matching the existing HRI entry points.
PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from paradex.calibration.utils import load_current_C2R, save_current_camparam
from paradex.io.camera_system.remote_camera_controller import remote_camera_controller
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


def _episode_source_pose_path(root: Path, override: str | None) -> Path:
    if override:
        return Path(override).expanduser()
    # ``object_6d_pose_v2.npz`` uses the same object-frame convention as the
    # current one-shot capture RPC.  Prefer it when an episode has both files;
    # the v1 pose would otherwise introduce a spurious rigid rotation.
    for candidate in (
        root / "object_6d" / "pose_000000.txt",
        root / "object_6d_pose_v2.npz",
        root / "object_6d_pose.npz",
    ):
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(
        "source object frame-0 pose not found; expected "
        f"{root / 'object_6d' / 'pose_000000.txt'}, "
        f"{root / 'object_6d_pose_v2.npz'}, or {root / 'object_6d_pose.npz'}"
    )


def _load_episode(args: argparse.Namespace) -> Episode:
    root = Path(args.capture_root).expanduser() / args.robot / args.object / str(args.episode)
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

    pose_path = _episode_source_pose_path(root, args.source_object_pose)
    source_object_world = _load_pose_file(pose_path, frame="world")
    source_c2r = _load_array(root / "C2R.npy")
    return Episode(root, arm_poses, arm_times, hand_commands, hand_times, source_object_world, _as_transform(source_c2r, label="source C2R"))


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
    """Render a replay-owned mesh projection when the pose RPC emits no PNG.

    The 6D service returns a pose in the ``left`` camera frame but recent
    server versions no longer persist their own overlay image.  The raw image,
    calibration, and pose are already captured locally, so render a wireframe
    projection here instead of depending on that server-side side effect.
    """

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
    """Run the one-shot RPC inference and save its raw response alongside capture."""

    request = {
        "command": "infer",
        "image_path": _shared_rel_path(save_dir),
        "mesh_name": args.mesh_name,
    }
    pose_out = _extract_object_6d_response(
        _send_rpc_once(args.rpc_addr, request, timeout_ms=args.rpc_timeout_ms)
    )
    (save_dir / "object_6d.json").write_text(
        json.dumps(pose_out, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    debug_dir = save_dir / "debug"
    if args.debug_image_mode == "popup":
        _write_local_projection_debug_image(
            save_dir,
            mesh_name=args.mesh_name,
            mesh_root_dir=Path(args.mesh_root_dir),
            pose_out=pose_out,
        )
        _open_rpc_debug_images(debug_dir)
    else:
        print(f"[debug] RPC projection/overlay path: {debug_dir}")
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
    hand_qpos = _as_allegro_v5_actions(hand_actions[indices], label="Allegro V5 preview hand action")
    limits = robot.get_joint_limits()
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
    """Convert an RPC original-mesh pose to the ``*_viser.obj`` mesh frame.

    The pose RPC returns the original object-mesh frame.  ``*_viser.obj`` is
    the same mesh after ``*_viser_align.npy`` has been applied, so its local
    geometry needs the inverse transform when it is placed with that pose.
    This is display-only: it must not alter the object-relative arm plan.
    """

    object_pose_robot = _as_transform(object_pose_robot, label="object pose for Viser")
    if not apply_mesh_alignment:
        return object_pose_robot
    align_path = viser_mesh_path.with_name(f"{viser_mesh_path.stem}_align.npy")
    if not align_path.is_file():
        print(f"[preview] no Viser mesh alignment at {align_path}; using RPC pose directly")
        return object_pose_robot
    viser_from_original = _as_transform(np.load(align_path), label=str(align_path))
    print(f"[preview] applying display-only inverse mesh alignment: {align_path}")
    return object_pose_robot @ np.linalg.inv(viser_from_original)


def _preview_replay(
    args: argparse.Namespace,
    episode: Episode,
    arm_poses: np.ndarray,
    hand_actions: np.ndarray,
    current_object_robot: np.ndarray,
    output_dir: Path,
) -> None:
    """Show the planned xArm+Allegro V5 mesh trajectory and current object in Viser."""

    from paradex.visualization.visualizer.viser import ViserViewer

    live_arm_pose, live_arm_qpos = _live_xarm_preview_state()
    live_hand_qpos = _live_allegro_v5_preview_qpos()
    approach_poses, approach_seconds = _cartesian_approach_trajectory(
        live_arm_pose,
        arm_poses[0],
        linear_speed_mps=args.approach_linear_speed_mps,
        angular_speed_rps=args.approach_angular_speed_rps,
        min_seconds=args.approach_min_seconds,
        rate_hz=args.approach_rate_hz,
    )
    hand_delta = np.max(np.abs(live_hand_qpos - hand_actions[0]))
    if hand_delta > 1e-8:
        minimum_hand_frames = max(1, int(np.ceil(args.approach_min_seconds * args.approach_rate_hz)))
        approach_poses = _extend_stationary_approach(
            approach_poses, required_frames=minimum_hand_frames
        )
    approach_hand_actions = _hand_approach_trajectory(
        live_hand_qpos, hand_actions[0], frame_count=len(approach_poses)
    )
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
        label=f"live arm+hand approach ({max(approach_seconds, len(approach_poses) / args.approach_rate_hz):.2f}s)",
    )
    approach_joint_trajectory = _prepend_preview_start_state(
        approach_joint_trajectory, live_arm_qpos, live_hand_qpos
    )
    replay_joint_trajectory = _preview_joint_trajectory(args, episode, arm_poses, hand_actions)
    expected_approach_hand = np.vstack(
        (
            live_hand_qpos,
            approach_hand_actions[_preview_indices(len(approach_hand_actions), args.preview_max_frames)],
        )
    )
    if not np.allclose(approach_joint_trajectory[:, 6:], expected_approach_hand):
        raise RuntimeError("preview hand trajectory does not match the generated hand approach")
    np.savez_compressed(
        output_dir / "viser_preview_trajectory.npz",
        approach_joint_trajectory=approach_joint_trajectory,
        replay_joint_trajectory=replay_joint_trajectory,
        approach_hand_action=expected_approach_hand,
    )
    mesh_path = _resolve_capture_object6d_mesh(args.mesh_name, args.mesh_root_dir)
    object_pose_for_viser = _viser_object_pose(
        current_object_robot,
        mesh_path,
        apply_mesh_alignment=not args.no_viser_object_align,
    )
    viewer = ViserViewer(scene_title=f"Object-relative replay: {args.object}/{args.episode}")
    viewer.add_floor(height=0.0)
    viewer.add_robot("robot", str(args.robot_urdf), include_arm_meshes=True)
    viewer.robot_dict["robot"].update_cfg(approach_joint_trajectory[0])
    viewer.add_object(args.object, _load_preview_mesh(mesh_path), object_pose_for_viser)
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


def _execute(args: argparse.Namespace, arm_poses: np.ndarray, hand_actions: np.ndarray, arm_times: np.ndarray) -> None:
    from paradex.io.robot_controller import get_arm, get_hand

    arm = get_arm("xarm", servo_api="cartesian_aa")
    hand = get_hand("allegro_v5", hand_side="right")
    try:
        current_arm_pose = _as_transform(arm.get_data()["position"], label="live xArm pose")
        current_hand_qpos = _live_allegro_v5_qpos(hand)
        approach_poses, approach_seconds = _cartesian_approach_trajectory(
            current_arm_pose,
            arm_poses[0],
            linear_speed_mps=args.approach_linear_speed_mps,
            angular_speed_rps=args.approach_angular_speed_rps,
            min_seconds=args.approach_min_seconds,
            rate_hz=args.approach_rate_hz,
        )
        hand_delta = np.max(np.abs(current_hand_qpos - hand_actions[0]))
        if hand_delta > 1e-8:
            minimum_hand_frames = max(1, int(np.ceil(args.approach_min_seconds * args.approach_rate_hz)))
            approach_poses = _extend_stationary_approach(
                approach_poses, required_frames=minimum_hand_frames
            )
        approach_hand_actions = _hand_approach_trajectory(
            current_hand_qpos, hand_actions[0], frame_count=len(approach_poses)
        )
        approach_duration = max(
            approach_seconds,
            len(approach_poses) / args.approach_rate_hz,
        )
        hand.set_command_slew_rate(
            _hand_approach_slew_rate(
                current_hand_qpos,
                hand_actions[0],
                seconds=approach_duration,
            ),
            initial_action=current_hand_qpos,
        )
        print(
            f"[execute] approach: {len(approach_poses)} Cartesian frames over "
            f"{approach_duration:.2f}s before replay"
        )
        for i, pose in enumerate(approach_poses):
            arm.move(pose)
            hand.move(approach_hand_actions[i])
            if arm.is_error() or hand.is_error():
                raise RuntimeError(f"controller error during approach frame {i}")
            if i + 1 < len(approach_poses):
                time.sleep(1.0 / args.approach_rate_hz)

        if not hand.wait_for_published_action(hand_actions[0], timeout_seconds=1.0):
            raise RuntimeError("Allegro V5 did not publish the replay-start hand target")
        hand.set_command_slew_rate(None)

        print(f"[execute] streaming {len(arm_poses)} object-relative Cartesian frames")
        if args.settle_seconds > 0:
            time.sleep(args.settle_seconds)
        for i, pose in enumerate(arm_poses):
            arm.move(pose)
            hand.move(hand_actions[i])
            if arm.is_error() or hand.is_error():
                raise RuntimeError(f"controller error at frame {i}")
            if i + 1 < len(arm_poses):
                dt = max(0.0, (arm_times[i + 1] - arm_times[i]) / args.rate_scale)
                time.sleep(dt)
    finally:
        arm.end()
        hand.end()


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--robot", default="allegro_v5", choices=("allegro_v5",), help="source robot folder under capture root")
    parser.add_argument("--object", required=True, help="object/mesh name")
    parser.add_argument("--episode", required=True, type=int)
    parser.add_argument("--capture-root", type=Path, default=DEFAULT_CAPTURE_ROOT)
    parser.add_argument("--mesh-name", default=None, help="tracker mesh name (defaults to --object)")
    parser.add_argument("--source-object-pose", default=None, help="override source frame-0 pose file")
    parser.add_argument("--mesh-root-dir", type=Path, default=DEFAULT_MESH_ROOT,
                        help="viser-aligned mesh root checked before the RPC request")
    parser.add_argument("--robot-urdf", type=Path, default=DEFAULT_ROBOT_URDF,
                        help="combined xArm + Allegro V5 URDF used for preview")
    parser.add_argument("--current-object-pose", default=None, help="skip capture/tracking and load a current pose")
    parser.add_argument("--current-pose-frame", choices=("world", "robot"), default="world")
    parser.add_argument("--current-c2r-path", default=None, help="required only to convert a supplied world pose with non-current calibration")
    parser.add_argument("--rcc-entry", default="image_main.py")
    parser.add_argument("--rpc-addr", default="tcp://192.168.0.14:5570",
                        help="6D pose service used by capture_object6d.py")
    parser.add_argument("--rpc-timeout-ms", type=int, default=300000)
    parser.add_argument("--debug-image-mode", choices=("none", "popup"), default="none",
                        help="open RPC-produced object projection/overlay images after inference")
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
                        help="show the *_viser.obj mesh with the unadjusted RPC pose (debug only)")
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--rate-scale", type=float, default=1.0)
    parser.add_argument("--approach-linear-speed-mps", type=float, default=0.05,
                        help="maximum xArm translation speed while approaching replay start")
    parser.add_argument("--approach-angular-speed-rps", type=float, default=0.5,
                        help="maximum xArm angular speed while approaching replay start")
    parser.add_argument("--approach-min-seconds", type=float, default=5.0,
                        help="minimum duration of the initial current-pose-to-replay approach")
    parser.add_argument("--approach-rate-hz", type=float, default=50.0,
                        help="Cartesian command rate for the initial approach")
    parser.add_argument("--settle-seconds", type=float, default=1.0)
    parser.add_argument("--execute", action="store_true", help="enable real robot/hand commands (off by default)")
    return parser


def main() -> None:
    args = _parser().parse_args()
    if args.rate_scale <= 0:
        raise ValueError("--rate-scale must be positive")
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
    episode = _load_episode(args)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = args.output_dir or (Path(shared_dir) / "capture" / "eccv2026" / "relative_replay" / args.robot / args.object / f"{args.episode}_{timestamp}")
    output_dir = output_dir.expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)

    current_object_robot, current_c2r = _current_object_robot_pose(args, output_dir / "initial_object_capture")
    source_object_robot = np.linalg.inv(episode.source_c2r) @ episode.source_object_world
    arm_poses = relative_arm_actions(source_object_robot, current_object_robot, episode.arm_poses)
    hand_actions = _zoh_resample(episode.hand_times, episode.hand_commands, episode.arm_times)

    np.savez_compressed(
        output_dir / "relative_replay_plan.npz",
        arm_action=arm_poses,
        arm_time=episode.arm_times,
        hand_action=hand_actions,
        source_object_robot=source_object_robot,
        current_object_robot=current_object_robot,
        object_delta=current_object_robot @ np.linalg.inv(source_object_robot),
        source_c2r=episode.source_c2r,
        current_c2r=current_c2r,
    )
    print(f"[plan] saved {output_dir / 'relative_replay_plan.npz'}")
    print("[plan] source object robot xyz:", np.round(source_object_robot[:3, 3], 4).tolist())
    print("[plan] current object robot xyz:", np.round(current_object_robot[:3, 3], 4).tolist())
    print("[plan] delta xyz:", np.round((current_object_robot @ np.linalg.inv(source_object_robot))[:3, 3], 4).tolist())
    if args.preview or args.execute:
        _preview_replay(args, episode, arm_poses, hand_actions, current_object_robot, output_dir)
    if args.execute:
        answer = input("Preview complete. Type PLAY to execute the robot trajectory: ").strip()
        if answer != "PLAY":
            print("[execute] cancelled; no robot commands sent.")
            return
        _execute(args, arm_poses, hand_actions, episode.arm_times)
    else:
        print("[plan] Dry run complete. Re-run with --execute to command the robot.")


if __name__ == "__main__":
    main()
