"""Capture an object pose, preview a staged grasp, and optionally execute it."""

from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
import time
import xml.etree.ElementTree as ET
from pathlib import Path
from threading import Event
from typing import Any, Dict, Mapping

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from paradex.calibration.utils import load_current_C2R, save_current_camparam
from paradex.io.camera_system.remote_camera_controller import remote_camera_controller
from paradex.utils.path import shared_dir


DEFAULT_MESH_ROOT = Path(shared_dir) / "mesh_new"
DEFAULT_HAND_URDF = (
    PROJECT_ROOT / "rsc" / "robot" / "allegro_v5" / "allegro_right_A.urdf"
)
DEFAULT_ROBOT_URDF = PROJECT_ROOT / "rsc" / "robot" / "xarm_allegro_v5.urdf"
DEFAULT_ALLEGRO_POSE_ROOT = (
    Path(shared_dir) / "retargeter_alignment" / "allegro" / "allegro_alignment"
)
DEFAULT_OPEN_POSE = DEFAULT_ALLEGRO_POSE_ROOT / "000008"
DEFAULT_CLOSED_POSE = DEFAULT_ALLEGRO_POSE_ROOT / "000009"
ALLEGRO_JOINT_PARTS = ("base", "proximal", "medial", "distal")
ALLEGRO_THUMB_OPPOSITION_MAX_RAD = 1.496
ALLEGRO_THUMB_CLOSE_EXPONENT = 1.35
ALLEGRO_FINGERS = ("index", "middle", "ring", "thumb")
ALLEGRO_FINGER_SLICES = {
    finger: slice(index * 4, (index + 1) * 4)
    for index, finger in enumerate(ALLEGRO_FINGERS)
}
TACTILE_MAX_AGE_S = 0.25
TACTILE_CONTACT_DEBOUNCE_SAMPLES = 3


def _v5_hand_qpos(qpos: Mapping[str, float]) -> np.ndarray:
    """Convert semantic qpos to the V5 URDF's index/middle/ring/thumb order."""

    return np.asarray(
        [
            qpos[f"{finger}_{part}"]
            for finger in ("index", "middle", "ring", "thumb")
            for part in ALLEGRO_JOINT_PARTS
        ],
        dtype=np.float64,
    )


def _as_transform(value: Any, *, label: str) -> np.ndarray:
    transform = np.asarray(value, dtype=np.float64)
    if transform.shape == (3, 4):
        transform = np.vstack((transform, [0.0, 0.0, 0.0, 1.0]))
    if transform.shape != (4, 4):
        raise ValueError(f"{label} must be 3x4 or 4x4, got {transform.shape}")
    return transform


def _pose_world(response: Dict[str, Any]) -> np.ndarray:
    payload = response.get("object_6d", response)
    if not isinstance(payload, dict) or payload.get("pose_world") is None:
        raise RuntimeError(f"pose_world missing from RPC response: {response}")
    return _as_transform(payload["pose_world"], label="pose_world")


def _load_allegro_pose(sample_dir: Path) -> Dict[str, float]:
    """Load the semantic qpos used by the capture-time Allegro URDF."""

    target_path = sample_dir.expanduser() / "target_robot.json"
    target = json.loads(target_path.read_text(encoding="utf-8"))
    semantic_qpos = target.get("urdf_hand_qpos_rad")
    if not isinstance(semantic_qpos, dict):
        raise ValueError(f"urdf_hand_qpos_rad missing from {target_path}")

    qpos: Dict[str, float] = {}
    for finger in ("thumb", "index", "middle", "ring"):
        for part in ALLEGRO_JOINT_PARTS:
            semantic_name = f"{finger}_{part}"
            if semantic_name not in semantic_qpos:
                raise ValueError(f"{semantic_name} missing from {target_path}")
            value = float(semantic_qpos[semantic_name])
            if not np.isfinite(value):
                raise ValueError(f"non-finite {semantic_name} in {target_path}")
            qpos[semantic_name] = value
    # Keep the thumb fully opposed toward the palm for gripper-like closure.
    qpos["thumb_base"] = ALLEGRO_THUMB_OPPOSITION_MAX_RAD
    return qpos


def _interpolate_qpos(
    pose_a: Dict[str, float], pose_b: Dict[str, float], parameter: float
) -> Dict[str, float]:
    parameter = float(np.clip(parameter, 0.0, 1.0))
    if pose_a.keys() != pose_b.keys():
        raise ValueError("open and closed Allegro poses use different joints")
    thumb_parameter = parameter**ALLEGRO_THUMB_CLOSE_EXPONENT
    qpos: Dict[str, float] = {}
    for name in pose_a:
        joint_parameter = thumb_parameter if name.startswith("thumb_") else parameter
        qpos[name] = (
            (1.0 - joint_parameter) * pose_a[name]
            + joint_parameter * pose_b[name]
        )
    return qpos


def _pose_segment(
    start_pose: np.ndarray,
    target_pose: np.ndarray,
    seconds: float,
    rate_hz: float,
) -> np.ndarray:
    """Interpolate one smooth Cartesian wrist segment."""

    from scipy.spatial.transform import Rotation, Slerp

    start_pose = _as_transform(start_pose, label="segment start")
    target_pose = _as_transform(target_pose, label="segment target")
    frame_count = max(1, int(np.ceil(seconds * rate_hz)))
    linear_parameter = np.linspace(0.0, 1.0, frame_count + 1)[1:]
    parameter = linear_parameter * linear_parameter * (3.0 - 2.0 * linear_parameter)
    rotations = Slerp(
        [0.0, 1.0],
        Rotation.from_matrix(
            np.stack((start_pose[:3, :3], target_pose[:3, :3]))
        ),
    )(parameter).as_matrix()
    poses = np.tile(np.eye(4, dtype=np.float64), (frame_count, 1, 1))
    poses[:, :3, :3] = rotations
    poses[:, :3, 3] = (
        (1.0 - parameter[:, None]) * start_pose[:3, 3]
        + parameter[:, None] * target_pose[:3, 3]
    )
    poses[-1] = target_pose
    return poses


def _plan_wrist_motion(
    current_wrist_pose: np.ndarray,
    pregrasp_wrist_pose: np.ndarray,
    grasp_wrist_pose: np.ndarray,
    *,
    translation_speed_mps: float,
    rotation_speed_rps: float,
    descent_speed_mps: float,
    rate_hz: float,
) -> list[tuple[str, np.ndarray, float]]:
    """Lift, translate at a safe height, rotate in place, then descend."""

    from scipy.spatial.transform import Rotation

    lift_target = current_wrist_pose.copy()
    lift_target[2, 3] = max(current_wrist_pose[2, 3], pregrasp_wrist_pose[2, 3])
    translate_target = pregrasp_wrist_pose.copy()
    translate_target[:3, :3] = current_wrist_pose[:3, :3]
    translate_target[2, 3] = lift_target[2, 3]
    rotate_target = pregrasp_wrist_pose.copy()
    rotate_target[2, 3] = lift_target[2, 3]
    lift_distance = float(lift_target[2, 3] - current_wrist_pose[2, 3])
    translation_distance = float(
        np.linalg.norm(translate_target[:3, 3] - lift_target[:3, 3])
    )
    rotation_distance = float(
        Rotation.from_matrix(
            current_wrist_pose[:3, :3].T @ rotate_target[:3, :3]
        ).magnitude()
    )
    descent_distance = float(
        np.linalg.norm(grasp_wrist_pose[:3, 3] - rotate_target[:3, 3])
    )
    durations = (
        max(lift_distance / translation_speed_mps, 1.0),
        max(translation_distance / translation_speed_mps, 1.0),
        max(rotation_distance / rotation_speed_rps, 1.0),
        max(descent_distance / descent_speed_mps, 1.0),
    )
    return [
        (
            "lift",
            _pose_segment(current_wrist_pose, lift_target, durations[0], rate_hz),
            durations[0],
        ),
        (
            "translate",
            _pose_segment(lift_target, translate_target, durations[1], rate_hz),
            durations[1],
        ),
        (
            "rotate",
            _pose_segment(translate_target, rotate_target, durations[2], rate_hz),
            durations[2],
        ),
        (
            "descend",
            _pose_segment(rotate_target, grasp_wrist_pose, durations[3], rate_hz),
            durations[3],
        ),
    ]


def _link6_from_wrist(robot_urdf: Path) -> np.ndarray:
    """Read the fixed xArm link6-to-Allegro-wrist transform from the URDF."""

    from scipy.spatial.transform import Rotation

    root = ET.parse(robot_urdf).getroot()
    joint = root.find(".//joint[@name='arm_to_hand']")
    if joint is None:
        raise ValueError(f"arm_to_hand joint missing from {robot_urdf}")
    origin = joint.find("origin")
    if origin is None:
        raise ValueError(f"arm_to_hand origin missing from {robot_urdf}")
    xyz = np.fromstring(origin.attrib.get("xyz", "0 0 0"), sep=" ")
    rpy = np.fromstring(origin.attrib.get("rpy", "0 0 0"), sep=" ")
    if xyz.shape != (3,) or rpy.shape != (3,):
        raise ValueError(f"invalid arm_to_hand origin in {robot_urdf}")
    transform = np.eye(4, dtype=np.float64)
    transform[:3, 3] = xyz
    transform[:3, :3] = Rotation.from_euler("xyz", rpy).as_matrix()
    return transform


def _wrist_plan_to_link6(
    phases: list[tuple[str, np.ndarray, float]], link6_from_wrist: np.ndarray
) -> list[tuple[str, np.ndarray, float]]:
    wrist_from_link6 = np.linalg.inv(link6_from_wrist)
    return [
        (name, wrist_poses @ wrist_from_link6, seconds)
        for name, wrist_poses, seconds in phases
    ]


def _tactile_levels(tactile: Any) -> dict[str, float] | None:
    if tactile is None:
        return None
    values = np.asarray(tactile, dtype=np.float64).reshape(-1)
    if values.size < 4 or not np.all(np.isfinite(values)):
        return None
    blocks = np.array_split(np.abs(values), 4)
    return {
        finger: float(np.max(block))
        for finger, block in zip(ALLEGRO_FINGERS, blocks)
    }


def _fresh_tactile(feedback: Mapping[str, Any]) -> dict[str, float] | None:
    tactile_time = feedback.get("tactile_time")
    if (
        tactile_time is None
        or not np.isfinite(tactile_time)
        or time.perf_counter() - float(tactile_time) > TACTILE_MAX_AGE_S
    ):
        return None
    return _tactile_levels(feedback.get("tactile"))


def _hand_geometry(
    hand_urdf: Path,
    open_qpos: Dict[str, float],
    closed_qpos: Dict[str, float],
) -> tuple[float, np.ndarray, float]:
    """Return V5 rotation radius, open vertices, and grasp Y center."""

    from paradex.visualization.robot import RobotModule

    robot = RobotModule(str(hand_urdf))
    robot.update_cfg(_v5_hand_qpos(open_qpos))
    vertices = np.asarray(robot.scene.to_geometry().vertices)
    rotation_radius = float(np.linalg.norm(vertices, axis=1).max())

    robot.update_cfg(_v5_hand_qpos(closed_qpos))
    thumb = robot.get_transform("link_15_0_tip", "world", False)[:3, 3]
    fingers = np.mean(
        [
            robot.get_transform(link, "world", False)[:3, 3]
            for link in ("link_3_0_tip", "link_7_0_tip", "link_11_0_tip")
        ],
        axis=0,
    )
    return rotation_radius, vertices, float(0.5 * (thumb[1] + fingers[1]))


def _add_camera_frustums(server: Any, save_dir: Path, robot_from_world: np.ndarray) -> None:
    """Show only the captured camera frustums, expressed in robot coordinates."""

    cam_param_dir = save_dir / "cam_param"
    intrinsics = json.loads(
        (cam_param_dir / "intrinsics.json").read_text(encoding="utf-8")
    )
    extrinsics = json.loads(
        (cam_param_dir / "extrinsics.json").read_text(encoding="utf-8")
    )
    from scipy.spatial.transform import Rotation

    for camera_id, camera_from_world_raw in extrinsics.items():
        if camera_id not in intrinsics:
            continue
        camera_from_world = _as_transform(
            camera_from_world_raw, label=f"camera extrinsic {camera_id}"
        )
        robot_from_camera = robot_from_world @ np.linalg.inv(camera_from_world)
        camera = intrinsics[camera_id]
        intrinsic = np.asarray(camera["intrinsics_undistort"], dtype=float).reshape(3, 3)
        height = float(camera["height"])
        width = float(camera["width"])
        server.scene.add_camera_frustum(
            f"/cameras/{camera_id}",
            fov=2.0 * np.arctan2(height * 0.5, intrinsic[1, 1]),
            aspect=width / height,
            scale=0.035,
            line_width=1.5,
            color=(90, 150, 255),
            position=robot_from_camera[:3, 3],
            wxyz=Rotation.from_matrix(robot_from_camera[:3, :3]).as_quat()[
                [3, 0, 1, 2]
            ],
        )


def _load_mesh(path: Path):
    import trimesh

    loaded = trimesh.load(path, force="scene", process=False)
    meshes = []
    for node_name in loaded.graph.nodes_geometry:
        transform, geometry_name = loaded.graph[node_name]
        mesh = loaded.geometry[geometry_name].copy()
        mesh.apply_transform(transform)
        meshes.append(mesh)
    if not meshes:
        raise ValueError(f"no mesh geometry in {path}")
    return trimesh.util.concatenate(meshes)


def _principal_frame(vertices: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return mesh centroid, principal axes (columns), and axis extents."""

    vertices = np.asarray(vertices, dtype=np.float64)
    center = vertices.mean(axis=0)
    centered = vertices - center
    eigenvalues, axes = np.linalg.eigh(centered.T @ centered / len(centered))
    axes = axes[:, np.argsort(eigenvalues)[::-1]]

    # Eigenvector signs are arbitrary. Make each axis deterministic and retain
    # a right-handed frame so repeated previews do not flip by 180 degrees.
    for column in range(3):
        axis = axes[:, column]
        dominant = int(np.argmax(np.abs(axis)))
        if axis[dominant] < 0.0:
            axes[:, column] *= -1.0
    if np.linalg.det(axes) < 0.0:
        axes[:, 2] *= -1.0

    projected = centered @ axes
    extents = projected.max(axis=0) - projected.min(axis=0)
    return center, axes, extents


def _wrist_pose(
    object_pose_robot: np.ndarray,
    vertices_object: np.ndarray,
    center_object: np.ndarray,
    axes_object: np.ndarray,
    extents: np.ndarray,
    open_hand_vertices: np.ndarray,
    grasp_height_ratio: float,
    bottom_clearance_m: float,
    wrist_tilt_rad: float,
) -> tuple[np.ndarray, int, float, float, float]:
    from scipy.spatial.transform import Rotation

    axes_robot = object_pose_robot[:3, :3] @ axes_object
    horizontal_strength = np.linalg.norm(axes_robot[:2], axis=0)
    # Use the longest PCA axis that is not nearly vertical.
    horizontal_candidates = [
        int(index)
        for index in np.argsort(extents)[::-1]
        if horizontal_strength[index] > 0.25
    ]
    yaw_axis_index = (
        horizontal_candidates[0]
        if horizontal_candidates
        else int(np.argmax(horizontal_strength))
    )
    long_axis = axes_robot[:, yaw_axis_index].copy()
    long_axis[2] = 0.0
    if np.linalg.norm(long_axis) < 1e-8:
        long_axis = np.array([1.0, 0.0, 0.0])
    long_axis /= np.linalg.norm(long_axis)

    # Allegro convention for this gripper-like pose:
    #   local +X: palm approach (down)
    #   local +Y: row of fingers, placed along the object's long axis
    #   local +Z: open grasp aperture, placed across the object's narrow width
    approach = np.array([0.0, 0.0, -1.0])
    aperture_axis = np.cross(approach, long_axis)
    aperture_axis /= np.linalg.norm(aperture_axis)

    center_robot = (
        object_pose_robot[:3, :3] @ center_object + object_pose_robot[:3, 3]
    )
    robotward = np.array([-center_robot[0], -center_robot[1], 0.0])
    # The palm-to-wrist direction is local -Z. Flip both horizontal axes when
    # necessary so the wrist is always on the robot side of the object.
    if np.linalg.norm(robotward) > 1e-8 and np.dot(-aperture_axis, robotward) < 0.0:
        long_axis *= -1.0
        aperture_axis *= -1.0

    vertices_robot = (
        np.asarray(vertices_object) @ object_pose_robot[:3, :3].T
        + object_pose_robot[:3, 3]
    )
    object_bottom_z = float(vertices_robot[:, 2].min())
    object_top_z = float(vertices_robot[:, 2].max())
    object_height = object_top_z - object_bottom_z
    lowest_hand_z = max(
        object_bottom_z + bottom_clearance_m,
        object_bottom_z + grasp_height_ratio * object_height,
    )

    # V5 wrist and palm frames coincide. With local +X pointing down, the
    # visual mesh's largest local X is its lowest world-space point.
    wrist = np.eye(4, dtype=np.float64)
    wrist[:3, :3] = np.column_stack((approach, long_axis, aperture_axis))
    # Pitch around local Y so this is a front/back tilt, not lateral roll.
    # Positive local-Y is the requested forward direction for the mounted V5.
    wrist[:3, :3] = wrist[:3, :3] @ Rotation.from_rotvec(
        np.array([0.0, wrist_tilt_rad, 0.0])
    ).as_matrix()
    wrist[:2, 3] = center_robot[:2]
    local_vertical_offsets = np.asarray(open_hand_vertices) @ wrist[2, :3]
    wrist[2, 3] = lowest_hand_z - float(local_vertical_offsets.min())
    return wrist, yaw_axis_index, object_bottom_z, object_top_z, lowest_hand_z


def _live_wrist_pose(robot_urdf: Path) -> np.ndarray:
    """Read the current xArm pose without enabling Cartesian commands."""

    from paradex.retargetor.experiment.replay_object_relative import (
        _live_xarm_preview_state,
    )

    link6_pose, _joint_state = _live_xarm_preview_state()
    return link6_pose @ _link6_from_wrist(robot_urdf)


def _execute_robot_plan(
    args: argparse.Namespace,
    save_dir: Path,
    pregrasp_wrist_pose: np.ndarray,
    grasp_wrist_pose: np.ndarray,
    open_qpos: Dict[str, float],
    closed_qpos: Dict[str, float],
) -> dict[str, Any]:
    """Execute the displayed Cartesian plan and tactile-latched hand closure."""

    from paradex.dataset_acqusition.capture import CaptureSession
    from paradex.io.robot_controller import get_arm, get_hand
    from paradex.retargetor.allegro_alignment import urdf_qpos_to_retargeter_action

    arm = None
    hand = None
    capture = None
    try:
        execution_dir = save_dir / "execution"
        execution_path = execution_dir.resolve().relative_to(
            Path(shared_dir).resolve()
        ).as_posix()
        capture = CaptureSession(camera=True)
        capture.start(execution_path)
        print(f"[capture] recording robot execution: {execution_dir}")

        arm = get_arm("xarm", servo_api="cartesian_aa")
        hand = get_hand("allegro_v5", hand_side="right", tactile=True)
        if not hand.connection_event.wait(timeout=3.0):
            raise RuntimeError("timed out waiting for Allegro V5 feedback")
        feedback = hand.get_data()
        current_hand = np.asarray(feedback["qpos"], dtype=np.float64).reshape(-1)
        if current_hand.shape != (16,) or not np.all(np.isfinite(current_hand)):
            raise RuntimeError("invalid Allegro V5 joint feedback")

        open_action = np.asarray(
            urdf_qpos_to_retargeter_action(open_qpos), dtype=np.float64
        )
        hand.set_command_slew_rate(
            args.hand_speed_rad_s,
            initial_action=current_hand,
        )
        hand.move(open_action)
        open_timeout = max(
            2.0,
            float(np.max(np.abs(open_action - current_hand)))
            / args.hand_speed_rad_s
            + 1.0,
        )
        if not hand.wait_for_published_action(
            open_action, timeout_seconds=open_timeout, atol=1e-3
        ):
            raise RuntimeError("Allegro V5 did not reach the open command")

        tactile_deadline = time.monotonic() + 2.0
        while _fresh_tactile(hand.get_data()) is None:
            if time.monotonic() >= tactile_deadline:
                raise RuntimeError("fresh Allegro tactile feedback is unavailable")
            time.sleep(0.02)

        current_link6_pose = _as_transform(
            arm.get_data()["position"], label="live xArm link6 pose"
        )
        link6_from_wrist = _link6_from_wrist(args.robot_urdf)
        current_wrist_pose = current_link6_pose @ link6_from_wrist
        wrist_phases = _plan_wrist_motion(
            current_wrist_pose,
            pregrasp_wrist_pose,
            grasp_wrist_pose,
            translation_speed_mps=args.translation_speed_mps,
            rotation_speed_rps=args.rotation_speed_rps,
            descent_speed_mps=args.descent_speed_mps,
            rate_hz=args.motion_rate_hz,
        )
        link6_phases = _wrist_plan_to_link6(wrist_phases, link6_from_wrist)
        for phase_name, poses, seconds in link6_phases:
            print(f"[execute] {phase_name}: one Cartesian target / {seconds:.2f}s")
            arm.move_cartesian_timed(poses[-1], seconds=seconds)
            if arm.is_error():
                raise RuntimeError(f"xArm controller error after {phase_name}")

        thresholds = {
            finger: (
                args.ring_tactile_threshold
                if finger == "ring"
                else args.tactile_threshold
            )
            for finger in ALLEGRO_FINGERS
        }
        contact_counts = {finger: 0 for finger in ALLEGRO_FINGERS}
        latched = {finger: False for finger in ALLEGRO_FINGERS}
        held_actions: dict[str, np.ndarray] = {}
        start = time.monotonic()
        while True:
            parameter = min((time.monotonic() - start) / args.grasp_seconds, 1.0)
            desired_qpos = _interpolate_qpos(open_qpos, closed_qpos, parameter)
            desired_action = np.asarray(
                urdf_qpos_to_retargeter_action(desired_qpos), dtype=np.float64
            )
            feedback = hand.get_data()
            levels = _fresh_tactile(feedback)
            feedback_action = np.asarray(feedback["qpos"], dtype=np.float64)
            if levels is None or feedback_action.shape != (16,):
                if feedback_action.shape == (16,):
                    hand.move(feedback_action)
                raise RuntimeError("tactile feedback became stale during grasp")
            for finger in ALLEGRO_FINGERS:
                if not latched[finger]:
                    contact_counts[finger] = (
                        contact_counts[finger] + 1
                        if levels[finger] >= thresholds[finger]
                        else 0
                    )
                    if contact_counts[finger] >= TACTILE_CONTACT_DEBOUNCE_SAMPLES:
                        latched[finger] = True
                        held_actions[finger] = feedback_action[
                            ALLEGRO_FINGER_SLICES[finger]
                        ].copy()
                        print(
                            f"[grasp] {finger} latched at tactile={levels[finger]:.0f}"
                        )
                if latched[finger]:
                    desired_action[ALLEGRO_FINGER_SLICES[finger]] = held_actions[finger]
            hand.move(desired_action)
            if parameter >= 1.0 or all(latched.values()):
                break
            time.sleep(1.0 / args.grasp_fps)

        lift_pose = grasp_wrist_pose.copy()
        lift_pose[2, 3] += args.postgrasp_lift_m
        lift_seconds = max(
            args.postgrasp_lift_m / args.postgrasp_lift_speed_mps,
            1.0,
        )
        lift_link6 = lift_pose @ np.linalg.inv(link6_from_wrist)
        print(f"[execute] postgrasp_lift: one Cartesian target / {lift_seconds:.2f}s")
        arm.move_cartesian_timed(lift_link6, seconds=lift_seconds)
        if arm.is_error():
            raise RuntimeError("xArm controller error after postgrasp_lift")
        return {
            "latched_fingers": latched,
            "tactile_thresholds": thresholds,
            "postgrasp_lift_m": args.postgrasp_lift_m,
            "execution_capture_dir": str(execution_dir),
        }
    finally:
        if arm is not None:
            arm.end()
        if hand is not None:
            hand.end()
        if capture is not None:
            try:
                if getattr(capture, "save_path", None) is not None:
                    capture.stop()
            finally:
                capture.end()


def _show_preview(
    args: argparse.Namespace,
    save_dir: Path,
    response: Dict[str, Any],
    mesh_path: Path,
) -> None:
    import viser
    from scipy.spatial.transform import Rotation

    from paradex.visualization.visualizer.viser import ViserRobotModule

    mesh = _load_mesh(mesh_path)
    center_object, axes_object, extents = _principal_frame(mesh.vertices)
    open_qpos = _load_allegro_pose(args.open_pose_sample)
    closed_qpos = _load_allegro_pose(args.closed_pose_sample)
    pose_world = _pose_world(response)
    c2r = _as_transform(load_current_C2R(), label="C2R")
    pose_robot = np.linalg.inv(c2r) @ pose_world
    rotation_radius, open_hand_vertices, lateral_offset = _hand_geometry(
        args.hand_urdf, open_qpos, closed_qpos
    )
    (
        grasp_wrist_pose,
        yaw_axis_index,
        object_bottom_z,
        object_top_z,
        lowest_hand_z,
    ) = _wrist_pose(
        pose_robot,
        mesh.vertices,
        center_object,
        axes_object,
        extents,
        open_hand_vertices,
        args.grasp_height_ratio,
        args.bottom_clearance_m,
        np.deg2rad(args.wrist_tilt_deg),
    )
    grasp_wrist_pose[:2, 3] -= grasp_wrist_pose[:2, 1] * lateral_offset
    pregrasp_wrist_pose = grasp_wrist_pose.copy()
    pregrasp_wrist_pose[2, 3] = max(
        grasp_wrist_pose[2, 3] + args.approach_distance_m,
        object_top_z + rotation_radius + args.rotation_clearance_m,
    )
    current_wrist_pose = _live_wrist_pose(args.robot_urdf)
    wrist_phases = _plan_wrist_motion(
        current_wrist_pose,
        pregrasp_wrist_pose,
        grasp_wrist_pose,
        translation_speed_mps=args.translation_speed_mps,
        rotation_speed_rps=args.rotation_speed_rps,
        descent_speed_mps=args.descent_speed_mps,
        rate_hz=args.motion_rate_hz,
    )
    postgrasp_lift_pose = grasp_wrist_pose.copy()
    postgrasp_lift_pose[2, 3] += args.postgrasp_lift_m
    postgrasp_lift_seconds = max(
        args.postgrasp_lift_m / args.postgrasp_lift_speed_mps,
        1.0,
    )
    postgrasp_lift_poses = _pose_segment(
        grasp_wrist_pose,
        postgrasp_lift_pose,
        postgrasp_lift_seconds,
        args.motion_rate_hz,
    )
    link6_from_wrist = _link6_from_wrist(args.robot_urdf)
    link6_phases = _wrist_plan_to_link6(wrist_phases, link6_from_wrist)
    wrist_trajectory = np.concatenate([poses for _name, poses, _seconds in wrist_phases])
    link6_trajectory = np.concatenate([poses for _name, poses, _seconds in link6_phases])
    phase_ends = np.cumsum([len(poses) for _name, poses, _seconds in wrist_phases])
    np.savez_compressed(
        save_dir / "grasp_plan.npz",
        wrist_poses=wrist_trajectory,
        link6_poses=link6_trajectory,
        postgrasp_lift_wrist_poses=postgrasp_lift_poses,
        postgrasp_lift_link6_poses=(
            postgrasp_lift_poses @ np.linalg.inv(link6_from_wrist)
        ),
        phase_names=np.asarray([name for name, _poses, _seconds in wrist_phases]),
        phase_ends=phase_ends,
    )

    np.save(save_dir / "C2R.npy", c2r)
    preview = {
        "pose_robot": pose_robot.tolist(),
        "principal_center_object": center_object.tolist(),
        "principal_axes_object_columns": axes_object.tolist(),
        "principal_extents_m": extents.tolist(),
        "wrist_pose_robot": grasp_wrist_pose.tolist(),
        "pregrasp_wrist_pose_robot": pregrasp_wrist_pose.tolist(),
        "grasp_wrist_pose_robot": grasp_wrist_pose.tolist(),
        "current_wrist_pose_robot": current_wrist_pose.tolist(),
        "palm_center_xy_robot": grasp_wrist_pose[:2, 3].tolist(),
        "thumb_opposition_lateral_offset_m": lateral_offset,
        "object_bottom_z_robot": object_bottom_z,
        "object_top_z_robot": object_top_z,
        "planned_lowest_open_hand_z_robot": lowest_hand_z,
        "open_hand_downward_reach_m": float(
            grasp_wrist_pose[2, 3] - lowest_hand_z
        ),
        "hand_rotation_radius_m": rotation_radius,
        "wrist_pitch_deg_forward": args.wrist_tilt_deg,
        "rotation_clearance_m": args.rotation_clearance_m,
        "bottom_clearance_m": args.bottom_clearance_m,
        "grasp_height_ratio": args.grasp_height_ratio,
        "wrist_yaw_axis_index": yaw_axis_index,
        "long_axis_robot": grasp_wrist_pose[:3, 1].tolist(),
        "closing_axis_robot": grasp_wrist_pose[:3, 2].tolist(),
        "approach_axis_robot": grasp_wrist_pose[:3, 0].tolist(),
        "wristward_axis_robot": (-grasp_wrist_pose[:3, 2]).tolist(),
        "open_pose_sample": str(args.open_pose_sample.resolve()),
        "closed_pose_sample": str(args.closed_pose_sample.resolve()),
        "open_hand_qpos": open_qpos,
        "closed_hand_qpos": closed_qpos,
        "approach_distance_m": args.approach_distance_m,
        "grasp_seconds": args.grasp_seconds,
        "postgrasp_lift_m": args.postgrasp_lift_m,
        "postgrasp_lift_seconds": postgrasp_lift_seconds,
        "motion_phases": [
            {"name": name, "frames": len(poses), "seconds": seconds}
            for name, poses, seconds in wrist_phases
        ],
    }
    (save_dir / "grasp_preview.json").write_text(
        json.dumps(preview, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    server = viser.ViserServer()
    server.gui.configure_theme(dark_mode=True)
    server.scene.world_axes.visible = False
    server.scene.enable_default_lights(True)
    _add_camera_frustums(server, save_dir, np.linalg.inv(c2r))

    mesh_robot = mesh.copy()
    mesh_robot.apply_transform(pose_robot)
    server.scene.add_mesh_trimesh("/object", mesh_robot)

    center_robot = pose_robot[:3, :3] @ center_object + pose_robot[:3, 3]
    axes_robot = pose_robot[:3, :3] @ axes_object
    colors = ((255, 60, 60), (60, 255, 60), (60, 120, 255))
    for index, color in enumerate(colors):
        half_length = max(float(extents[index]) * 0.6, 0.04)
        server.scene.add_spline_catmull_rom(
            f"/principal_axes/{index}",
            positions=np.stack(
                (
                    center_robot - axes_robot[:, index] * half_length,
                    center_robot + axes_robot[:, index] * half_length,
                )
            ),
            color=color,
            line_width=4.0,
        )

    hand_root = server.scene.add_frame(
        "/allegro",
        show_axes=False,
        position=current_wrist_pose[:3, 3],
        wxyz=Rotation.from_matrix(current_wrist_pose[:3, :3]).as_quat()[
            [3, 0, 1, 2]
        ],
    )
    hand = ViserRobotModule(
        target=server,
        urdf_path=str(args.hand_urdf),
        root_node_name="/allegro/hand",
    )
    hand.update_cfg(_v5_hand_qpos(open_qpos))
    replay_requested = Event()
    replay_requested.set()
    replay = server.gui.add_button("Replay robot plan")

    @replay.on_click
    def _(_event) -> None:
        replay_requested.set()

    print(f"[saved] preview: {save_dir / 'grasp_preview.json'}")
    print(
        "[viser] lift, translate high without wrist rotation, rotate, descend, "
        f"close over {args.grasp_seconds:.1f}s, then lift {args.postgrasp_lift_m:.2f}m; "
        "use 'Replay robot plan'"
    )
    for phase_name, poses, seconds in wrist_phases:
        print(f"[plan] {phase_name}: {len(poses)} frames / {seconds:.2f}s")
    print("[execute] close Viser first; the terminal will then ask for Y")
    print("[viser] Ctrl+C to close and return to the terminal")
    try:
        while True:
            if not replay_requested.wait(timeout=0.1):
                continue
            replay_requested.clear()
            hand.update_cfg(_v5_hand_qpos(open_qpos))
            hand_root.position = current_wrist_pose[:3, 3]
            hand_root.wxyz = Rotation.from_matrix(
                current_wrist_pose[:3, :3]
            ).as_quat()[[3, 0, 1, 2]]
            for phase_name, poses, _seconds in wrist_phases:
                for pose in poses:
                    hand_root.position = pose[:3, 3]
                    hand_root.wxyz = Rotation.from_matrix(pose[:3, :3]).as_quat()[
                        [3, 0, 1, 2]
                    ]
                    time.sleep(1.0 / args.motion_rate_hz)

            start = time.monotonic()
            while True:
                parameter = min((time.monotonic() - start) / args.grasp_seconds, 1.0)
                hand.update_cfg(
                    _v5_hand_qpos(
                        _interpolate_qpos(open_qpos, closed_qpos, parameter)
                    )
                )
                if parameter >= 1.0:
                    break
                time.sleep(1.0 / args.grasp_fps)
            for pose in postgrasp_lift_poses:
                hand_root.position = pose[:3, 3]
                hand_root.wxyz = Rotation.from_matrix(pose[:3, :3]).as_quat()[
                    [3, 0, 1, 2]
                ]
                time.sleep(1.0 / args.motion_rate_hz)
    except KeyboardInterrupt:
        pass
    finally:
        server.stop()

    try:
        confirmation = input(
            "[execute] Send the grasp plan to the real xArm + Allegro? [Y/N]: "
        ).strip()
    except EOFError:
        confirmation = ""
    if confirmation.upper() != "Y":
        print("[execute] cancelled; no robot command was sent")
        return
    result = _execute_robot_plan(
        args,
        save_dir,
        pregrasp_wrist_pose,
        grasp_wrist_pose,
        open_qpos,
        closed_qpos,
    )
    result_path = save_dir / "grasp_execution.json"
    result_path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"[saved] execution: {result_path}")


def _shared_data_path(path: Path) -> str:
    shared_root = Path(shared_dir).resolve()
    try:
        relative = path.resolve().relative_to(shared_root)
    except ValueError as exc:
        raise ValueError(f"capture path must be under {shared_root}: {path}") from exc
    return f"shared_data/{relative.as_posix()}"


def _capture(save_dir: Path, rcc_entry: str) -> None:
    save_dir.mkdir(parents=True, exist_ok=True)
    save_current_camparam(str(save_dir))

    controller = remote_camera_controller(rcc_entry)
    try:
        controller.start("image", False, f"{_shared_data_path(save_dir)}/raw")
        controller.stop()
    finally:
        controller.end()


def _request_pose(
    rpc_addr: str,
    image_path: str,
    mesh_name: str,
    timeout_ms: int,
) -> Dict[str, Any]:
    import zmq

    context = zmq.Context()
    socket = context.socket(zmq.REQ)
    socket.setsockopt(zmq.RCVTIMEO, timeout_ms)
    socket.setsockopt(zmq.SNDTIMEO, timeout_ms)
    socket.setsockopt(zmq.LINGER, 0)
    try:
        socket.connect(rpc_addr)
        socket.send_json(
            {
                "command": "infer",
                "image_path": image_path,
                "mesh_name": mesh_name,
            }
        )
        response = socket.recv_json()
    finally:
        socket.close()
        context.term()

    if not isinstance(response, dict):
        raise RuntimeError(f"invalid RPC response: {response!r}")
    return response


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mesh-name",
        "--mesh_name",
        "--object",
        dest="mesh_name",
        required=True,
        help="mesh/object name understood by the 6D pose server",
    )
    parser.add_argument(
        "--save-path",
        default="capture/eccv2026/grasp",
        help="capture root relative to shared_data",
    )
    parser.add_argument("--rcc-entry", default="image_main.py")
    parser.add_argument("--rpc-addr", default="tcp://192.168.0.14:5570")
    parser.add_argument("--rpc-timeout-ms", type=int, default=300000)
    parser.add_argument("--mesh-root-dir", type=Path, default=DEFAULT_MESH_ROOT)
    parser.add_argument("--hand-urdf", type=Path, default=DEFAULT_HAND_URDF)
    parser.add_argument("--robot-urdf", type=Path, default=DEFAULT_ROBOT_URDF)
    parser.add_argument(
        "--open-pose-sample",
        type=Path,
        default=DEFAULT_OPEN_POSE,
        help="saved Allegro pose directory used as the open endpoint",
    )
    parser.add_argument(
        "--closed-pose-sample",
        type=Path,
        default=DEFAULT_CLOSED_POSE,
        help="saved Allegro pose directory used as the closed endpoint",
    )
    parser.add_argument(
        "--approach-distance-m",
        "--approach-height-m",
        dest="approach_distance_m",
        type=float,
        default=0.08,
        help="minimum vertical approach distance before grasping",
    )
    parser.add_argument(
        "--rotation-clearance-m",
        type=float,
        default=0.05,
        help="clearance between the object top and the hand's rotation envelope",
    )
    parser.add_argument(
        "--grasp-height-ratio",
        type=float,
        default=0.0,
        help="target height within the mesh, where 0 is bottom and 1 is top",
    )
    parser.add_argument(
        "--bottom-clearance-m",
        type=float,
        default=0.005,
        help="minimum open-hand clearance above the estimated object bottom",
    )
    parser.add_argument(
        "--wrist-tilt-deg",
        type=float,
        default=20.0,
        help="forward wrist pitch toward the fingertips and away from the robot",
    )
    parser.add_argument(
        "--postgrasp-lift-m",
        type=float,
        default=0.07,
        help="vertical robot-Z lift after grasping",
    )
    parser.add_argument(
        "--postgrasp-lift-speed-mps",
        type=float,
        default=0.015,
        help="vertical lift speed after grasping",
    )
    parser.add_argument(
        "--grasp-seconds",
        type=float,
        default=5.0,
        help="seconds for the open-to-closed Allegro motion",
    )
    parser.add_argument(
        "--translation-speed-mps",
        type=float,
        default=0.03,
        help="wrist translation speed before rotation (default: 0.03 m/s)",
    )
    parser.add_argument(
        "--rotation-speed-rps",
        type=float,
        default=np.deg2rad(15.0),
        help="in-place wrist rotation speed (default: 15 deg/s)",
    )
    parser.add_argument(
        "--descent-speed-mps",
        type=float,
        default=0.015,
        help="final vertical descent speed (default: 0.015 m/s)",
    )
    parser.add_argument("--motion-rate-hz", type=float, default=30.0)
    parser.add_argument("--grasp-fps", type=float, default=30.0)
    parser.add_argument("--hand-speed-rad-s", type=float, default=0.25)
    parser.add_argument("--tactile-threshold", type=float, default=200.0)
    parser.add_argument("--ring-tactile-threshold", type=float, default=150.0)
    parser.add_argument("--no-vis", action="store_true")
    args = parser.parse_args()
    if args.approach_distance_m <= 0.0:
        raise ValueError("--approach-distance-m must be positive")
    if args.rotation_clearance_m < 0.0 or not np.isfinite(args.rotation_clearance_m):
        raise ValueError("--rotation-clearance-m must be non-negative and finite")
    if not 0.0 <= args.grasp_height_ratio <= 1.0:
        raise ValueError("--grasp-height-ratio must be in [0, 1]")
    if args.bottom_clearance_m < 0.0 or not np.isfinite(args.bottom_clearance_m):
        raise ValueError("--bottom-clearance-m must be non-negative and finite")
    if not 0.0 <= args.wrist_tilt_deg <= 30.0:
        raise ValueError("--wrist-tilt-deg must be in [0, 30]")
    if args.postgrasp_lift_m <= 0.0 or not np.isfinite(args.postgrasp_lift_m):
        raise ValueError("--postgrasp-lift-m must be positive and finite")
    positive_options = {
        "--grasp-seconds": args.grasp_seconds,
        "--translation-speed-mps": args.translation_speed_mps,
        "--rotation-speed-rps": args.rotation_speed_rps,
        "--descent-speed-mps": args.descent_speed_mps,
        "--motion-rate-hz": args.motion_rate_hz,
        "--grasp-fps": args.grasp_fps,
        "--hand-speed-rad-s": args.hand_speed_rad_s,
        "--postgrasp-lift-speed-mps": args.postgrasp_lift_speed_mps,
    }
    for option, value in positive_options.items():
        if value <= 0.0 or not np.isfinite(value):
            raise ValueError(f"{option} must be a positive finite value")
    for option, value in (
        ("--tactile-threshold", args.tactile_threshold),
        ("--ring-tactile-threshold", args.ring_tactile_threshold),
    ):
        if value < 0.0 or not np.isfinite(value):
            raise ValueError(f"{option} must be a non-negative finite value")
    mesh_path = args.mesh_root_dir.expanduser() / args.mesh_name / f"{args.mesh_name}.obj"
    if not args.no_vis and not mesh_path.is_file():
        raise FileNotFoundError(f"object mesh not found: {mesh_path}")
    if not args.no_vis and not args.hand_urdf.expanduser().is_file():
        raise FileNotFoundError(f"Allegro URDF not found: {args.hand_urdf}")
    if not args.no_vis and not args.robot_urdf.expanduser().is_file():
        raise FileNotFoundError(f"xArm+Allegro URDF not found: {args.robot_urdf}")
    for label, sample in (
        ("open", args.open_pose_sample),
        ("closed", args.closed_pose_sample),
    ):
        if not args.no_vis and not (sample.expanduser() / "target_robot.json").is_file():
            raise FileNotFoundError(f"Allegro {label} pose sample not found: {sample}")

    timestamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir = Path(shared_dir) / args.save_path / args.mesh_name / timestamp

    print(f"[capture] {save_dir}")
    _capture(save_dir, args.rcc_entry)

    image_path = _shared_data_path(save_dir)
    print(f"[rpc] {args.rpc_addr}: {args.mesh_name}")
    response = _request_pose(
        args.rpc_addr,
        image_path,
        args.mesh_name,
        args.rpc_timeout_ms,
    )

    response_path = save_dir / "object_6d.json"
    response_path.write_text(
        json.dumps(response, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"[saved] images: {save_dir / 'raw' / 'images'}")
    print(f"[saved] response: {response_path}")
    if not args.no_vis:
        _show_preview(args, save_dir, response, mesh_path)


if __name__ == "__main__":
    main()
