#!/usr/bin/env python3
"""Run interactive multiview Object6D capture and optional robot replay."""

from __future__ import annotations

import argparse
import datetime
import json
import os
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Dict, Optional

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

import cv2
import numpy as np
import trimesh
import zmq

from paradex.calibration.utils import save_current_C2R, save_current_camparam
from paradex.io.camera_system.remote_camera_controller import remote_camera_controller
from paradex.utils.path import shared_dir
from paradex.visualization.visualizer.viser import ViserViewer
from paradex.retargetor.experiment import replay_object_relative as replay_core
from paradex.retargetor.experiment.replay_pose_retrieval import (
    DEFAULT_CANDIDATE_EPISODES,
    DEFAULT_EPISODE_ROOT,
    load_candidate_episodes,
    parse_episode_ids,
    replay_closest_episode,
)


DEFAULT_RETRIEVAL_TRANSLATION_SCALE_M = 0.05
DEFAULT_RETRIEVAL_ROTATION_SCALE_RAD = 0.25


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Capture images, run remote Object6D, and show the pose in Viser."
    )
    parser.add_argument("--name", "--mesh_name", dest="mesh_name", required=True)
    parser.add_argument("--save_path", default="object_6d/sil")
    parser.add_argument("--rpc_addr", default="tcp://192.168.0.3:5570")
    parser.add_argument("--rpc_timeout_ms", type=int, default=300_000)
    parser.add_argument("--rcc_entry", default="image_main.py")
    parser.add_argument(
        "--mesh_root_dir",
        default=os.path.join(str(shared_dir), "mesh_blender"),
    )
    parser.add_argument("--no_vis", action="store_true")
    parser.add_argument(
        "--replay",
        action="store_true",
        help="retrieve the closest banana episode and prepare annotated dish replay",
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help="allow replay execution after confirming with y",
    )
    parser.add_argument(
        "--auto-execute",
        action="store_true",
        help=(
            "execute each planned replay without confirmation; implies --execute "
            "and is intended for use with --no-replay-preview"
        ),
    )
    preview_group = parser.add_mutually_exclusive_group()
    preview_group.add_argument(
        "--replay-preview",
        dest="replay_preview",
        action="store_true",
        help="show the replay preview before execution (default)",
    )
    preview_group.add_argument(
        "--no-replay-preview",
        dest="replay_preview",
        action="store_false",
        help="skip replay preview generation and visualization",
    )
    parser.set_defaults(replay_preview=None)
    parser.add_argument("--episode-root", type=Path, default=DEFAULT_EPISODE_ROOT)
    parser.add_argument(
        "--candidate-episodes",
        type=parse_episode_ids,
        default=DEFAULT_CANDIDATE_EPISODES,
        help="comma-separated candidate episode IDs",
    )
    parser.add_argument(
        "--retrieval-translation-scale-m",
        type=float,
        default=DEFAULT_RETRIEVAL_TRANSLATION_SCALE_M,
    )
    parser.add_argument(
        "--retrieval-rotation-scale-rad",
        type=float,
        default=DEFAULT_RETRIEVAL_ROTATION_SCALE_RAD,
        help=(
            "rotation error represented by one retrieval-score unit; the 0.25 rad "
            "default makes rotation twice as influential as the previous 0.5 rad default"
        ),
    )
    parser.add_argument("--rate-scale", type=float, default=1.0)
    parser.add_argument("--approach-linear-speed-mps", type=float, default=0.13)
    parser.add_argument("--approach-angular-speed-rps", type=float, default=0.8)
    parser.add_argument("--approach-min-seconds", type=float, default=1.0)
    parser.add_argument("--approach-rate-hz", type=float, default=50.0)
    parser.add_argument("--dish-transfer-linear-speed-mps", type=float, default=0.2)
    parser.add_argument(
        "--dish-transfer-max-distance-m",
        type=float,
        default=1.0,
        help="refuse a banana-derived dish correction longer than this distance",
    )
    parser.add_argument("--dish-transfer-min-seconds", type=float, default=1.0)
    parser.add_argument("--dish-transfer-rate-hz", type=float, default=50.0)
    parser.add_argument(
        "--dish-clearance-m",
        type=float,
        default=0.02,
        help="robot +z offset above the detected dish center for banana release",
    )
    parser.add_argument("--settle-seconds", type=float, default=1.0)
    parser.add_argument(
        "--robot-urdf", type=Path, default=replay_core.DEFAULT_ROBOT_URDF
    )
    parser.add_argument("--preview-max-frames", type=int, default=150)
    parser.add_argument("--preview-ik-max-nfev", type=int, default=50)
    parser.add_argument("--preview-position-scale", type=float, default=0.05)
    parser.add_argument("--preview-rotation-scale", type=float, default=0.5)
    parser.add_argument("--no-viser-object-align", action="store_true")
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if args.auto_execute:
        args.execute = True
    if args.execute and not args.replay:
        raise ValueError("--execute requires --replay")
    if args.replay_preview and not args.replay:
        raise ValueError("--replay-preview requires --replay")
    if args.replay and args.replay_preview is None:
        args.replay_preview = True
    positive = {
        "--retrieval-translation-scale-m": args.retrieval_translation_scale_m,
        "--retrieval-rotation-scale-rad": args.retrieval_rotation_scale_rad,
        "--rate-scale": args.rate_scale,
        "--approach-linear-speed-mps": args.approach_linear_speed_mps,
        "--approach-angular-speed-rps": args.approach_angular_speed_rps,
        "--approach-rate-hz": args.approach_rate_hz,
        "--dish-transfer-linear-speed-mps": args.dish_transfer_linear_speed_mps,
        "--dish-transfer-max-distance-m": args.dish_transfer_max_distance_m,
        "--dish-transfer-rate-hz": args.dish_transfer_rate_hz,
        "--preview-max-frames": args.preview_max_frames,
        "--preview-ik-max-nfev": args.preview_ik_max_nfev,
        "--preview-position-scale": args.preview_position_scale,
        "--preview-rotation-scale": args.preview_rotation_scale,
    }
    invalid = [name for name, value in positive.items() if value <= 0]
    if invalid:
        raise ValueError(f"positive value required for: {', '.join(invalid)}")
    if (
        args.approach_min_seconds < 0
        or args.dish_transfer_min_seconds < 0
        or args.dish_clearance_m < 0
        or args.settle_seconds < 0
    ):
        raise ValueError(
            "approach/dish-transfer/settle durations and dish clearance must be "
            "non-negative"
        )


def send_rpc_once(address: str, request: dict, timeout_ms: int) -> dict:
    context = zmq.Context()
    socket = context.socket(zmq.REQ)
    socket.setsockopt(zmq.RCVTIMEO, timeout_ms)
    socket.setsockopt(zmq.SNDTIMEO, timeout_ms)
    socket.setsockopt(zmq.LINGER, 0)
    try:
        socket.connect(address)
        started = time.perf_counter()
        socket.send_json(request)
        response = socket.recv_json()
        round_trip_seconds = time.perf_counter() - started
        print(
            "Object6D RPC request round-trip (send -> receive): "
            f"{round_trip_seconds:.3f} s"
        )
        if response.get("status") == "error":
            raise RuntimeError(response.get("error", "Object6D RPC failed"))
        response["_client_rpc_round_trip_seconds"] = round_trip_seconds
        return response
    finally:
        socket.close()
        context.term()


def to_shared_data_path(path: Path) -> str:
    path = path.resolve()
    root = Path(shared_dir).resolve()
    try:
        relative = path.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"Capture path must be under {root}: {path}") from exc
    return (Path("shared_data") / relative).as_posix()


def to_4x4(value: Any) -> np.ndarray:
    matrix = np.asarray(value, dtype=np.float32)
    if matrix.shape == (4, 4):
        return matrix
    if matrix.shape == (3, 4):
        result = np.eye(4, dtype=np.float32)
        result[:3] = matrix
        return result
    if matrix.size == 12:
        result = np.eye(4, dtype=np.float32)
        result[:3] = matrix.reshape(3, 4)
        return result
    raise ValueError(f"Expected a 3x4 or 4x4 transform, got {matrix.shape}")


def extract_pose(response: Dict[str, Any]) -> Dict[str, Any]:
    payload = response.get("object_6d", response)
    pose_world = payload.get("pose_world")
    if pose_world is None:
        raise RuntimeError(f"pose_world missing from Object6D response: {response}")
    return {
        "pose_world": pose_world,
        "initial_pose_world": payload.get("initial_pose_world"),
        "pose_left_cam": payload.get("pose_left_cam"),
        "R_world": payload.get("R_world"),
        "t_world": payload.get("t_world"),
        "refined_iou": payload.get("refined_iou"),
        "timing_seconds": payload.get("timing_seconds"),
        "sam3_mask_count": payload.get("sam3_mask_count"),
        "projection_grid_paths": payload.get("projection_grid_paths"),
        "wooden_object": payload.get("wooden_object"),
        "client_rpc_round_trip_seconds": response.get("_client_rpc_round_trip_seconds"),
    }


def get_cutting_board_point(result: Dict[str, Any]) -> Optional[np.ndarray]:
    triangulation = result.get("wooden_object")
    if not isinstance(triangulation, dict):
        return None
    if triangulation.get("status") != "success":
        return None
    point = triangulation.get("point_world")
    if point is None:
        return None
    point_world = np.asarray(point, dtype=np.float32).reshape(-1)
    if point_world.shape != (3,) or not np.isfinite(point_world).all():
        raise ValueError(f"Invalid cutting-board triangulation point: {point}")
    return point_world


def load_mesh(mesh_root: Path, mesh_name: str) -> trimesh.Trimesh:
    object_dir = mesh_root / mesh_name
    candidates = [
        object_dir / f"{mesh_name}.obj",
        object_dir / f"{mesh_name}_viser.obj",
        object_dir / f"{mesh_name}_remeshed.obj",
    ]
    mesh_path = next((path for path in candidates if path.exists()), None)
    if mesh_path is None:
        raise FileNotFoundError(
            f"No visualization mesh found for '{mesh_name}' under {object_dir}"
        )
    geometry = trimesh.load(mesh_path, force="mesh", process=False)
    if isinstance(geometry, trimesh.Trimesh):
        return geometry
    if isinstance(geometry, trimesh.Scene):
        meshes = [
            item
            for item in geometry.geometry.values()
            if isinstance(item, trimesh.Trimesh)
        ]
        if meshes:
            return trimesh.util.concatenate(meshes)
    raise ValueError(f"Unsupported mesh content: {mesh_path}")


def camera_intrinsic(payload: Dict[str, Any]) -> Dict[str, Any]:
    K = payload.get("intrinsics_undistort")
    if K is None:
        K = payload.get("original_intrinsics", payload.get("intrinsics"))
    if K is None:
        raise ValueError("Camera calibration is missing intrinsics")
    K_array = np.asarray(K, dtype=np.float32)
    return {
        "intrinsics_undistort": K_array.tolist(),
        "width": int(payload.get("width") or round(float(K_array[0, 2]) * 2)),
        "height": int(payload.get("height") or round(float(K_array[1, 2]) * 2)),
    }


def find_camera_image(capture_root: Path, serial: str) -> Optional[Path]:
    image_dir = capture_root / "raw" / "images"
    for suffix in (".png", ".jpg", ".jpeg", ".PNG", ".JPG", ".JPEG"):
        path = image_dir / f"{serial}{suffix}"
        if path.exists():
            return path
    return None


def add_cameras(viewer: ViserViewer, capture_root: Path, c2r: np.ndarray) -> None:
    cam_param = capture_root / "cam_param"
    with (cam_param / "intrinsics.json").open("r", encoding="utf-8") as file:
        intrinsics = json.load(file)
    with (cam_param / "extrinsics.json").open("r", encoding="utf-8") as file:
        extrinsics = json.load(file)

    for serial in sorted(set(intrinsics) & set(extrinsics)):
        cam_from_world = to_4x4(extrinsics[serial])
        # C2R maps robot-frame coordinates into the calibration/world frame.
        # Therefore cam_from_robot = cam_from_world @ C2R.
        cam_from_robot = cam_from_world @ c2r
        robot_from_cam = np.linalg.inv(cam_from_robot)
        image_path = find_camera_image(capture_root, serial)
        image_rgb = None
        if image_path is not None:
            image_bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
            if image_bgr is not None:
                image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        viewer.add_camera(
            name=serial,
            extrinsic=robot_from_cam,
            intrinsic=camera_intrinsic(intrinsics[serial]),
            color=(80, 80, 80),
            size=0.08,
            show_axes=True,
            image=image_rgb,
        )


def capture_once(capture_root: Path, remote_path: str, rcc_entry: str) -> None:
    capture_root.mkdir(parents=True, exist_ok=True)
    save_current_camparam(str(capture_root))
    save_current_C2R(str(capture_root))
    controller = remote_camera_controller(rcc_entry)
    try:
        print(f"Capturing multiview images to {capture_root}")
        controller.start("image", False, remote_path)
        controller.stop()
    finally:
        controller.end()


def run_round(
    args: argparse.Namespace,
    *,
    replay_arm: Any | None = None,
    replay_hand: Any | None = None,
    replay_episodes: list[replay_core.Episode] | None = None,
) -> None:
    """Capture, infer, plan, and optionally execute one replay round."""

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    relative_capture = Path(args.save_path) / timestamp
    capture_root = Path(shared_dir) / relative_capture
    remote_path = (Path("shared_data") / relative_capture / "raw").as_posix()

    capture_once(capture_root, remote_path, args.rcc_entry)

    request = {
        "command": "infer",
        "image_path": to_shared_data_path(capture_root),
        "mesh_name": args.mesh_name,
        "save_projection_grids": True,
        "wooden_object_triangulation": True,
        "wooden_object_prompt": "wooden object",
    }
    response = send_rpc_once(args.rpc_addr, request, args.rpc_timeout_ms)
    pose = extract_pose(response)

    c2r_path = capture_root / "C2R.npy"
    c2r = to_4x4(np.load(c2r_path))
    robot_from_world = np.linalg.inv(c2r)
    pose_world = to_4x4(pose["pose_world"])
    pose_robot = robot_from_world @ pose_world
    pose["C2R"] = c2r.astype(float).tolist()
    pose["pose_robot"] = pose_robot.astype(float).tolist()

    cutting_board_point_world = get_cutting_board_point(pose)
    cutting_board_point_robot = None
    if cutting_board_point_world is not None:
        point_world_h = np.append(cutting_board_point_world, 1.0)
        cutting_board_point_robot = (robot_from_world @ point_world_h)[:3]
        pose["cutting_board_point_robot"] = cutting_board_point_robot.astype(
            float
        ).tolist()
        triangulation_payload = pose.get("wooden_object")
        if isinstance(triangulation_payload, dict):
            triangulation_payload["point_robot"] = cutting_board_point_robot.astype(
                float
            ).tolist()

    result_path = capture_root / "object_6d.json"
    result_path.write_text(
        json.dumps(pose, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"Saved Object6D result: {result_path}")

    triangulation = pose.get("wooden_object")
    if isinstance(triangulation, dict):
        triangulation_path = capture_root / "cutting_board_triangulation.json"
        triangulation_path.write_text(
            json.dumps(triangulation, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        print(f"Saved cutting-board triangulation: {triangulation_path}")

    if cutting_board_point_world is None:
        status = (
            triangulation.get("status", "missing")
            if isinstance(triangulation, dict)
            else "missing"
        )
        print(f"Cutting-board triangulation unavailable: status={status}")
    else:
        print(f"Cutting-board point_world: {cutting_board_point_world.tolist()}")
        print(f"Cutting-board point_robot: {cutting_board_point_robot.tolist()}")

    if args.replay:
        if args.mesh_name != "banana":
            raise ValueError(
                "the configured retrieval dataset currently supports --name banana only"
            )
        args.object = args.mesh_name
        args.replay_output_dir = capture_root / "replay"
        args.mesh_root_dir = Path(args.mesh_root_dir).expanduser()
        replay_closest_episode(
            args,
            current_object_robot=pose_robot,
            current_c2r=c2r,
            dish_point_robot=cutting_board_point_robot,
            arm=replay_arm,
            hand=replay_hand,
            episodes=replay_episodes,
        )
        return

    if args.no_vis:
        return

    mesh = load_mesh(Path(args.mesh_root_dir).expanduser(), args.mesh_name)
    # ViserViewer.update() expects player GUI fields to exist in this version.
    viewer = ViserViewer(
        up_direction=np.array([0.0, 0.0, 1.0]),
        scene_title="Object6D capture (robot frame)",
    )
    viewer.add_floor(height=0.0)
    viewer.add_frame("robot_origin", np.eye(4, dtype=np.float32))
    add_cameras(viewer, capture_root, c2r)
    viewer.add_object(args.mesh_name, mesh, pose_robot, opacity=1.0)
    if cutting_board_point_robot is not None:
        point_frame = np.eye(4, dtype=np.float32)
        point_frame[:3, 3] = cutting_board_point_robot
        viewer.add_frame("cutting_board_triangulation", point_frame)
        viewer.add_sphere(
            "cutting_board_triangulation",
            cutting_board_point_robot,
            radius=0.02,
            color=(1.0, 0.1, 0.0),
        )
        viewer.server.scene.add_label(
            "/cutting_board_triangulation/frame/label",
            "cutting board triangulation",
        )

    print("Viser is running. Press Ctrl+C to exit.")
    try:
        while True:
            viewer.update()
            time.sleep(0.01)
    except KeyboardInterrupt:
        pass


def _preload_replay_episodes(args: argparse.Namespace) -> list[replay_core.Episode]:
    started = time.perf_counter()
    print(
        f"[preload] loading {len(args.candidate_episodes)} candidate episode(s) "
        "from shared storage."
    )
    episodes = load_candidate_episodes(args.episode_root, args.candidate_episodes)
    print(
        f"[preload] {len(episodes)} candidate episode(s) ready in "
        f"{time.perf_counter() - started:.3f}s."
    )
    return episodes


def run_replay_idle_loop(args: argparse.Namespace) -> None:
    """Wait for capture commands and return to idle after every replay round."""

    replay_arm = None
    replay_hand = None
    replay_episodes = _preload_replay_episodes(args)
    try:
        if args.execute:
            from paradex.io.robot_controller import get_arm, get_hand

            print("[idle] connecting persistent xArm and Allegro controllers.")
            replay_arm = get_arm("xarm", servo_api="cartesian_aa")
            try:
                replay_hand = get_hand("allegro_v5", hand_side="right")
            except Exception:
                replay_arm.end()
                replay_arm = None
                raise

        print("[idle] Ready. Enter c to capture and replay, or q to quit.")
        while True:
            try:
                key = input("[idle] c=capture, q=quit > ").strip().lower()
            except EOFError:
                print("\n[idle] stdin closed; exiting.")
                return
            except KeyboardInterrupt:
                print("\n[idle] interrupted; exiting.")
                return

            if key == "q":
                print("[idle] exiting.")
                return
            if key != "c":
                if key:
                    print(f"[idle] unknown command {key!r}; enter c or q.")
                continue

            print("[round] starting capture -> Object6D -> replay planning.")
            try:
                run_round(
                    args,
                    replay_arm=replay_arm,
                    replay_hand=replay_hand,
                    replay_episodes=replay_episodes,
                )
            except Exception as exc:
                print(f"[round] failed: {exc}", file=sys.stderr)
                traceback.print_exc()
                print("[round] returning to idle after failure.")
            else:
                print("[round] complete; returning to idle.")
    finally:
        if replay_hand is not None:
            replay_hand.end()
        if replay_arm is not None:
            replay_arm.end()


def main() -> None:
    args = parse_args()
    validate_args(args)
    if args.replay:
        run_replay_idle_loop(args)
    else:
        run_round(args)


if __name__ == "__main__":
    main()
