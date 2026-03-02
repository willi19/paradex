import argparse
import os
import re
import shutil
import subprocess

import cv2
import numpy as np

from paradex.calibration.utils import load_camparam
from paradex.utils.path import shared_dir
from paradex.visualization.robot import RobotModule
from paradex.robot.inspire import inspire_action_to_qpos_dof12, inspire_f1_action_to_qpos_dof6
from paradex.utils.load_data import load_series, resample_to
from paradex.image.grid import make_image_grid
from paradex.image.undistort import precomute_undistort_map, apply_undistort_map
from paradex.object.utils import load_object_mesh, apply_transform
from paradex.robot.utils import get_robot_urdf_path
from paradex.image.projection import BatchRenderer


def _build_video_source_map(videos_dir, cam_ids):
    avi_paths = sorted(
        path for path in (os.path.join(videos_dir, name) for name in os.listdir(videos_dir))
        if os.path.isfile(path) and path.lower().endswith(".avi")
    )
    if not avi_paths:
        raise ValueError(f"No .avi files found in {videos_dir}")

    cam_to_video = {}
    for cam_id in cam_ids:
        exact = None
        partial = None
        for path in avi_paths:
            stem = os.path.splitext(os.path.basename(path))[0]
            if stem == cam_id:
                exact = path
                break
            if cam_id in stem and partial is None:
                partial = path
        chosen = exact if exact is not None else partial
        if chosen is None:
            raise ValueError(
                f"No matching .avi for cam '{cam_id}' in {videos_dir}. "
                f"Available: {[os.path.basename(p) for p in avi_paths]}"
            )
        cam_to_video[cam_id] = chosen
    return cam_to_video


def _resolve_video_frame_index(frame_id, frame_offset):
    # Frame IDs are usually 1-based in capture metadata; VideoCapture expects 0-based.
    return int(frame_id) + int(frame_offset) - 1


def load_object_world_trajectory_npz(npz_path: str) -> np.ndarray:
    if not os.path.exists(npz_path):
        raise FileNotFoundError(f"Object trajectory file not found: {npz_path}")

    payload = np.load(npz_path, allow_pickle=True)
    arr = None
    if isinstance(payload, np.lib.npyio.NpzFile):
        # Case 1: npz contains per-frame 4x4 arrays with keys like frame_35.npy
        if payload.files:
            per_frame = []
            for key in payload.files:
                v = np.asarray(payload[key])
                if v.shape == (4, 4):
                    m = re.search(r"frame_(\d+)", key)
                    frame_idx = int(m.group(1)) if m else None
                    per_frame.append((frame_idx, key, v))
                else:
                    per_frame = []
                    break

            if per_frame:
                if all(idx is not None for idx, _, _ in per_frame):
                    per_frame.sort(key=lambda x: x[0])
                else:
                    per_frame.sort(key=lambda x: x[1])
                arr = np.stack([v for _, _, v in per_frame], axis=0)

        preferred_keys = (
            "obj_T_frames",
            "T",
            "poses",
            "trajectory",
            "traj",
            "arr_0",
        )
        if arr is None:
            for key in preferred_keys:
                if key in payload.files:
                    arr = payload[key]
                    break
        if arr is None:
            if not payload.files:
                raise ValueError(f"No arrays in {npz_path}")
            arr = payload[payload.files[0]]
    else:
        arr = payload

    arr = np.asarray(arr)
    if arr.ndim == 2 and arr.shape == (4, 4):
        arr = arr[None, ...]
    elif arr.ndim == 2 and arr.shape[1] == 16:
        arr = arr.reshape(arr.shape[0], 4, 4)
    elif arr.ndim == 3 and arr.shape[1:] == (3, 4):
        padded = np.tile(np.eye(4, dtype=float), (arr.shape[0], 1, 1))
        padded[:, :3, :] = arr
        arr = padded

    if arr.ndim != 3 or arr.shape[1:] != (4, 4):
        raise ValueError(
            f"Unsupported object trajectory shape from {npz_path}: {arr.shape}. "
            "Expected (N,4,4) world poses."
        )

    return arr.astype(float)


def project_robot_and_object(
    arm,
    hand,
    object,
    capture_root,
    capture_ep,
    object_mesh_name,
    project_robot,
    project_object,
    start_frame,
    end_frame,
    stride,
    output_dir,
    overlay_option,
    output_type,
    grid_scale,
    device,
    frame_offset,
    arm_time_offset,
    hand_time_offset,
):
    ep_root = os.path.join(shared_dir, "capture", capture_root, object, str(capture_ep))
    raw_root = os.path.join(ep_root, "raw")
    arm_dir = os.path.join(raw_root, "arm")
    hand_dir = os.path.join(raw_root, "hand")

    intrinsic, extrinsic_from_camparam = load_camparam(ep_root)
    # c2r is robot->world in this pipeline; invert to get world->robot.
    c2r = np.load(os.path.join(ep_root, "C2R.npy"))
    robot_from_world = np.linalg.inv(c2r)

    qpos_video = None
    video_times = None
    video_frame_ids = None

    robot = None
    robot_dof = None
    link_label_map = None
    finger_prefix_map = None
    finger_colors = None

    obj_traj_raw = None
    obj_mesh = None
    obj_base_vertices = None

    if project_robot:
        arm_qpos, arm_time = load_series(arm_dir, ("position.npy", "action_qpos.npy", "action.npy"))

        if overlay_option == "action":
            hand_action, hand_time = load_series(hand_dir, ("action.npy", "right_joint_states.npy"))
        else:
            hand_action, hand_time = load_series(hand_dir, ("position.npy", "right_joint_states.npy"))

        hand_time_path = os.path.join(hand_dir, "time.npy")
        if not os.path.exists(hand_time_path):
            if len(arm_time) > 1:
                hand_time = np.linspace(arm_time[0], arm_time[-1], hand_action.shape[0], dtype=float)
            else:
                hand_time = np.arange(hand_action.shape[0], dtype=float)

        if arm_time_offset != 0.0:
            arm_time = arm_time + arm_time_offset
        if hand_time_offset != 0.0:
            hand_time = hand_time + hand_time_offset

        hand_action = resample_to(hand_time, hand_action, arm_time)

        if hand == "inspire":
            hand_qpos = inspire_action_to_qpos_dof12(hand_action)
        elif hand == "inspire_f1":
            hand_qpos = inspire_f1_action_to_qpos_dof6(hand_action)
        else:
            hand_qpos = hand_action

        full_qpos = np.concatenate([arm_qpos, hand_qpos], axis=1)

        ts_dir = os.path.join(raw_root, "timestamps")
        ts_path = os.path.join(ts_dir, "timestamp.npy")
        frame_id_path = os.path.join(ts_dir, "frame_id.npy")
        if os.path.exists(ts_path) and os.path.exists(frame_id_path):
            video_times = np.load(ts_path)
            video_frame_ids = np.load(frame_id_path)
            qpos_video = resample_to(arm_time, full_qpos, video_times)
        else:
            video_times = arm_time
            video_frame_ids = np.arange(1, full_qpos.shape[0] + 1, dtype=int)
            qpos_video = full_qpos

        urdf_path = get_robot_urdf_path(arm, hand)
        robot = RobotModule(urdf_path)
        robot_dof = robot.get_num_joints()
        robot.update_cfg(full_qpos[0, :robot_dof])
        robot.get_robot_mesh()
        finger_prefix_map = {
            "right_thumb_": "thumb",
            "right_index_": "index",
            "right_middle_": "middle",
            "right_ring_": "ring",
            "right_little_": "pinky",
            "thumb_tip": "thumb",
            "index_tip": "index",
            "middle_tip": "middle",
            "ring_tip": "ring",
            "little_tip": "pinky",
            "thumb_force_sensor": "thumb",
            "index_force_sensor": "index",
            "middle_force_sensor": "middle",
            "ring_force_sensor": "ring",
            "little_force_sensor": "pinky",
            "left_thumb_": "thumb",
            "left_index_": "index",
            "left_middle_": "middle",
            "left_ring_": "ring",
            "left_little_": "pinky",
        }
        finger_colors = {
            "thumb": (255, 140, 0),
            "index": (0, 200, 255),
            "middle": (0, 255, 100),
            "ring": (255, 0, 200),
            "pinky": (255, 220, 0),
        }
    else:
        ts_dir = os.path.join(raw_root, "timestamps")
        ts_path = os.path.join(ts_dir, "timestamp.npy")
        frame_id_path = os.path.join(ts_dir, "frame_id.npy")
        if os.path.exists(ts_path) and os.path.exists(frame_id_path):
            video_times = np.load(ts_path)
            video_frame_ids = np.load(frame_id_path)

    if project_object:
        obj_traj_path = os.path.join(ep_root, "object_tracking_result", "obj_T_frames.npz")
        obj_traj_raw = load_object_world_trajectory_npz(obj_traj_path)
        print(f"Loaded object trajectory with {obj_traj_raw.shape[0]} frames from {obj_traj_path}")

        object_mesh_path = os.path.join(shared_dir, "mesh", object_mesh_name, f"{object_mesh_name}.obj")
        obj_mesh = load_object_mesh(object_mesh_path)
        obj_base_vertices = np.asarray(obj_mesh.vertices, dtype=np.float32)

    if output_dir is None:
        output_dir = os.path.join(ep_root, f"overlay_{overlay_option}")
    os.makedirs(output_dir, exist_ok=True)

    videos_dir = os.path.join(ep_root, "videos")
    if not os.path.isdir(videos_dir):
        raise ValueError(f"videos directory does not exist ({videos_dir}).")
    cam_to_video = _build_video_source_map(videos_dir, intrinsic.keys())
    video_caps = {}
    video_num_frames = {}
    for cam_id, video_path in cam_to_video.items():
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise RuntimeError(f"Failed to open video for cam '{cam_id}': {video_path}")
        video_caps[cam_id] = cap
        count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        video_num_frames[cam_id] = max(0, count)

    if qpos_video is not None:
        total_frames = qpos_video.shape[0]
        print("aaaaa")
    elif video_frame_ids is not None:
        total_frames = len(video_frame_ids)
        print("bbbbb")
    else:
        print("ccccc")
        if not video_num_frames:
            raise ValueError("Cannot infer frames: no videos available.")
        valid_counts = [n for n in video_num_frames.values() if n > 0]
        if not valid_counts:
            raise ValueError("Cannot infer frames: all videos report 0 frames.")
        total_frames = min(valid_counts)
        video_frame_ids = np.arange(1, total_frames + 1, dtype=int)
        video_times = np.arange(total_frames, dtype=float)

    if project_object:
        if video_times is None:
            video_times = np.arange(total_frames, dtype=float)
        if len(video_times) != total_frames:
            raise ValueError(
                f"Timeline mismatch: len(video_times)={len(video_times)} vs total_frames={total_frames}"
            )

        if obj_traj_raw.shape[0] == total_frames:
            obj_traj_video = obj_traj_raw
        else:
            obj_time = np.linspace(video_times[0], video_times[-1], obj_traj_raw.shape[0], dtype=float)
            obj_traj_video = resample_to(
                obj_time, obj_traj_raw.reshape(obj_traj_raw.shape[0], -1), video_times
            ).reshape(total_frames, 4, 4)
        obj_traj_robot = np.einsum("ij,tjk->tik", robot_from_world, obj_traj_video)

    start = max(0, start_frame)
    end = total_frames if end_frame is None else min(end_frame, total_frames)
    
    if start >= end:
        raise ValueError(f"Invalid frame range: start={start}, end={end}, total={total_frames}")
    frame_indices = list(range(start, end, max(1, stride)))

    cam_info = {}
    render_extrinsics = {}
    undistort_maps = {}
    for cam_id, intr in intrinsic.items():
        height = intr["height"]
        width = intr["width"]

        cam_from_world = np.eye(4)
        cam_from_world[:3, :] = extrinsic_from_camparam[cam_id]
        cam_from_robot = cam_from_world @ c2r

        render_extrinsics[cam_id] = cam_from_robot[:3, :]
        cam_info[cam_id] = {"extr": cam_from_robot[:3, :], "width": width, "height": height}
        _, mapx, mapy = precomute_undistort_map(intr)
        undistort_maps[cam_id] = (mapx, mapy)

    renderer = BatchRenderer(intrinsic, render_extrinsics)

    writers_overlay = {}
    grid_dir = None
    if output_type == "video":
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        for cam_id, info in cam_info.items():
            h, w = info["height"], info["width"]
            writers_overlay[cam_id] = cv2.VideoWriter(
                os.path.join(output_dir, f"{cam_id}_overlay.mp4"), fourcc, 30, (w, h)
            )
    else:
        grid_dir = os.path.join(output_dir, "grid")
        os.makedirs(grid_dir, exist_ok=True)

    for fidx in frame_indices:
        print(f"Processing frame {fidx} / {total_frames}...")

        render_meshes = []
        mesh_colors = []
        mesh_alphas = []

        if project_robot:
            robot.update_cfg(qpos_video[fidx, :robot_dof])
            scene = robot.scene
            robot_color = (40, 200, 40)
            robot_alpha = 0.35
            finger_alpha = 0.55

            if link_label_map is None:
                link_label_map = {}
                for link_name in scene.geometry.keys():
                    label = None
                    for prefix, name in finger_prefix_map.items():
                        if link_name.startswith(prefix):
                            label = name
                            break
                    link_label_map[link_name] = label

            for link_name, mesh in scene.geometry.items():
                transform = scene.graph.get(link_name)[0]
                link_mesh = mesh.copy()
                link_mesh.apply_transform(transform)
                render_meshes.append(link_mesh)

                label = link_label_map.get(link_name)
                if label is None:
                    mesh_colors.append(robot_color)
                    mesh_alphas.append(robot_alpha)
                else:
                    mesh_colors.append(finger_colors[label])
                    mesh_alphas.append(finger_alpha)

        if project_object:
            obj_pose = obj_traj_robot[fidx]
            obj_verts_np = apply_transform(obj_base_vertices, obj_pose)
            obj_mesh_frame = obj_mesh.copy()
            obj_mesh_frame.vertices = obj_verts_np
            render_meshes.append(obj_mesh_frame)
            mesh_colors.append((255, 80, 80))
            mesh_alphas.append(0.45)

        id_dict = None
        if render_meshes:
            _, _, _, id_dict = renderer.render_multi(render_meshes)

        overlays_for_grid = []
        for cam_id in sorted(cam_info.keys()):
            info = cam_info[cam_id]
            width = info["width"]
            height = info["height"]

            frame_num = int(video_frame_ids[fidx])
            frame_pos = _resolve_video_frame_index(frame_num, frame_offset)
            image_bgr = None
            if frame_pos >= 0:
                cap = video_caps[cam_id]
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_pos)
                ok, frame = cap.read()
                if ok and frame is not None:
                    image_bgr = frame

            if image_bgr is None:
                image = np.zeros((height, width, 3), dtype=np.uint8)
            else:
                if image_bgr.shape[0] != height or image_bgr.shape[1] != width:
                    image_bgr = cv2.resize(image_bgr, (width, height))
                mapx, mapy = undistort_maps[cam_id]
                image_bgr = apply_undistort_map(image_bgr, mapx, mapy)
                image = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

            overlay = image
            if id_dict is not None:
                id_map = id_dict[cam_id][..., 0]
                ids = np.rint(id_map).astype(np.int32)
                if len(render_meshes) > 0:
                    ids = np.clip(ids, 0, len(render_meshes))
                    color_lut = np.zeros((len(render_meshes) + 1, 3), dtype=np.float32)
                    alpha_lut = np.zeros((len(render_meshes) + 1,), dtype=np.float32)
                    color_lut[1:] = np.asarray(mesh_colors, dtype=np.float32)
                    alpha_lut[1:] = np.asarray(mesh_alphas, dtype=np.float32)
                    colors = color_lut[ids]
                    alphas = alpha_lut[ids][..., None]
                    overlay = image.astype(np.float32) * (1.0 - alphas) + colors * alphas
                    overlay = overlay.astype(np.uint8)

            if output_type == "video":
                writers_overlay[cam_id].write(cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
            else:
                overlays_for_grid.append(overlay)

        if output_type == "grid" and overlays_for_grid:
            grid_img = make_image_grid(overlays_for_grid)
            if grid_scale != 1.0:
                if grid_scale <= 0.0:
                    raise ValueError(f"grid_scale must be > 0, got {grid_scale}")
                new_w = max(1, int(round(grid_img.shape[1] * grid_scale)))
                new_h = max(1, int(round(grid_img.shape[0] * grid_scale)))
                grid_img = cv2.resize(grid_img, (new_w, new_h), interpolation=cv2.INTER_AREA)
            frame_name = int(video_frame_ids[fidx])
            cv2.imwrite(
                os.path.join(grid_dir, f"frame_{frame_name:05d}.png"),
                cv2.cvtColor(grid_img, cv2.COLOR_RGB2BGR),
            )

    for w in writers_overlay.values():
        w.release()
    for cap in video_caps.values():
        cap.release()

    if output_type == "grid" and grid_dir is not None:
        if not frame_indices:
            raise ValueError("No frames rendered for grid output; cannot build video.")
        start_number = int(video_frame_ids[frame_indices[0]])
        ffmpeg_path = shutil.which("ffmpeg")
        if ffmpeg_path is None:
            raise RuntimeError("ffmpeg not found in PATH; cannot build grid video.")
        input_pattern = os.path.join(grid_dir, "frame_%05d.png")
        output_path = os.path.join(grid_dir, f"grid_overlay.mp4")
        cmd = [
            ffmpeg_path,
            "-y",
            "-framerate",
            "30",
            "-start_number",
            str(start_number),
            "-i",
            input_pattern,
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-vf",
            "scale=1920:-2",
            output_path,
        ]
        subprocess.run(cmd, check=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--arm", type=str, default="xarm")
    parser.add_argument("--hand", type=str, default="inspire_f1")
    parser.add_argument("--object", type=str, required=True)
    parser.add_argument("--capture-ep", type=str, default="0")
    parser.add_argument("--object-mesh-name", type=str)
    parser.add_argument("--capture-root", type=str, default="eccv2026/inspire_f1", help="Capture root directory name.")
    parser.add_argument("--project-object", action="store_true", help="Project the tracked object mesh in addition to the robot.")
    parser.add_argument("--project-robot", action="store_true", help="Project the robot mesh.")
    parser.add_argument("--start-frame", type=int, default=0, help="Start frame index (inclusive).")
    parser.add_argument("--end-frame", type=int, default=None, help="End frame index (exclusive). Defaults to full length.")
    parser.add_argument("--stride", type=int, default=1, help="Frame stride.")
    parser.add_argument("--output-dir", type=str, default=None, help="Output directory for projected masks/overlays.")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--overlay-option", type=str, choices=["action", "position"], default="position", help="Whether to overlay using hand action or hand position data.")
    parser.add_argument("--output-type", type=str, choices=["video", "grid"], default="grid", help="Output overlaid result as videos (per camera) or as tiled grid images per frame.")
    parser.add_argument("--grid-scale", type=float, default=0.25, help="Downscale factor for saved grid frames before writing PNG (e.g., 0.5 halves width/height).")
    parser.add_argument("--frame-offset", type=int, default=0, help="Shift overlay target video frames by this offset.")
    parser.add_argument("--arm-time-offset", type=float, default=0.09, help="Shift arm timestamps by this many seconds (positive delays arm).")
    parser.add_argument("--hand-time-offset", type=float, default=0.0, help="Shift hand timestamps by this many seconds (positive delays hand).")
    args = parser.parse_args()

    if args.object_mesh_name == None:
        object_mesh_name = args.object

    project_robot_and_object(
        arm=args.arm,
        hand=args.hand,
        object=args.object,
        capture_root=args.capture_root,
        capture_ep=args.capture_ep,
        object_mesh_name=object_mesh_name,
        project_robot=args.project_robot,
        project_object=args.project_object,
        start_frame=args.start_frame,
        end_frame=args.end_frame,
        stride=args.stride,
        output_dir=args.output_dir,
        overlay_option=args.overlay_option,
        output_type=args.output_type,
        grid_scale=args.grid_scale,
        device=args.device,
        frame_offset=args.frame_offset,
        arm_time_offset=args.arm_time_offset,
        hand_time_offset=args.hand_time_offset,
    )


if __name__ == "__main__":
    main()
