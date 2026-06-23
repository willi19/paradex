import argparse
import os
import shutil
import subprocess

import cv2
import numpy as np

from paradex.calibration.utils import load_camparam
from paradex.utils.path import shared_dir
from paradex.utils.load_data import load_series, resample_to
from paradex.image.grid import make_image_grid
from paradex.image.undistort import precomute_undistort_map, apply_undistort_map
from paradex.image.projection import BatchRenderer
from paradex.visualization.robot import RobotModule
from paradex.robot.utils import get_robot_urdf_path


# Allegro v5 URDF link naming: link_<i>_0
# Finger assignment matches the joint blocks in xarm_allegro_v5.urdf:
#   index : link_0_0 .. link_3_0  (+ tip)
#   middle: link_4_0 .. link_7_0  (+ tip)
#   ring  : link_8_0 .. link_11_0 (+ tip)
#   thumb : link_12_0 .. link_15_0 (+ tip)
ALLEGRO_FINGER_RANGES = {
    "index": range(0, 4),
    "middle": range(4, 8),
    "ring": range(8, 12),
    "thumb": range(12, 16),
}
FINGER_COLORS = {
    "thumb": (255, 140, 0),
    "index": (0, 200, 255),
    "middle": (0, 255, 100),
    "ring": (255, 0, 200),
}
ROBOT_COLOR = (40, 200, 40)
ROBOT_ALPHA = 0.35
FINGER_ALPHA = 0.55


def _link_finger_label(link_name):
    # Strip a trailing "_tip" so tip meshes inherit the finger color.
    base = link_name[:-4] if link_name.endswith("_tip") else link_name
    if not base.startswith("link_") or not base.endswith("_0"):
        return None
    try:
        idx = int(base[len("link_"):-len("_0")])
    except ValueError:
        return None
    for finger, rng in ALLEGRO_FINGER_RANGES.items():
        if idx in rng:
            return finger
    return None


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


def project_allegro_v5(
    object_name,
    capture_ep,
    capture_root,
    overlay_option,
    start_frame,
    end_frame,
    stride,
    frames,
    output_dir,
    output_type,
    grid_scale,
    frame_offset,
    arm_time_offset,
    hand_time_offset,
):
    ep_root = os.path.join(shared_dir, "capture", capture_root, object_name, str(capture_ep))
    raw_root = os.path.join(ep_root, "raw")
    arm_dir = os.path.join(raw_root, "arm")
    hand_dir = os.path.join(raw_root, "hand")

    intrinsic, extrinsic_from_camparam = load_camparam(ep_root)
    c2r = np.load(os.path.join(ep_root, "C2R.npy"))

    arm_qpos, arm_time = load_series(arm_dir, ("position.npy", "action_qpos.npy", "action.npy"))
    if overlay_option == "action":
        hand_action, hand_time = load_series(hand_dir, ("action.npy", "position.npy"))
    else:
        hand_action, hand_time = load_series(hand_dir, ("position.npy", "action.npy"))

    if arm_time_offset != 0.0:
        arm_time = arm_time + arm_time_offset
    if hand_time_offset != 0.0:
        hand_time = hand_time + hand_time_offset

    hand_qpos = resample_to(hand_time, hand_action, arm_time)
    if hand_qpos.shape[1] != 16:
        raise ValueError(f"Expected allegro_v5 hand action with 16 dof, got {hand_qpos.shape}")

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

    urdf_path = get_robot_urdf_path("xarm", "allegro_v5")
    robot = RobotModule(urdf_path)
    robot_dof = robot.get_num_joints()
    robot.update_cfg(full_qpos[0, :robot_dof])
    robot.get_robot_mesh()

    if output_dir is None:
        suffix = "_frames" if frames else ""
        output_dir = os.path.join(ep_root, f"overlay_allegro_v5_{overlay_option}{suffix}")
    os.makedirs(output_dir, exist_ok=True)

    videos_dir = os.path.join(ep_root, "videos")
    if not os.path.isdir(videos_dir):
        raise ValueError(f"videos directory does not exist ({videos_dir}).")
    cam_to_video = _build_video_source_map(videos_dir, intrinsic.keys())
    video_caps = {cam_id: cv2.VideoCapture(path) for cam_id, path in cam_to_video.items()}
    for cam_id, cap in video_caps.items():
        if not cap.isOpened():
            raise RuntimeError(f"Failed to open video for cam '{cam_id}': {cam_to_video[cam_id]}")

    total_frames = qpos_video.shape[0]
    if frames:
        frame_indices = [int(f) for f in frames if 0 <= int(f) < total_frames]
        if not frame_indices:
            raise ValueError(f"No valid frames in {frames}; total_frames={total_frames}")
    else:
        start = max(0, start_frame)
        end = total_frames if end_frame is None else min(end_frame, total_frames)
        if start >= end:
            raise ValueError(f"Invalid frame range: start={start}, end={end}, total={total_frames}")
        frame_indices = list(range(start, end, max(1, stride)))

    cam_info = {}
    render_extrinsics = {}
    undistort_maps = {}
    for cam_id, intr in intrinsic.items():
        cam_from_world = np.eye(4)
        cam_from_world[:3, :] = extrinsic_from_camparam[cam_id]
        cam_from_robot = cam_from_world @ c2r
        render_extrinsics[cam_id] = cam_from_robot[:3, :]
        cam_info[cam_id] = {
            "extr": cam_from_robot[:3, :],
            "width": intr["width"],
            "height": intr["height"],
        }
        _, mapx, mapy = precomute_undistort_map(intr)
        undistort_maps[cam_id] = (mapx, mapy)

    renderer = BatchRenderer(intrinsic, render_extrinsics)

    writers_overlay = {}
    grid_dir = None
    if output_type == "video":
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        for cam_id, info in cam_info.items():
            writers_overlay[cam_id] = cv2.VideoWriter(
                os.path.join(output_dir, f"{cam_id}_overlay.mp4"),
                fourcc, 30, (info["width"], info["height"]),
            )
    else:
        grid_dir = os.path.join(output_dir, "grid")
        os.makedirs(grid_dir, exist_ok=True)

    link_label_map = None
    for fidx in frame_indices:
        print(f"Processing frame {fidx} / {total_frames}...")
        robot.update_cfg(qpos_video[fidx, :robot_dof])
        scene = robot.scene

        if link_label_map is None:
            link_label_map = {name: _link_finger_label(name) for name in scene.geometry.keys()}

        render_meshes = []
        mesh_colors = []
        mesh_alphas = []
        for link_name, mesh in scene.geometry.items():
            transform = scene.graph.get(link_name)[0]
            link_mesh = mesh.copy()
            link_mesh.apply_transform(transform)
            render_meshes.append(link_mesh)
            label = link_label_map.get(link_name)
            if label is None:
                mesh_colors.append(ROBOT_COLOR)
                mesh_alphas.append(ROBOT_ALPHA)
            else:
                mesh_colors.append(FINGER_COLORS[label])
                mesh_alphas.append(FINGER_ALPHA)

        _, _, _, id_dict = renderer.render_multi(render_meshes)

        overlays_for_grid = []
        for cam_id in sorted(cam_info.keys()):
            info = cam_info[cam_id]
            width, height = info["width"], info["height"]

            frame_num = int(video_frame_ids[fidx])
            frame_pos = frame_num + frame_offset - 1
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

            id_map = id_dict[cam_id][..., 0]
            ids = np.clip(np.rint(id_map).astype(np.int32), 0, len(render_meshes))
            color_lut = np.zeros((len(render_meshes) + 1, 3), dtype=np.float32)
            alpha_lut = np.zeros((len(render_meshes) + 1,), dtype=np.float32)
            color_lut[1:] = np.asarray(mesh_colors, dtype=np.float32)
            alpha_lut[1:] = np.asarray(mesh_alphas, dtype=np.float32)
            colors = color_lut[ids]
            alphas = alpha_lut[ids][..., None]
            overlay = (image.astype(np.float32) * (1.0 - alphas) + colors * alphas).astype(np.uint8)

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

    if output_type == "grid" and grid_dir is not None and frame_indices:
        start_number = int(video_frame_ids[frame_indices[0]])
        ffmpeg_path = shutil.which("ffmpeg")
        if ffmpeg_path is None:
            raise RuntimeError("ffmpeg not found in PATH; cannot build grid video.")
        input_pattern = os.path.join(grid_dir, "frame_%05d.png")
        output_path = os.path.join(grid_dir, "grid_overlay.mp4")
        cmd = [
            ffmpeg_path, "-y",
            "-framerate", "30",
            "-start_number", str(start_number),
            "-i", input_pattern,
            "-c:v", "libx264",
            "-pix_fmt", "yuv420p",
            "-vf", "scale=1920:-2",
            output_path,
        ]
        subprocess.run(cmd, check=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--object", type=str, required=True)
    parser.add_argument("--ep", type=str, default="0")
    parser.add_argument("--capture-root", type=str, default="eccv2026/allegro_v5")
    parser.add_argument("--overlay-option", type=str, choices=["action", "position"], default="position")
    parser.add_argument("--start-frame", type=int, default=0)
    parser.add_argument("--end-frame", type=int, default=None)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument(
        "--frames", type=int, nargs="+", default=None,
        help="Explicit frame indices to render (overrides --start-frame/--end-frame/--stride).",
    )
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--output-type", type=str, choices=["video", "grid"], default="grid")
    parser.add_argument("--grid-scale", type=float, default=0.25)
    parser.add_argument("--frame-offset", type=int, default=0)
    parser.add_argument("--arm-time-offset", type=float, default=0.09)
    parser.add_argument("--hand-time-offset", type=float, default=0.0)
    args = parser.parse_args()

    project_allegro_v5(
        object_name=args.object,
        capture_ep=args.ep,
        capture_root=args.capture_root,
        overlay_option=args.overlay_option,
        start_frame=args.start_frame,
        end_frame=args.end_frame,
        stride=args.stride,
        frames=args.frames,
        output_dir=args.output_dir,
        output_type=args.output_type,
        grid_scale=args.grid_scale,
        frame_offset=args.frame_offset,
        arm_time_offset=args.arm_time_offset,
        hand_time_offset=args.hand_time_offset,
    )


if __name__ == "__main__":
    main()
