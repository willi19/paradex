import argparse
import math
import os
import sys
from typing import List, Tuple

import numpy as np
import torch

from paradex.calibration.utils import load_camparam
from paradex.dataset_acqusition.match_sync import fill_framedrop, get_synced_data
from paradex.utils.path import rsc_path
from paradex.visualization.robot import RobotModule


PROCESSING_REPO = "/home/temp_id/paradex_processing_latest"
if PROCESSING_REPO not in sys.path:
    sys.path.insert(0, PROCESSING_REPO)
from utils.vis_utils_nvdiff import BatchRenderer


def load_series(data_dir: str, candidates: Tuple[str, ...]) -> Tuple[np.ndarray, np.ndarray]:
    # Load first existing npy among candidates and its time.npy if present.
    for name in candidates:
        path = os.path.join(data_dir, name)
        if os.path.exists(path):
            data = np.load(path)
            time_path = os.path.join(data_dir, "time.npy")
            if os.path.exists(time_path):
                t = np.load(time_path)
            else:
                t = np.arange(data.shape[0], dtype=float)
            # Align lengths if off-by-one between data and time.
            if len(t) != data.shape[0]:
                n = min(len(t), data.shape[0])
                data = data[:n]
                t = t[:n]
            return data, t
    raise FileNotFoundError(f"No data found in {data_dir} for {candidates}")


def resample_to(times_src: np.ndarray, data_src: np.ndarray, times_dst: np.ndarray) -> np.ndarray:
    # Simple per-joint linear interpolation onto destination timestamps.
    if data_src.shape[0] == times_dst.shape[0] and np.allclose(times_src, times_dst):
        return data_src
    order = np.argsort(times_src)
    times_src = times_src[order]
    data_src = data_src[order]
    out = np.zeros((times_dst.shape[0], data_src.shape[1]), dtype=float)
    for j in range(data_src.shape[1]):
        out[:, j] = np.interp(times_dst, times_src, data_src[:, j])
    return out


def inspire_action_to_qpos(action: np.ndarray) -> np.ndarray:
    # Map Inspire raw action (little, ring, middle, index, thumb_2, thumb_1) to joint angles.
    limits = {
        "pinky_proximal_joint": 1.6,
        "ring_proximal_joint": 1.6,
        "middle_proximal_joint": 1.6,
        "index_proximal_joint": 1.6,
        "thumb_proximal_pitch_joint": 0.55,
        "thumb_proximal_yaw_joint": 1.15,
    }
    qpos = np.zeros_like(action, dtype=float)
    qpos[:, 0] = limits["thumb_proximal_yaw_joint"] * (1.0 - action[:, 5] / 1000.0)
    qpos[:, 1] = limits["thumb_proximal_pitch_joint"] * (1.0 - action[:, 4] / 1000.0)
    qpos[:, 2] = limits["index_proximal_joint"] * (1.0 - action[:, 3] / 1000.0)
    qpos[:, 3] = limits["middle_proximal_joint"] * (1.0 - action[:, 2] / 1000.0)
    qpos[:, 4] = limits["ring_proximal_joint"] * (1.0 - action[:, 1] / 1000.0)
    qpos[:, 5] = limits["pinky_proximal_joint"] * (1.0 - action[:, 0] / 1000.0)
    return qpos


def load_image(image_dir: str, cam_id: str, frame_idx: int, size: Tuple[int, int]) -> np.ndarray:
    # Frames are stored as 1-indexed jpgs (00001.jpg, ...).
    filename = os.path.join(image_dir, cam_id, f"{frame_idx:05d}.jpg")
    if os.path.exists(filename):
        import cv2

        image = cv2.imread(filename)
        image = cv2.resize(image, (size[1], size[0]))
        return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return np.zeros((size[0], size[1], 3), dtype=np.uint8)


def overlay_mask(image: np.ndarray, mask: np.ndarray, color=(0, 255, 0), alpha=0.5) -> np.ndarray:
    # Alpha-blend binary mask onto RGB image.
    mask = (mask > 0.5).astype(np.float32)
    overlay = image.astype(np.float32).copy()
    color_arr = np.array(color, dtype=np.float32)[None, None, :]
    overlay = overlay * (1 - alpha * mask[..., None]) + color_arr * (alpha * mask[..., None])
    return overlay.astype(np.uint8)


def make_image_grid(images: List[np.ndarray]) -> np.ndarray:
    # Tile images (RGB) into a nearly square grid.
    if not images:
        return np.zeros((1, 1, 3), dtype=np.uint8)
    base_h, base_w = images[0].shape[:2]
    cols = int(math.ceil(math.sqrt(len(images))))
    rows = int(math.ceil(len(images) / cols))
    grid = np.zeros((rows * base_h, cols * base_w, 3), dtype=np.uint8)
    for idx, img in enumerate(images):
        if img.shape[:2] != (base_h, base_w):
            import cv2

            img = cv2.resize(img, (base_w, base_h))
        r, c = divmod(idx, cols)
        grid[r * base_h : (r + 1) * base_h, c * base_w : (c + 1) * base_w] = img
    return grid


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--arm", type=str, required=True)
    parser.add_argument("--hand", type=str, required=True)
    parser.add_argument("--object", type=str, required=True)
    parser.add_argument("--episode", type=int, default=0)
    parser.add_argument("--start-frame", type=int, default=0, help="Start frame index (inclusive).")
    parser.add_argument("--end-frame", type=int, default=None, help="End frame index (exclusive). Defaults to full length.")
    parser.add_argument("--stride", type=int, default=1, help="Frame stride.")
    parser.add_argument(
        "--output-dir",
        type=str,
        default="/home/temp_id/shared_data/capture/hri_inspire_left",
        help="Output directory for projected masks/overlays.",
    )
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument(
        "--project-origin",
        action="store_true",
        help="Project the robot origin (0,0,0) into each camera image for debugging (logged for first frame).",
    )
    parser.add_argument(
        "--overlay-option",
        type=str,
        choices=["action", "position"],
        default="position",
        help="Whether to overlay using hand action or hand position data.",
    )
    parser.add_argument(
        "--output-type",
        type=str,
        choices=["video", "grid"],
        default="video",
        help="Output overlaid result as videos (per camera) or as tiled grid images per frame.",
    )
    args = parser.parse_args()

    capture_root = os.path.join("/home/temp_id/shared_data/capture/hri_inspire_left", args.object, str(args.episode))
    # output_dir = capture_root if args.output_dir is None else args.output_dir

    raw_root = os.path.join(capture_root, "raw")
    arm_dir = os.path.join(raw_root, "arm")
    hand_dir = os.path.join(raw_root, "hand")

    # Load recorded trajectories (arm: angles, hand: raw action)
    arm_qpos, arm_time = load_series(arm_dir, ("position.npy", "action_qpos.npy", "action.npy"))
    # hand_action, hand_time = load_series(hand_dir, ("action.npy", "position.npy"))
    
    if args.overlay_option == "action":
        hand_action, hand_time = load_series(hand_dir, ("action.npy",)) 
    else:
        hand_action, hand_time = load_series(hand_dir, ("position.npy",))
        
        
    hand_action = get_synced_data(arm_time, hand_action, hand_time)
    if args.hand == "inspire":
        hand_qpos = inspire_action_to_qpos(hand_action)
    else:
        hand_qpos = hand_action
    full_qpos = np.concatenate([arm_qpos, hand_qpos], axis=1)

    # Sync to RGB timestamps if available (use fill_framedrop + get_synced_data to match miyungpa pipeline).
    ts_dir = os.path.join(raw_root, "timestamps")
    ts_path = os.path.join(ts_dir, "timestamp.npy")
    frame_id_path = os.path.join(ts_dir, "frame_id.npy")
    if os.path.exists(ts_path) and os.path.exists(frame_id_path):
        pc_time_raw = np.load(ts_path)
        frame_id_raw = np.load(frame_id_path)
        # Reconstruct a continuous PC time line and frame ids to handle drops.
        video_times, video_frame_ids = fill_framedrop(frame_id_raw, pc_time_raw)
        # Snap arm/hand states onto the pc_time line.
        qpos_video = get_synced_data(video_times, full_qpos, arm_time)
    else:
        video_times = arm_time
        video_frame_ids = np.arange(1, full_qpos.shape[0] + 1, dtype=int)
        qpos_video = full_qpos

    total_frames = qpos_video.shape[0]
    start = max(0, args.start_frame)
    end = total_frames if args.end_frame is None else min(args.end_frame, total_frames)
    if start >= end:
        raise ValueError(f"Invalid frame range: start={start}, end={end}, total={total_frames}")
    frame_indices = list(range(start, end, max(1, args.stride)))

    urdf_path = os.path.join(rsc_path, "robot", f"{args.arm}_{args.hand}_left.urdf")
    robot = RobotModule(urdf_path)
    # Prepare face indices once (topology is fixed across frames)
    robot.update_cfg(full_qpos[0])
    base_mesh = robot.get_robot_mesh()
    faces = torch.tensor(base_mesh.faces, dtype=torch.int32, device=args.device)

    intrinsic, extrinsic_from_camparam = load_camparam(capture_root)
    # C2R is world->robot (hand-eye calibration result).
    c2r = np.load(os.path.join(capture_root, "C2R.npy"))
    # world_from_robot = c2r

    if args.overlay_option == "action":
        output_dir = os.path.join(args.output_dir, f"{args.object}", f"{args.episode}", "overlay_action")
    else:
        output_dir = os.path.join(args.output_dir, f"{args.object}", f"{args.episode}", "overlay_position")
    os.makedirs(output_dir, exist_ok=True)
    image_dir = os.path.join(capture_root, "video_extracted")

    # Prepare renderers per camera (intrinsic + extrinsic)
    renderer_dict = {}
    cam_info = {}
    for cam_id, intr in intrinsic.items():
        K = intr["intrinsics_undistort"]
        height = intr["height"]
        width = intr["width"]
        cam_from_world = np.eye(4)
        cam_from_world[:3, :] = extrinsic_from_camparam[cam_id]
        # R2C: cam_from_robot = cam_from_world ∘ world_from_robot
        extr_full = cam_from_world @ c2r
        extr = extr_full[:3, :]

        renderer = BatchRenderer(
            opengl=False,
            cam_intrinsics=[K],
            cam_extrinsics=[extr],
            width=width,
            height=height,
            near=0.01,
            far=2.0,
            device=args.device,
        )
        renderer_dict[cam_id] = renderer
        cam_info[cam_id] = {"K": K, "extr": extr, "width": width, "height": height}

    # Set up video writers per camera
    import cv2

    writers_overlay = {}
    grid_dir = None
    if args.output_type == "video":
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        for cam_id, info in cam_info.items():
            h, w = info["height"], info["width"]
            writers_overlay[cam_id] = cv2.VideoWriter(
                os.path.join(output_dir, f"{cam_id}_overlay.mp4"), fourcc, 30, (w, h)
            )
            # writers_mask[cam_id] = cv2.VideoWriter(
            #     os.path.join(output_dir, f"{cam_id}_mask.mp4"), fourcc, 10, (w, h), isColor=False
            # )
    else:
        grid_dir = os.path.join(output_dir, "grid")
        os.makedirs(grid_dir, exist_ok=True)

    image_dir = os.path.join(capture_root, "video_extracted")

    for fidx in frame_indices:
        print(f"Processing frame {fidx} / {total_frames}...")
        robot.update_cfg(qpos_video[fidx])
        mesh = robot.get_robot_mesh()
        verts = torch.tensor(mesh.vertices, dtype=torch.float32, device=args.device)[None, ...]
        vtx_col = torch.ones_like(verts)
        obj_dict = {
            "type": "vertex_color",
            "verts": verts,
            "faces": faces,
            "vtx_col": vtx_col,
            "col_idx": faces,
        }

        overlays_for_grid = []
        for cam_id in sorted(renderer_dict.keys()):
            renderer = renderer_dict[cam_id]
            info = cam_info[cam_id]
            K = info["K"]
            extr = info["extr"]
            width = info["width"]
            height = info["height"]

            _, mask_soft = renderer.render(obj_dict)
            mask = mask_soft[0].squeeze(-1).detach().cpu().numpy()

            image = load_image(image_dir, cam_id, int(video_frame_ids[fidx]), (height, width))  # files are 1-indexed
            overlay = overlay_mask(image, mask, color=(0, 255, 0), alpha=0.5)

            if args.project_origin and fidx == frame_indices[0]:
                # Project robot origin (0,0,0,1) into the image.
                P = K @ extr  # 3x4
                homo = P @ np.array([0.0, 0.0, 0.0, 1.0])
                u = homo[0] / homo[2] if abs(homo[2]) > 1e-9 else np.inf
                v = homo[1] / homo[2] if abs(homo[2]) > 1e-9 else np.inf
                depth = homo[2]
                print(f"[{cam_id}] origin proj -> (u={u:.1f}, v={v:.1f}, depth={depth:.3f}) using cam_param+ C2R")
                if np.isfinite(u) and np.isfinite(v):
                    cv2.drawMarker(
                        overlay,
                        (int(round(u)), int(round(v))),
                        color=(255, 0, 0),
                        markerType=cv2.MARKER_CROSS,
                        markerSize=15,
                        thickness=2,
                    )

            if args.output_type == "video":
                writers_overlay[cam_id].write(cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
            else:
                overlays_for_grid.append(overlay)
            # writers_mask[cam_id].write((mask * 255).astype(np.uint8))

        if args.output_type == "grid" and overlays_for_grid:
            grid_img = make_image_grid(overlays_for_grid)
            frame_name = int(video_frame_ids[fidx])
            cv2.imwrite(
                os.path.join(grid_dir, f"frame_{frame_name:05d}.png"),
                cv2.cvtColor(grid_img, cv2.COLOR_RGB2BGR),
            )

    for w in writers_overlay.values():
        w.release()
    # for w in writers_mask.values():
    #     w.release()


if __name__ == "__main__":
    main()
