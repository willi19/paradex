import argparse
import glob
import math
import os
import pickle
import sys
from typing import Any, List, Optional, Tuple

import numpy as np
import torch
import trimesh

from paradex.calibration.utils import load_camparam
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


def to_numpy_array(x) -> np.ndarray:
    if isinstance(x, np.ndarray):
        return x
    if hasattr(x, "detach"):
        return np.asarray(x.detach())
    return np.asarray(x)


def _maybe_stack_array(payload: Any) -> Optional[np.ndarray]:
    if isinstance(payload, (list, tuple)):
        return None
    arr = to_numpy_array(payload)
    if arr.ndim == 3 and arr.shape[1:] == (4, 4):
        return arr.astype(float)
    return None


def _build_T_from_frame(frame: Any, idx: int) -> np.ndarray:
    if isinstance(frame, dict):
        for k in ("T", "pose", "world_T_obj", "T_world_obj", "T_object_world"):
            if k in frame:
                T = to_numpy_array(frame[k])
                break
        else:
            if "obj_R" in frame and "obj_t" in frame:
                R = to_numpy_array(frame["obj_R"])
                t = to_numpy_array(frame["obj_t"]).reshape(3)
                T = np.eye(4, dtype=float)
                T[:3, :3] = R
                T[:3, 3] = t
            else:
                raise ValueError(f"Frame {idx} missing T/obj_R/obj_t")
    else:
        T = to_numpy_array(frame)

    T = np.asarray(T)
    if T.shape == (4, 4):
        return T.astype(float)
    if T.ndim == 3 and T.shape[0] == 1 and T.shape[1:] == (4, 4):
        return T[0].astype(float)
    if T.size == 16:
        return T.reshape(4, 4).astype(float)
    if T.shape == (3, 4):
        padded = np.eye(4, dtype=float)
        padded[:3, :] = T
        return padded
    raise ValueError(f"Frame {idx} transform shape {T.shape} cannot be interpreted as 4x4")


def load_object_trajectory(track_path: str) -> np.ndarray:
    # Load pickle/npz/npy containing a sequence of 4x4 poses.
    if os.path.isdir(track_path):
        candidates = sorted(
            glob.glob(os.path.join(track_path, "*.pickle"))
            + glob.glob(os.path.join(track_path, "*.pkl"))
            + glob.glob(os.path.join(track_path, "*.npy"))
            + glob.glob(os.path.join(track_path, "*.npz"))
        )
        if not candidates:
            raise FileNotFoundError(f"No object trajectory found in {track_path}")
        path = candidates[0]
    else:
        path = track_path

    if path.endswith((".npy", ".npz")):
        payload = np.load(path, allow_pickle=True)
    else:
        with open(path, "rb") as f:
            payload = pickle.load(f)

    stacked = _maybe_stack_array(payload)
    if stacked is not None:
        return stacked

    if isinstance(payload, dict):
        for key in ("T", "poses", "traj", "trajectory"):
            if key in payload:
                stacked = _maybe_stack_array(payload[key])
                if stacked is not None:
                    return stacked
        frames = payload.get("frames", payload)
        if isinstance(frames, dict):
            frames = [frames[k] for k in sorted(frames.keys())]
    else:
        frames = payload

    if isinstance(frames, np.ndarray):
        arr = np.asarray(frames)
        if arr.ndim == 3 and arr.shape[1:] == (4, 4):
            return arr.astype(float)

    T_list = []
    for idx, frame in enumerate(frames):
        T_list.append(_build_T_from_frame(frame, idx))

    if not T_list:
        raise ValueError("No transforms loaded from object trajectory")

    return np.stack(T_list, axis=0)


def load_object_mesh(mesh_path: str) -> trimesh.Trimesh:
    mesh = trimesh.load(mesh_path, force="mesh")
    if isinstance(mesh, trimesh.Trimesh):
        return mesh
    if isinstance(mesh, list):
        return trimesh.util.concatenate(mesh)
    raise ValueError(f"Unexpected mesh type: {type(mesh)} from {mesh_path}")


def apply_transform(verts: np.ndarray, T: np.ndarray) -> np.ndarray:
    # Apply 4x4 transform to vertices (N,3) -> (N,3).
    homo = np.concatenate([verts, np.ones((verts.shape[0], 1), dtype=verts.dtype)], axis=1)
    transformed = (T @ homo.T).T
    return transformed[:, :3]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--arm", type=str, required=True)
    parser.add_argument("--hand", type=str, required=True)
    parser.add_argument("--object", type=str, required=True)
    parser.add_argument("--episode", type=int, default=0)
    parser.add_argument("--object-mesh", type=str, required=True, help="Path to object mesh (e.g., .obj/.stl).")
    parser.add_argument(
        "--object-trajectory",
        type=str,
        required=True,
        help="Path to trajectory pickle/npy/npz (or directory containing it).",
    )
    # parser.add_argument(
    #     "--capture-root",
    #     type=str,
    #     default="/home/temp_id/shared_data/capture/hri_inspire_left/blue_speaker/5",
    #     help="Capture root containing cam_param/ and raw/ directories.",
    # )
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
        
        
    hand_action = resample_to(hand_time, hand_action, arm_time)
    if args.hand == "inspire":
        hand_qpos = inspire_action_to_qpos(hand_action)
    else:
        hand_qpos = hand_action
    full_qpos = np.concatenate([arm_qpos, hand_qpos], axis=1)

    # Sync to RGB timestamps if available.
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

    total_frames = qpos_video.shape[0]
    start = max(0, args.start_frame)
    end = total_frames if args.end_frame is None else min(args.end_frame, total_frames)
    if start >= end:
        raise ValueError(f"Invalid frame range: start={start}, end={end}, total={total_frames}")
    frame_indices = list(range(start, end, max(1, args.stride)))

    urdf_path = os.path.join(rsc_path, "robot", f"{args.arm}_{args.hand}_left_new.urdf")
    robot = RobotModule(urdf_path)
    # Prepare face indices once (topology is fixed across frames)
    robot.update_cfg(full_qpos[0])
    base_mesh = robot.get_robot_mesh()
    faces = torch.tensor(base_mesh.faces, dtype=torch.int32, device=args.device)

    intrinsic, extrinsic_from_camparam = load_camparam(capture_root)
    # C2R is world->robot (hand-eye calibration result).
    c2r = np.load(os.path.join(capture_root, "C2R.npy"))
    # world_from_robot = c2r

    obj_traj_raw = load_object_trajectory(args.object_trajectory)
    print(f"Loaded object trajectory with {obj_traj_raw.shape[0]} frames from {args.object_trajectory}")
    obj_mesh = load_object_mesh(args.object_mesh)
    obj_base_vertices = np.asarray(obj_mesh.vertices, dtype=np.float32)
    obj_faces = torch.tensor(obj_mesh.faces, dtype=torch.int32, device=args.device)

    robot_from_world = np.linalg.inv(c2r)

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

    # Align object trajectory to video timeline and express in robot frame.
    obj_time = np.linspace(video_times[0], video_times[-1], obj_traj_raw.shape[0])
    obj_traj_video = resample_to(obj_time, obj_traj_raw.reshape(obj_traj_raw.shape[0], -1), video_times).reshape(
        len(video_times), 4, 4
    )
    obj_traj_robot = np.einsum("ij,tjk->tik", robot_from_world, obj_traj_video)

    for fidx in frame_indices:
        print(f"Processing frame {fidx} / {total_frames}...")
        robot.update_cfg(qpos_video[fidx])
        mesh = robot.get_robot_mesh()
        verts = torch.tensor(mesh.vertices, dtype=torch.float32, device=args.device)[None, ...]
        vtx_col = torch.ones((verts.shape[1], 3), dtype=torch.float32, device=args.device)
        robot_obj = {
            "type": "vertex_color",
            "verts": verts,
            "faces": faces,
            "vtx_col": vtx_col,
            "col_idx": faces,
        }

        obj_pose = obj_traj_robot[fidx]
        obj_verts_np = apply_transform(obj_base_vertices, obj_pose)
        obj_verts = torch.tensor(obj_verts_np, dtype=torch.float32, device=args.device)[None, ...]
        obj_vtx_col = torch.ones((obj_verts.shape[1], 3), dtype=torch.float32, device=args.device)
        object_obj = {
            "type": "vertex_color",
            "verts": obj_verts,
            "faces": obj_faces,
            "vtx_col": obj_vtx_col,
            "col_idx": obj_faces,
        }

        overlays_for_grid = []
        for cam_id in sorted(renderer_dict.keys()):
            renderer = renderer_dict[cam_id]
            info = cam_info[cam_id]
            K = info["K"]
            extr = info["extr"]
            width = info["width"]
            height = info["height"]

            mask_ids = renderer.render_id([robot_obj, object_obj])
            mask_ids = mask_ids[0, ..., 0].detach().cpu().numpy()
            robot_mask = (mask_ids > 0.5) & (mask_ids < 1.5)
            object_mask = mask_ids >= 1.5

            image = load_image(image_dir, cam_id, int(video_frame_ids[fidx]), (height, width))  # files are 1-indexed
            overlay = image
            if robot_mask.any():
                overlay = overlay_mask(overlay, robot_mask.astype(np.float32), color=(0, 255, 0), alpha=0.5)
            if object_mask.any():
                overlay = overlay_mask(overlay, object_mask.astype(np.float32), color=(255, 0, 0), alpha=0.5)

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
