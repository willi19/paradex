import argparse
import glob
import math
import os
import pickle
import shutil
import subprocess
import sys
import cv2

from typing import Any, List, Optional, Tuple

import numpy as np
import trimesh

from paradex.calibration.utils import load_camparam
from paradex.utils.path import rsc_path, home_path, shared_dir
from paradex.visualization.robot import RobotModule
from paradex.robot.inspire import inspire_action_to_qpos, inspire_action_to_qpos_dof12
from paradex.utils.load_data import load_series, resample_to
from paradex.image.overlay import overlay_mask
from paradex.image.grid import make_image_grid
from paradex.image.undistort import precomute_undistort_map, apply_undistort_map
from paradex.object.utils import load_object_trajectory, load_object_mesh, apply_transform

# PROCESSING_REPO = "/home/temp_id/paradex_processing_latest"
# if PROCESSING_REPO not in sys.path:
#     sys.path.insert(0, PROCESSING_REPO)
# from utils.vis_utils_nvdiff import BatchRenderer

from paradex.image.projection import BatchRenderer


def project_robot_and_object(
    arm, 
    hand,
    object,
    capture_root,           # hri_inspire_left
    capture_ep,             # Projection의 background가 되는 image들이 촬영된 episode
    replay_ep,              # project되는 것들이 기록된 episode (robot action, object trajectory)
    overlay_to_other_video, # if false, replay_ep = capture_ep
    object_mesh_name,
    project_robot,
    project_object,
    start_frame,
    end_frame,
    stride,
    output_dir,
    overlay_option,
    output_type,
    device,
    frame_offset,
    arm_time_offset,
    hand_time_offset
):
    
    
    capture_ep_root = os.path.join(shared_dir, "capture", capture_root, object, str(capture_ep))
    replay_ep_root = os.path.join(shared_dir, "capture", capture_root, object, str(replay_ep))
    
    capture_raw_root = os.path.join(capture_ep_root, "raw")
    replay_raw_root = os.path.join(replay_ep_root, "raw")
    
    arm_dir = os.path.join(replay_raw_root, "arm")
    hand_dir = os.path.join(replay_raw_root, "hand")
    
    intrinsic, extrinsic_from_camparam = load_camparam(capture_ep_root)
    # C2R is world->robot (hand-eye calibration result).
    c2r = np.load(os.path.join(capture_ep_root, "C2R.npy"))
    robot_from_world = np.linalg.inv(c2r)
    
    qpos_video = None
    video_times = None
    video_frame_ids = None
    
    robot = None
    robot_dof = None
    link_label_map = None
    finger_prefix_map = None
    finger_colors = None
    
    
    ## Robot projection
    if project_robot:
        arm_qpos, arm_time = load_series(arm_dir, ("position.npy", "action_qpos.npy", "action.npy"))

        # Overlay type
        if overlay_option == "action":
            hand_action, hand_time = load_series(hand_dir, ("action.npy",))
        else:
            hand_action, hand_time = load_series(hand_dir, ("position.npy",))

        if arm_time_offset != 0.0:
            arm_time = arm_time + arm_time_offset
        if hand_time_offset != 0.0:
            hand_time = hand_time + hand_time_offset

        hand_action = resample_to(hand_time, hand_action, arm_time)
        
        if hand == "inspire":
            hand_qpos = inspire_action_to_qpos_dof12(hand_action)
        else:
            hand_qpos = hand_action
            
        full_qpos = np.concatenate([arm_qpos, hand_qpos], axis=1)
        # Sync to RGB timestamps if available.
        ts_dir = os.path.join(capture_raw_root, "timestamps")
        ts_path = os.path.join(ts_dir, "timestamp.npy")
        frame_id_path = os.path.join(ts_dir, "frame_id.npy")
        
        # Resample actions according to video times
        if os.path.exists(ts_path) and os.path.exists(frame_id_path):
            video_times = np.load(ts_path)
            # video_times = np.load(frame_id_path)
            video_frame_ids = np.load(frame_id_path)
            qpos_video = resample_to(arm_time, full_qpos, video_times)
        else:
            video_times = arm_time
            video_frame_ids = np.arange(1, full_qpos.shape[0] + 1, dtype=int)
            qpos_video = full_qpos
        
        # Load robot urdf and mesh
        urdf_path = os.path.join(rsc_path, "robot", f"{arm}_{hand}_left_new_copy.urdf")
        robot = RobotModule(urdf_path)
        
        
        print("=== Robot joint order (qpos order) ===")
        for i, name in enumerate(robot.joint_names):
            print(i, name)

        robot_dof = robot.get_num_joints()
            
        robot_dof = robot.get_num_joints()
        # Prepare mesh topology once to avoid reloading URDF each frame.
        robot.update_cfg(full_qpos[0, :robot_dof])
        robot.get_robot_mesh()
        finger_prefix_map = {
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
        # If not projecting the robot, fall back to timestamps if they exist; otherwise infer from images later.
        ts_dir = os.path.join(capture_raw_root, "timestamps")
        ts_path = os.path.join(ts_dir, "timestamp.npy")
        frame_id_path = os.path.join(ts_dir, "frame_id.npy")
        if os.path.exists(ts_path) and os.path.exists(frame_id_path):
            video_times = np.load(ts_path)
            video_frame_ids = np.load(frame_id_path)
    
    

    ## Object projection
    if project_object:
        
        obj_traj_path = os.path.join(replay_ep_root, "object_tracking")
        obj_traj_raw = load_object_trajectory(obj_traj_path)
        
        
        print(f"Loaded object trajectory with {obj_traj_raw.shape[0]} frames from {obj_traj_path}")
        
        object_mesh_path = os.path.join(shared_dir, "mesh", object_mesh_name, object_mesh_name + ".obj")
        
        obj_mesh = load_object_mesh(object_mesh_path)
        obj_base_vertices = np.asarray(obj_mesh.vertices, dtype=np.float32)
    
        # Align object trajectory to video timeline and express in robot frame.
        obj_time = np.linspace(video_times[0], video_times[-1], obj_traj_raw.shape[0])
        obj_traj_video = resample_to(
            obj_time, obj_traj_raw.reshape(obj_traj_raw.shape[0], -1), video_times
        ).reshape(len(video_times), 4, 4)
        obj_traj_robot = np.einsum("ij,tjk->tik", robot_from_world, obj_traj_video)
    
    ## Overlay Option
    
    if output_dir == None:
        output_dir = os.path.join(capture_ep_root, "overlay_" + overlay_option)
        
        
    os.makedirs(output_dir, exist_ok=True)
    
    
    image_dir = os.path.join(capture_ep_root, "video_extracted")
    
    # When robot is projected, timeline comes from qpos_video; otherwisfe rely on loaded timestamps or image count.
    if qpos_video is not None:
        total_frames = qpos_video.shape[0]
    elif video_frame_ids is not None:
        total_frames = len(video_frame_ids)
    else:
        # Infer frame count and frame ids from extracted images.
        if not os.path.isdir(image_dir):
            raise ValueError(f"Cannot infer frames: image_dir does not exist ({image_dir}).")
        cam_dirs = [
            os.path.join(image_dir, cam)
            for cam in sorted(os.listdir(image_dir))
            if os.path.isdir(os.path.join(image_dir, cam))
        ]
        first_cam_dir = cam_dirs[0] if cam_dirs else None
        if first_cam_dir is None:
            raise ValueError("Cannot infer frames: no timestamps and no image directories found.")
        num_images = len(sorted(glob.glob(os.path.join(first_cam_dir, "*.jpg"))))
        if num_images == 0:
            raise ValueError("Cannot infer frames: no images found in image_dir.")
        total_frames = num_images
        video_frame_ids = np.arange(1, num_images + 1, dtype=int)
        video_times = np.arange(num_images, dtype=float)
        
    start = max(0, start_frame)
    end = total_frames if end_frame is None else min(end_frame, total_frames)
    if start >= end:
        raise ValueError(f"Invalid frame range: start={start}, end={end}, total={total_frames}")
    frame_indices = list(range(start, end, max(1, stride)))
    
    # Build camera info and extrinsics (robot frame -> each camera frame) for BatchRenderer.
    cam_info = {}
    render_extrinsics = {}
    undistort_maps = {}
    for cam_id, intr in intrinsic.items():
        K = intr["intrinsics_undistort"]
        height = intr["height"]
        width = intr["width"]

        cam_from_world = np.eye(4)
        cam_from_world[:3, :] = extrinsic_from_camparam[cam_id]
        cam_from_robot = cam_from_world @ c2r

        render_extrinsics[cam_id] = cam_from_robot[:3, :]
        cam_info[cam_id] = {"K": K, "extr": cam_from_robot[:3, :], "width": width, "height": height}
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
            # writers_mask[cam_id] = cv2.VideoWriter(
            #     os.path.join(output_dir, f"{cam_id}_mask.mp4"), fourcc, 10, (w, h), isColor=False
            # )
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

            frame_num = int(video_frame_ids[fidx]) + frame_offset
            img_path = os.path.join(image_dir, cam_id, f"{frame_num:05d}.jpg")
            if os.path.exists(img_path):
                image_bgr = cv2.imread(img_path)
                if image_bgr is None:
                    image_bgr = np.zeros((height, width, 3), dtype=np.uint8)
                if image_bgr.shape[0] != height or image_bgr.shape[1] != width:
                    image_bgr = cv2.resize(image_bgr, (width, height))
                mapx, mapy = undistort_maps[cam_id]
                image_bgr = apply_undistort_map(image_bgr, mapx, mapy)
                image = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
            else:
                image = np.zeros((height, width, 3), dtype=np.uint8)
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
            frame_name = int(video_frame_ids[fidx])
            cv2.imwrite(
                os.path.join(grid_dir, f"frame_{frame_name:05d}.png"),
                cv2.cvtColor(grid_img, cv2.COLOR_RGB2BGR),
            )
    
    
    for w in writers_overlay.values():
        w.release()
    # for w in writers_mask.values():
    #     w.release()
    if output_type == "grid" and grid_dir is not None:
        if not frame_indices:
            raise ValueError("No frames rendered for grid output; cannot build video.")
        start_number = int(video_frame_ids[frame_indices[0]])
        ffmpeg_path = shutil.which("ffmpeg")
        if ffmpeg_path is None:
            raise RuntimeError("ffmpeg not found in PATH; cannot build grid video.")
        input_pattern = os.path.join(grid_dir, "frame_%05d.png")
        output_path = os.path.join(grid_dir, f"{replay_ep}_to_{capture_ep}_grid_4k.mp4")
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
            "scale=3840:-2",
            output_path,
        ]
        subprocess.run(cmd, check=True)
    
    return    


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--arm", type=str, default="xarm")
    parser.add_argument("--hand", type=str, default="inspire")
    parser.add_argument("--object", type=str, required=True)
    parser.add_argument("--capture-ep", type=str, default="0")
    parser.add_argument("--replay-ep", type=str, default=None)
    parser.add_argument("--object-mesh-name", type=str, help="Path to object mesh (e.g., .obj/.stl).")
    parser.add_argument("--capture-root", type=str, default="hri_inspire_left", help="Capture root directory name.")
    # parser.add_argument("--object-trajectory", type=str, help="Path to trajectory pickle/npy/npz (or directory containing it).",)
    parser.add_argument("--project-object", action="store_true", help="Project the tracked object mesh in addition to the robot.")
    parser.add_argument("--project-robot", action="store_true", help="Project the robot mesh.")
    parser.add_argument("--start-frame", type=int, default=0, help="Start frame index (inclusive).")
    parser.add_argument("--end-frame", type=int, default=None, help="End frame index (exclusive). Defaults to full length.")
    parser.add_argument("--stride", type=int, default=1, help="Frame stride.")
    parser.add_argument("--output-dir", type=str, default=None, help="Output directory for projected masks/overlays.")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--overlay-option", type=str, choices=["action", "position"], default="position", help="Whether to overlay using hand action or hand position data.")
    parser.add_argument("--output-type", type=str, choices=["video", "grid"], default="grid", help="Output overlaid result as videos (per camera) or as tiled grid images per frame.",)
    parser.add_argument("--overlay-to-other-video", action="store_true", help="Overlay the projected masks onto another video.")
    parser.add_argument("--frame-offset", type=int, default=0, help="Shift overlay target video frames by this offset.")
    parser.add_argument("--arm-time-offset", type=float, default=0.0, help="Shift arm timestamps by this many seconds (positive delays arm).")
    parser.add_argument("--hand-time-offset", type=float, default=0.0, help="Shift hand timestamps by this many seconds (positive delays hand).")
    

    args = parser.parse_args()

    if args.replay_ep is None:
        args.replay_ep = args.capture_ep

    project_robot_and_object(
        arm=args.arm,
        hand=args.hand,
        object=args.object,
        capture_root=args.capture_root,
        capture_ep=args.capture_ep,
        replay_ep=args.replay_ep,
        overlay_to_other_video=args.overlay_to_other_video,
        object_mesh_name=args.object_mesh_name,
        project_robot=args.project_robot,
        project_object=args.project_object,
        start_frame=args.start_frame,
        end_frame=args.end_frame,
        stride=args.stride,
        output_dir=args.output_dir,
        overlay_option=args.overlay_option,
        output_type=args.output_type,
        device=args.device,
        frame_offset=args.frame_offset,
        arm_time_offset=args.arm_time_offset,
        hand_time_offset=args.hand_time_offset,
    )


if __name__ == "__main__":
    main()
