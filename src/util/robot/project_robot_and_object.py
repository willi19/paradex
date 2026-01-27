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
import torch
import trimesh

from paradex.calibration.utils import load_camparam
from paradex.utils.path import rsc_path, home_path, shared_dir
from paradex.visualization.robot import RobotModule
from paradex.robot.inspire import inspire_action_to_qpos
from paradex.utils.load_data import load_series, resample_to, load_image
from paradex.image.overlay import overlay_mask
from paradex.image.grid import make_image_grid
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
    overlay_option,
    output_type,
    device
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
    faces = None
    
    
    ## Robot projection
    if project_robot:
        arm_qpos, arm_time = load_series(arm_dir, ("position.npy", "action_qpos.npy", "action.npy"))

        # Overlay type
        if overlay_option == "action":
            hand_action, hand_time = load_series(hand_dir, ("action.npy",))
        else:
            hand_action, hand_time = load_series(hand_dir, ("position.npy",))

        hand_action = resample_to(hand_time, hand_action, arm_time)
        
        if hand == "inspire":
            hand_qpos = inspire_action_to_qpos(hand_action)
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
        urdf_path = os.path.join(rsc_path, "robot", f"{arm}_{hand}_left_new.urdf")
        robot = RobotModule(urdf_path)
        # Prepare face indices once (topology is fixed across frames)
        robot.update_cfg(full_qpos[0])
        base_mesh = robot.get_robot_mesh()
        faces = torch.tensor(base_mesh.faces, dtype=torch.int32, device=device)
        
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
        obj_faces = torch.tensor(obj_mesh.faces, dtype=torch.int32, device=device)
    
        # Align object trajectory to video timeline and express in robot frame.
        obj_time = np.linspace(video_times[0], video_times[-1], obj_traj_raw.shape[0])
        obj_traj_video = resample_to(
            obj_time, obj_traj_raw.reshape(obj_traj_raw.shape[0], -1), video_times
        ).reshape(len(video_times), 4, 4)
        obj_traj_robot = np.einsum("ij,tjk->tik", robot_from_world, obj_traj_video)
    
    ## Overlay Option
    output_dir = os.path.join(capture_ep_root, "overlay_" + overlay_option)
    os.makedirs(output_dir, exist_ok=True)
    
    
    image_dir = os.path.join(capture_ep_root, "video_extracted")
    
    # When robot is projected, timeline comes from qpos_video; otherwise rely on loaded timestamps or image count.
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
    
    
    # Prepare renderers per camera
    renderer_dict = {}
    cam_info = {}
    for cam_id, intr in intrinsic.items():
        K = intr["intrinsics_undistort"]
        height = intr["height"]
        width = intr["width"]
        cam_from_world = np.eye(4)
        cam_from_world[:3, :] = extrinsic_from_camparam[cam_id]
        
        extr_full = cam_from_world @ c2r
        extr = extr_full[:3, :]

        # renderer = BatchRenderer(
        #     opengl=False,
        #     cam_intrinsics=[K],
        #     cam_extrinsics=[extr],
        #     width=width,
        #     height=height,
        #     near=0.01,
        #     far=2.0,
        #     device=device,
        # )
        
        renderer = BatchRenderer(K, extr)
        
        renderer_dict[cam_id] = renderer
        cam_info[cam_id] = {"K": K, "extr": extr, "width": width, "height": height}
    
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
        robot_obj = None
        if project_robot:
            robot.update_cfg(qpos_video[fidx])
            mesh = robot.get_robot_mesh()
            verts = torch.tensor(mesh.vertices, dtype=torch.float32, device=device)[None, ...]
            vtx_col = torch.ones((verts.shape[1], 3), dtype=torch.float32, device=device)
            robot_obj = {
                "type": "vertex_color",
                "verts": verts,
                "faces": faces,
                "vtx_col": vtx_col,
                "col_idx": faces,
            }

        object_obj = None
        if project_object:
            obj_pose = obj_traj_robot[fidx]
            obj_verts_np = apply_transform(obj_base_vertices, obj_pose)
            obj_verts = torch.tensor(obj_verts_np, dtype=torch.float32, device=device)[None, ...]
            obj_vtx_col = torch.ones((obj_verts.shape[1], 3), dtype=torch.float32, device=device)
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

            render_objs = []
            if robot_obj is not None:
                render_objs.append(robot_obj)
            if object_obj is not None:
                render_objs.append(object_obj)
            image = load_image(image_dir, cam_id, int(video_frame_ids[fidx]), (height, width))  # files are 1-indexed
            overlay = image
            if render_objs:
                mask_ids = renderer.render_id(render_objs)
                mask_ids = mask_ids[0, ..., 0].detach().cpu().numpy()
                robot_mask = (mask_ids > 0.5) & (mask_ids < 1.5)
                object_mask = mask_ids >= 1.5 if object_obj is not None else np.zeros_like(robot_mask)
                if robot_mask.any():
                    overlay = overlay_mask(overlay, robot_mask.astype(np.float32), color=(0, 255, 0), alpha=0.5)
                if object_mask.any():
                    overlay = overlay_mask(overlay, object_mask.astype(np.float32), color=(255, 0, 0), alpha=0.5)

            if output_type == "video":
                writers_overlay[cam_id].write(cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
            else:
                overlays_for_grid.append(overlay)
            # writers_mask[cam_id].write((mask * 255).astype(np.uint8))

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
    parser.add_argument("--output-dir", type=str, default="/home/temp_id/shared_data/capture/hri_inspire_left", help="Output directory for projected masks/overlays.")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--overlay-option", type=str, choices=["action", "position"], default="position", help="Whether to overlay using hand action or hand position data.")
    parser.add_argument("--output-type", type=str, choices=["video", "grid"], default="grid", help="Output overlaid result as videos (per camera) or as tiled grid images per frame.",)
    parser.add_argument("--overlay-to-other-video", action="store_true", help="Overlay the projected masks onto another video.")
    

    args = parser.parse_args()

    if args.replay_ep is None:
        args.replay_ep = args.capture_ep

    # project_robot_and_object(
    #     arm=args.arm,
    #     hand=args.hand,
    #     object=args.object,
    #     capture_root=args.capture_root,
    #     capture_ep=args.capture_ep,
    #     replay_ep=args.replay_ep,
    #     overlay_to_other_video=args.overlay_to_other_video,
    #     object_mesh_name=args.object_mesh_name,
    #     project_robot=args.project_robot,
    #     project_object=args.project_object,
    #     start_frame=args.start_frame,
    #     end_frame=args.end_frame,
    #     stride=args.stride,
    #     overlay_option=args.overlay_option,
    #     output_type=args.output_type,
    #     device=args.device,
    # )

    # return

    capture_root = os.path.join(shared_dir, "capture", args.capture_root, args.object, str(args.capture_ep))

    raw_root = os.path.join(capture_root, "raw")
    arm_dir = os.path.join(raw_root, "arm")
    hand_dir = os.path.join(raw_root, "hand")

    # Load recorded trajectories only when projecting the robot; otherwise avoid relying on arm_time.
    qpos_video = None
    video_times = None
    video_frame_ids = None
    
    if args.project_robot:
        arm_qpos, arm_time = load_series(arm_dir, ("position.npy", "action_qpos.npy", "action.npy"))

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
    else:
        # If not projecting the robot, fall back to timestamps if they exist; otherwise infer from images later.
        ts_dir = os.path.join(raw_root, "timestamps")
        ts_path = os.path.join(ts_dir, "timestamp.npy")
        frame_id_path = os.path.join(ts_dir, "frame_id.npy")
        if os.path.exists(ts_path) and os.path.exists(frame_id_path):
            video_times = np.load(ts_path)
            video_frame_ids = np.load(frame_id_path)

    robot = None
    faces = None
    if args.project_robot:
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

    if args.project_object:
        if not args.object_mesh or not args.object_trajectory:
            raise ValueError("--object-mesh and --object-trajectory are required when --project-object is set.")
        obj_traj_raw = load_object_trajectory(args.object_trajectory)
        print(f"Loaded object trajectory with {obj_traj_raw.shape[0]} frames from {args.object_trajectory}")
        obj_mesh = load_object_mesh(args.object_mesh)
        obj_base_vertices = np.asarray(obj_mesh.vertices, dtype=np.float32)
        obj_faces = torch.tensor(obj_mesh.faces, dtype=torch.int32, device=args.device)

    robot_from_world = np.linalg.inv(c2r)

    if args.overlay_option == "action":
        output_dir = os.path.join(args.output_dir, f"{args.object}", f"{args.capture_ep}", "overlay_action")
    else:
        output_dir = os.path.join(args.output_dir, f"{args.object}", f"{args.capture_ep}", "overlay_position")
    os.makedirs(output_dir, exist_ok=True)

    output_dir = os.path.join(args.output_dir, f"{args.object}", str(args.capture_ep) + "_replay")

    image_dir = os.path.join(capture_root, "video_extracted")
        
    # When robot is projected, timeline comes from qpos_video; otherwise rely on loaded timestamps or image count.
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

    start = max(0, args.start_frame)
    end = total_frames if args.end_frame is None else min(args.end_frame, total_frames)
    if start >= end:
        raise ValueError(f"Invalid frame range: start={start}, end={end}, total={total_frames}")
    frame_indices = list(range(start, end, max(1, args.stride)))

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

        # renderer = BatchRenderer(
        #     opengl=False,
        #     cam_intrinsics=[K],
        #     cam_extrinsics=[extr],
        #     width=width,
        #     height=height,
        #     near=0.01,
        #     far=2.0,
        #     device=args.device,
        # )
        
        intrinsic, extrinsic_from_camparam = load_camparam(capture_root)

        renderer = BatchRenderer(intrinsic, extrinsic_from_camparam)

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

    
    if args.project_object:
        # Align object trajectory to video timeline and express in robot frame.
        obj_time = np.linspace(video_times[0], video_times[-1], obj_traj_raw.shape[0])
        obj_traj_video = resample_to(
            obj_time, obj_traj_raw.reshape(obj_traj_raw.shape[0], -1), video_times
        ).reshape(len(video_times), 4, 4)
        obj_traj_robot = np.einsum("ij,tjk->tik", robot_from_world, obj_traj_video)

    for fidx in frame_indices:
        print(f"Processing frame {fidx} / {total_frames}...")
        robot_obj = None
        if args.project_robot:
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

        object_obj = None
        if args.project_object:
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

            render_objs = []
            if robot_obj is not None:
                render_objs.append(robot_obj)
            if object_obj is not None:
                render_objs.append(object_obj)
            image = load_image(image_dir, cam_id, int(video_frame_ids[fidx]), (height, width))  # files are 1-indexed
            overlay = image
            if render_objs:
                mask_ids = renderer.render_id(render_objs)
                mask_ids = mask_ids[0, ..., 0].detach().cpu().numpy()
                robot_mask = (mask_ids > 0.5) & (mask_ids < 1.5)
                object_mask = mask_ids >= 1.5 if object_obj is not None else np.zeros_like(robot_mask)
                if robot_mask.any():
                    overlay = overlay_mask(overlay, robot_mask.astype(np.float32), color=(0, 255, 0), alpha=0.5)
                if object_mask.any():
                    overlay = overlay_mask(overlay, object_mask.astype(np.float32), color=(255, 0, 0), alpha=0.5)

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
    if args.output_type == "grid" and grid_dir is not None:
        if not frame_indices:
            raise ValueError("No frames rendered for grid output; cannot build video.")
        start_number = int(video_frame_ids[frame_indices[0]])
        ffmpeg_path = shutil.which("ffmpeg")
        if ffmpeg_path is None:
            raise RuntimeError("ffmpeg not found in PATH; cannot build grid video.")
        input_pattern = os.path.join(grid_dir, "frame_%05d.png")
        output_path = os.path.join(output_dir, "grid_4k.mp4")
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


if __name__ == "__main__":
    main()
