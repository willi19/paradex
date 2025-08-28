import os
import numpy as np
import json
import logging
import cv2
import trimesh
import copy
import time
import torch
from concurrent.futures import ThreadPoolExecutor

from paradex.video.raw_video import fill_framedrop, get_synced_data
from paradex.utils.file_io import home_path, shared_dir, rsc_path, get_robot_urdf_path, download_dir, load_camparam, load_c2r
from paradex.visualization_.robot_module import Robot_Module
from paradex.visualization_.renderer import BatchRenderer
from paradex.utils.upload_file import copy_file
from paradex.video.convert_codec import change_to_h264
from paradex.image.overlay import overlay_mask
from paradex.image.projection import get_cammtx, project_point, project_mesh, project_mesh_nvdiff
from paradex.image.merge import merge_image

def load_robot_type(root_dir):
    raw_dir = os.path.join(os.path.join(root_dir, "raw"))
    if not os.path.exists(raw_dir):
        return None, None
    
    arm_name = None
    for tmp_name in ['xarm', 'franka']:
        if tmp_name in os.listdir(raw_dir):
            arm_name = tmp_name
            break
    
    hand_name = None
    for tmp_name in ['inspire', 'allegro']:
        if tmp_name in os.listdir(raw_dir):
            hand_name = tmp_name
            break
    
    return arm_name, hand_name

def match_sync(root_dir, logger=None):
    if logger:
        logger.update_status("match_sync : Checking directory structure...")
    
    raw_dir = os.path.join(os.path.join(root_dir, "raw"))
    if not os.path.exists(raw_dir):
        if logger:
            logger.error("No raw directory")
        return False
    
    arm_name, hand_name = load_robot_type(root_dir)
    
    if not os.path.exists(os.path.join(raw_dir, "timestamp", "camera_timestamp.json")):
        if logger:
            logger.error("match_sync : No timestamp file")
        return False
        
    timestamp = json.load(open(os.path.join(raw_dir, "timestamp", "camera_timestamp.json")))
    pc_time, fid = fill_framedrop(timestamp)
    
    if not pc_time:
        if logger:
            logger.error("match_sync : Timestamp file not valid")
        return False
    
    os.makedirs(os.path.join(root_dir), exist_ok=True)
    
    # process arm
    if arm_name:
        arm_action_orig = np.load(os.path.join(raw_dir, arm_name, "action.npy")) # T X 4 X 4
        arm_qpos_orig = np.load(os.path.join(raw_dir, arm_name, "position.npy")) # T X dof
        arm_pc_time = np.load(os.path.join(raw_dir, arm_name, "time.npy"))
        
        arm_action_sync = get_synced_data(pc_time, arm_action_orig, arm_pc_time)
        arm_qpos_sync = get_synced_data(pc_time, arm_qpos_orig, arm_pc_time)
        os.makedirs(os.path.join(root_dir, "arm"), exist_ok=True)
        np.save(os.path.join(root_dir, "arm", "action.npy"), arm_action_sync)
        np.save(os.path.join(root_dir, "arm", "qpos.npy"), arm_qpos_sync)
    
    if hand_name:
        hand_action_orig = np.load(os.path.join(raw_dir, hand_name, "action.npy"))
        hand_qpos_orig = np.load(os.path.join(raw_dir, hand_name, "position.npy"))
        hand_pc_time = np.load(os.path.join(raw_dir, hand_name, "time.npy"))
        
        hand_action_sync = get_synced_data(pc_time, hand_action_orig, hand_pc_time)
        hand_qpos_sync = get_synced_data(pc_time, hand_qpos_orig, hand_pc_time)
        
        os.makedirs(os.path.join(root_dir, "hand"), exist_ok=True)
        np.save(os.path.join(root_dir, "hand", "action.npy"), hand_action_sync)
        np.save(os.path.join(root_dir, "hand", "qpos.npy"), hand_qpos_sync)
    
    if os.path.exists(os.path.join(raw_dir, "state")):
        state_orig = np.load(os.path.join(raw_dir, "state", "state.npy"))
        if len(state_orig) != 0:
            
            state_pc_time = np.load(os.path.join(raw_dir, "state", "time.npy"))
            state_sync = get_synced_data(pc_time, state_orig, state_pc_time)
    
            np.save(os.path.join(root_dir, "state.npy"), state_sync)

    return True

def process_video_list(video_dir, data, process_frame, logger=None):
    video_name_list = os.listdir(video_dir)    
    cap_dict = {}
    finished = {}
    max_frames = -1
    for video_name in video_name_list:
        name = video_name.split(".")[0]
        cap_dict[name] = cv2.VideoCapture(os.path.join(video_dir, video_name))
        finished[name] = False
        fps = cap_dict[name].get(cv2.CAP_PROP_FPS)
        w = int(cap_dict[name].get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap_dict[name].get(cv2.CAP_PROP_FRAME_HEIGHT))
        num_frames = int(cap_dict[name].get(cv2.CAP_PROP_FRAME_COUNT))
        max_frames = max(max_frames, num_frames)
    
    fid = 0
    start_time = time.time()
    while True:
        img_dict = {}
        cnt = 0
        asdf_start_time = time.time()
        for name, cap in cap_dict.items():
            if finished[name]:
                img_dict[name] = np.zeros((h, w, 3))
                continue
            
            ret, frame = cap.read()
            if not ret:
                finished[name] = True
                img_dict[name] = np.zeros((h, w, 3))
                continue
            
            img_dict[name] = frame.copy()
            cnt += 1
        
        if cnt == 0:
            break
        
        process_frame(img_dict, video_dir, fid, data)
        fid += 1
        
        # ETA 계산
        elapsed_time = time.time() - start_time
        if fid > 0:
            time_per_frame = elapsed_time / fid
            remaining_frames = max_frames - fid
            eta_seconds = time_per_frame * remaining_frames
            eta_str = f"{int(eta_seconds//60):02d}:{int(eta_seconds%60):02d}"
        else:
            eta_str = "calculating..."
        
        if logger:
            logger.update_status(f"Processing frame {fid}/{max_frames} - ETA: {eta_str}")
        print(f"Processing frame {fid}/{max_frames} - ETA: {eta_str}")
        
    for _, cap in cap_dict.items():
        cap.release()

def download_files(src_dir, dst_dir, logger=None):
    """
    Copy all files from src_dir to dst_dir/overlay.
    - Skip if the file already exists in dst_dir.
    - Copy to a temporary file (<name>._tmp) first, then rename to the final file.
    - Returns: list of file names (without extension), sorted.
    """
    
    os.makedirs(dst_dir, exist_ok=True)
    for video_name in os.listdir(src_dir):
        src_file = os.path.join(src_dir, video_name)
        dst_file = os.path.join(dst_dir, video_name)
        tmp_file = dst_file + "._tmp"

        # 이미 완료된 파일이 있으면 skip
        if os.path.exists(dst_file):
            continue

        if logger:
            logger.update_status(f"Downloading {src_file} → {dst_file}")

        copy_file(src_file, tmp_file)
        os.replace(tmp_file, dst_file)

def overlay(root_dir, logger=None, overwrite=False):
    os.makedirs(os.path.join(root_dir, "overlay"), exist_ok=True)
    # Download video
    download_root_dir = root_dir.replace(shared_dir, download_dir)
    download_video_dir = os.path.join(download_root_dir, "videos")
    os.makedirs(download_video_dir, exist_ok=True)
    
    serial_list = []
    for video_name in os.listdir(os.path.join(root_dir, "videos")):
        if logger:
            logger.update_status(f"Overlay : Downloading video {os.path.join(root_dir, 'video', video_name)}")
        # copy_file(os.path.join(root_dir, "videos", video_name), os.path.join(download_video_dir, video_name))
        serial_list.append(video_name.split(".")[0])
    serial_list.sort()
    
    if not overwrite:
        done = True
        for serial_name in serial_list:
            if not os.path.exists(os.path.join(root_dir, "overlay", f"{serial_name}.mp4")):
                done = False
                break
        if done:
            if logger:
                logger.update_status(f"Overlay : Done")
            return
    
    # load object
    obj_dict = {}
    if os.path.exists(os.path.join(root_dir, "obj_T.npy")):
        obj_T_dict = np.load(os.path.join(root_dir, "obj_T.npy"), allow_pickle=True).item()
        for obj_name, obj_T in obj_T_dict.items():
            obj_dict[obj_name] = {
                                    "mesh":trimesh.load(os.path.join(rsc_path, "object", obj_name, f"{obj_name}.obj")),
                                    "T":obj_T
                                  }
    # load robot
    rm = None
    arm, hand = load_robot_type(root_dir)
    if arm:
        qpos = np.load(os.path.join(root_dir, "arm", "qpos.npy"))
        if hand:
            qpos_hand = np.load(os.path.join(root_dir, "hand", "qpos.npy"))
            qpos = np.concatenate([qpos, qpos_hand], axis=1)
        rm = Robot_Module(get_robot_urdf_path(arm, hand), state=qpos)
    
    intrinsic, extrinsic = load_camparam(root_dir)
    extrinsic_list, intrinsic_list = [], []
    cammtx_list = []
    
    c2r = load_c2r(root_dir)
    
    for serial_name in serial_list:
        extmat = extrinsic[serial_name]
        extrinsic_list.append(extmat @ c2r)
        
        intmat = intrinsic[serial_name]['intrinsics_undistort'].copy()
        intrinsic_list.append(intmat)
        
        cammtx_list.append(intrinsic_list[-1] @ extrinsic_list[-1])
        
    renderer = BatchRenderer(intrinsic_list, extrinsic_list, width=2048, height=1536, device='cuda')

    # Output video stream
    os.makedirs(os.path.join(download_root_dir, "overlay"), exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    out_video = {}
    for serial_name in serial_list:
        out_path = os.path.join(download_root_dir, "overlay", f"{serial_name}_tmp.avi")
        out_video[serial_name] = cv2.VideoWriter(out_path, fourcc, 30, (2048, 1536))
        
    # Process 
    def process_frame(img_dict, video_path, fid, data):
        (obj_dict, renderer, rm, serial_list, out_video) = data
        img_tensor_dict = {img_name:torch.from_numpy(image).to('cuda', dtype=torch.float32) for img_name, image in img_dict.items()}
        color_tensor = torch.tensor((0, 255, 0), device='cuda', dtype=torch.float32)
        
        for obj_name, obj_info in obj_dict.items():
            transformed_mesh = copy.deepcopy(obj_info["mesh"])
            transformed_mesh.apply_transform(obj_info["T"][fid])
            frame, mask = project_mesh_nvdiff(transformed_mesh, renderer)
            mask = mask.bool()
            
            for i, serial_num in enumerate(serial_list):
                overlay_mask(img_tensor_dict[serial_num], mask[i,:,:,0], 0.7, color_tensor)
            
        robot_mesh_list = rm.get_mesh(fid)
        robot_mesh = robot_mesh_list[0]
        for i in range(1, len(robot_mesh_list)):
            robot_mesh += robot_mesh_list[i]
        
        frame, mask = project_mesh_nvdiff(robot_mesh, renderer)
        mask = mask.bool()
        
        for i, serial_num in enumerate(serial_list):
            overlay_mask(img_tensor_dict[serial_num], mask[i,:,:,0], 0.7, color_tensor)

        for serial_name in serial_list:
            img = torch.clamp(img_tensor_dict[serial_name], 0, 255).cpu().numpy().astype(np.uint8)
            out_video[serial_name].write(img)
            
    data = (obj_dict, renderer, rm, serial_list, out_video)
    process_video_list(download_video_dir, data, process_frame)
    
    for serial_name in serial_list:
        out_video[serial_name].release()
        change_to_h264(os.path.join(download_root_dir, "overlay", f"{serial_name}_tmp.avi"), os.path.join(download_root_dir, "overlay", f"{serial_name}.mp4"))
        copy_file(os.path.join(download_root_dir, "overlay", f"{serial_name}.mp4"), os.path.join(root_dir, "overlay", f"{serial_name}.mp4"))
    
def merge(root_dir, logger=None, overwrite=True):
    if not overwrite and os.path.exists(os.path.join(root_dir, "merged.mp4")):
        if logger:
            logger.update_status(f"Merge : Done")
        return
    
    # Download video
    video_dir = os.path.join(root_dir, "overlay")
    download_root_dir = root_dir.replace(shared_dir, download_dir)
    download_video_dir = os.path.join(download_root_dir, "overlay")
    download_files(video_dir, download_video_dir)
    
    serial_list = [serial_num.split(".")[0] for serial_num in os.listdir(video_dir)]
    serial_list.sort()
    
    
    # Output video stream
    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    out_path = os.path.join(download_root_dir, "merged_tmp.avi")
    out_video = cv2.VideoWriter(out_path, fourcc, 30, (2048, 1536))
        
    # Process 
    def process_frame(img_dict, video_path, fid, data):
        out_video = data
        frame = merge_image(img_dict)
        frame = cv2.resize(frame, (2048,1536))
        out_video.write(frame)
                       
    process_video_list(download_video_dir, out_video, process_frame)
    
    out_video.release()
    change_to_h264(os.path.join(download_root_dir, "merged_tmp.avi"), os.path.join(download_root_dir, "merged.mp4"))
    copy_file(os.path.join(download_root_dir, "merged.mp4"), os.path.join(root_dir, "merged.mp4"))