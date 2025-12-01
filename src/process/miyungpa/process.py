import os
import shutil
import numpy as np
import cv2

from paradex.utils.path import shared_dir
from paradex.dataset_acqusition.match_sync import get_synced_data, fill_framedrop
from paradex.image.image_dict import ImageDict
from paradex.calibration.utils import load_camparam, load_c2r
from paradex.visualization.robot import RobotModule
from paradex.robot.utils import get_robot_urdf_path

def match_sync(root_dir):
    sensor_data_path = os.path.join(root_dir, "raw")
    
    if not os.path.exists(os.path.join(sensor_data_path, "timestamps", "timestamp.npy")) or not os.path.exists(os.path.join(sensor_data_path, "timestamps", "frame_id.npy")):
        print("No timestamp folder found!")
        return
    
    frameid = np.load(os.path.join(sensor_data_path, "timestamps", "frame_id.npy"))
    pc_time = np.load(os.path.join(sensor_data_path, "timestamps", "timestamp.npy"))
    
    pc_time, frameid = fill_framedrop(frameid, pc_time)
    
    for sensor_name in ["arm", "hand"]:
        sensor_path = os.path.join(sensor_data_path, sensor_name)
        if not os.path.isdir(sensor_path):
            continue
        
        time_path = os.path.join(sensor_path, "time.npy")
        if not os.path.exists(time_path):
            continue
        os.makedirs(os.path.join(root_dir, sensor_name), exist_ok=True)
        
        sensor_timestamps = np.load(time_path)
        for data_name in os.listdir(sensor_path):
            if data_name == "time.npy":
                continue
            data_path = os.path.join(sensor_path, data_name)
            if not os.path.isfile(data_path):
                continue
            data = np.load(data_path)
                
            synced_data = get_synced_data(pc_time, data, sensor_timestamps)
            np.save(os.path.join(root_dir, sensor_name, data_name), synced_data)

def overlay(root_dir):
    video_cap = {video_name.split(".")[0] : cv2.VideoCapture(os.path.join(root_dir, "videos", video_name)) for video_name in os.listdir(os.path.join(root_dir, "videos"))}
    
    for name in list(video_cap.keys()):
        length = int(video_cap[name].get(cv2.CAP_PROP_FRAME_COUNT))
        if length == 0:
            video_cap[name].release()
            del video_cap[name]
            
    video_open = {name:True for name in video_cap.keys()}
    max_length = max([int(video_cap[name].get(cv2.CAP_PROP_FRAME_COUNT)) for name in video_cap.keys()])
    print(f"Overlaying {len(video_cap)} videos with max length {max_length}")
    os.makedirs(os.path.join(root_dir, "overlay"), exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    frame_shape = (int(video_cap[list(video_cap.keys())[0]].get(cv2.CAP_PROP_FRAME_WIDTH)), int(video_cap[list(video_cap.keys())[0]].get(cv2.CAP_PROP_FRAME_HEIGHT)))
    fps = video_cap[list(video_cap.keys())[0]].get(cv2.CAP_PROP_FPS)
    
    out_cap = {name : cv2.VideoWriter(os.path.join(root_dir, "overlay", f"{name}.avi"), fourcc, fps, frame_shape) for name in video_cap.keys()}

    intrinsics, extrinsics = load_camparam(root_dir)
    c2r = load_c2r(root_dir)
    rm = RobotModule(get_robot_urdf_path(arm_name="xarm", hand_name="allegro"))
    hand_state = np.load(os.path.join(root_dir, "hand", "joint_value.npy"))
    arm_state = np.load(os.path.join(root_dir, "arm", "position.npy"))
    
    idx = 0
    
    for idx in range(max_length):
        frame_dict = {}
        for video_name, cap in video_cap.items():
            if not video_open[video_name]:
                continue
            
            ret, frame = cap.read()
            if not ret:
                video_open[video_name] = False
                continue
            
            frame_dict[video_name] = frame
            
        if not frame_dict:
            break
        
        part_intrinsics = {name: intrinsics[name] for name in frame_dict.keys()}
        part_extrinsics = {name: extrinsics[name] for name in frame_dict.keys()}
        
        imgdict = ImageDict(frame_dict, part_intrinsics, part_extrinsics, path=None)
        rm.update_cfg(np.concatenate([arm_state[idx], hand_state[idx]]))
        robot_mesh = rm.get_robot_mesh()
        robot_mesh.apply_transform(c2r)
        
        overlayed_dict = imgdict.project_mesh(robot_mesh, color=(0,255,0))
        for video_name, out in out_cap.items():
            if video_open[video_name]:
                out.write(overlayed_dict.images[video_name])  
                
    for video_name, cap in video_cap.items():
        cap.release()
    for video_name, out in out_cap.items():
        out.release()

def process_demo(demo_path):
    match_sync(demo_path)
    overlay(demo_path)
    
#     overlay(root_dir, logger, overwrite=False)
#     print(logger)
#     logger = []
    
#     merge(root_dir, logger, overwrite=False)
#     print(logger)
#     logger = []
    
#     download_root_dir = root_dir.replace(shared_dir, download_dir)
#     if os.path.exists(download_root_dir):
#         shutil.rmtree(download_root_dir)


demo_root_path = os.path.join(shared_dir, "capture/miyungpa")
for obj_name in os.listdir(demo_root_path):
    demo_path = os.path.join(demo_root_path, obj_name)
    index_list = os.listdir(demo_path)
    for index in index_list:
        print(f"Processing {obj_name} - {index}")
        process_demo(os.path.join(demo_path, index))