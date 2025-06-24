import threading
import queue
import numpy as np
import cv2
import json
import time
import zmq
import os
from paradex.utils.file_io import config_dir, shared_dir
from paradex.io.capture_pc.connect import git_pull, run_script

import math
from pathlib import Path
from scene import Scene
import torch
from geometry import get_visualhull_ctr

if torch.cuda.is_available():
    device = torch.device("cuda:0")
    print(f"CUDA is available. Using device: {torch.cuda.get_device_name(device)}")
else:
    device = torch.device("cpu")
    print("CUDA is not available. Using CPU.")

BOARD_COLORS = [
    (255, 0, 0), (0, 255, 0), (0, 0, 255),
    (255, 255, 0), (255, 0, 255), (0, 255, 255),
    (128, 128, 255)
]
hide_list = ['22641005','22645021','23280594','23180202','22641023','23029839','22640993']


def make_grid_img_inorder(cur_img_dict, height, width):
    
    big_img_width = width * 6
    big_img_height = height * 4
    
    canvas = np.zeros((big_img_height, big_img_width, 3), dtype=np.uint8)

    for serial_idx, serial_num in enumerate(cur_img_dict):
        row = serial_idx // 6
        col = serial_idx % 6
        x_start = col * width
        y_start = row * height
        
        img = cur_img_dict[serial_num]
        if img is not None:    
            canvas[y_start:y_start+height, x_start:x_start+width] = img
        
    return canvas


# argsparse
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--root_path', type=str, required=True)
parser.add_argument('--rgb_framerate', type=int, default=0, help='Framerate receiving RGB images from cameras, 0 for not receiving RGB images')
args = parser.parse_args()

output_dir = Path('/home/temp_id/paradex_processing/visualize/cache')
if os.path.exist(output_dir):
    os.rmdir(output_dir)
    os.makedirs(output_dir)
transl_dir = output_dir / 'transl'
transl_dir.mkdir(parents=True, exist_ok=True)
grid_dir = output_dir / 'grid'
grid_dir.mkdir(parents=True, exist_ok=True)

# make scene
obj_name = args.root_path.split("/")[-2]
mask_sub_dir = 'mask_hq/%s/%05d'%(obj_name, 0)
mask_root = Path(args.root_path)/mask_sub_dir
org_scene = Scene(root_path=Path(args.root_path), rescale_factor=1.0, mask_dir_nm=mask_sub_dir, device=device)

# === SETUP ===
pc_info = json.load(open(os.path.join(config_dir, "environment", "pc.json"), "r"))
serial_list = []
for pc in pc_info.keys():
    serial_list.extend(pc_info[pc]['cam_list'])

context = zmq.Context() # socket
socket_dict = {}
terminate_dict = {pc: False for pc in pc_info.keys()}
start_dict = {pc: False for pc in pc_info.keys()}

saved_corner_img = {serial_num :np.zeros((1536, 2048, 3), dtype=np.uint8) for serial_num in serial_list}
cur_state = {serial_num: (np.array([]), np.array([]), 0) for serial_num in serial_list}
capture_idx = 0
capture_state = {pc: False for pc in pc_info.keys()}

# filename = time.strftime("%Y%m%d_%H%M%S", time.localtime())

detection_results = {}
cur_rgb = {cam_id: None for cam_id in serial_list}

def listen_socket(pc_name, socket):
    while True:
        msg = socket.recv_string()
        if msg == "terminated":
            terminate_dict[pc_name] = True
            print(f"[{pc_name}] Terminated.")
            break
        
        elif msg == "camera_ready":
            start_dict[pc_name] = True
            print(f"[{pc_name}] Camera ready.")
            continue
        
        elif msg == "camera_error":
            print(f"[{pc_name}] Camera error.")
            terminate_dict[pc_name] = True
            continue

        try:
            data = json.loads(msg)
        except json.JSONDecodeError:
            print(f"[{pc_name}] Non-JSON message: {msg}")
            continue
        
        serial_num = data["serial_num"]

        if data.get("type") == "demo": # Get yolo output
            detections_xyxy =  np.array(data["detections.xyxy"], dtype=float)
            detections_confidence = np.array(data["detections.confidence"], dtype=float)
            
            detections_mask = None # data["detections.mask"]
            detections_bbox_center = None
            
            if detections_xyxy.size > 0:
                detections_bbox_center = detections_xyxy[:, :2] + (detections_xyxy[:, 2:] - detections_xyxy[:, :2]) / 2
                bbox = detections_xyxy[0]
                # bbox mask.
                detections_mask = np.zeros((1, 1536, 2048), dtype=bool)
                detections_mask[0, int(bbox[1]):int(bbox[3]), int(bbox[0]):int(bbox[2])] = True

            if data["frame"] not in detection_results:
                detection_results[data["frame"]] = {}
            
            detection_results[data["frame"]][serial_num] = {}
            detection_results[data["frame"]][serial_num]["mask"] = detections_mask
            detection_results[data["frame"]][serial_num]["xyxy"] = detections_xyxy
            detection_results[data["frame"]][serial_num]["confidence"] = detections_confidence
            detection_results[data["frame"]][serial_num]["frame_num"] = data["frame"]
            detection_results[data["frame"]][serial_num]["bbox_center"] = detections_bbox_center
            cur_rgb[serial_num] = np.array(data["resized_rgb"]) if args.rgb_framerate > 0 and data["frame"]%(args.rgb_framerate) == 0 and data["resized_rgb"]!='None' else None
            
            print(f"[{pc_name}] Received data for frame {data['frame']} from {serial_num}. Detections: {len(detections_xyxy)} Length: {len(detection_results[data['frame']])}")
            
        else:
            print(f"[{pc_name}] Unknown JSON type: {data.get('type')}")


def wait_for_cameras():
    while True:
        all_ready = True
        for pc_name, info in pc_info.items():
            if not start_dict[pc_name]:
                all_ready = False
                break
        if all_ready:
            print("All cameras are ready.")
            break
        time.sleep(0.1)


def main_ui_loop():
    curr_frame = 10
    cam_N = 10

    C2R = np.load(f"{shared_dir}/handeye_calibration/20250617_171318/0/C2R.npy")
    C2R = np.linalg.inv(C2R) # convert to camera coordinate system
    
    # for visualization
    import matplotlib.pyplot as plt
    plt.ion()  # interactive mode on
    fig, ax = plt.subplots()
    # imshow_o
    imshow_obj = ax.imshow(np.zeros((1536, 3072, 3)))  # Convert BGR to RGB
    # plt.pause(0.001)
    
    
    while True:
        
        if curr_frame in detection_results and len(detection_results[curr_frame]) < 24: 
            time.sleep(0.01)
            continue
        
        if curr_frame in detection_results and len(detection_results[curr_frame]) == 24:
            detect_img = 0
            for cam_id in detection_results[curr_frame]:
                if detection_results[curr_frame][cam_id]["xyxy"].size > 0:
                    detect_img += 1
            if detect_img < cam_N:
                print("Not enough detections, waiting for more...")
                time.sleep(0.01)
                curr_frame += 1
                continue
            
            detection_results_curr = detection_results[curr_frame]
            confidence_dict = {cam_id: detection_results_curr[cam_id]["confidence"][0] for cam_id in detection_results_curr if detection_results_curr[cam_id]["confidence"].size > 0 and detection_results_curr[cam_id]["confidence"][0] > 0.001 and cam_id not in hide_list}
            # print(confidence_dict)
            top_n_cams2confidence = sorted(confidence_dict.items(), key=lambda x: x[1], reverse=True)[:cam_N]
            top_n_cams = [cam_id for cam_id, confidence in top_n_cams2confidence]

            A = []
            for cam_id in top_n_cams:
                cx, cy = detection_results_curr[cam_id]["bbox_center"][0]
                P = org_scene.proj_matrix[cam_id]
                A.append(cx * P[2] - P[0])
                A.append(cy * P[2] - P[1])
            A = np.stack(A, axis=0)
            _, _, Vt = np.linalg.svd(A)
            X_h = Vt[-1]
            X = X_h[:3] / X_h[3]  # Convert from homogeneous coordinates to 3D point
            
            initial_translate = X @ C2R[:3, :3].T + C2R[:3, 3] # convert to camera coordinate system
            print(f"{initial_translate}")
            np.save(os.path.join(transl_dir, f'transl_{curr_frame}.npy'), initial_translate)
            
            if cur_rgb[serial_list[0]] is not None:
                grid_img = make_grid_img_inorder(cur_rgb, int(1536/16), int(2048/16))
                cv2.imwrite(os.path.join(grid_dir, f"grid_{curr_frame}.jpg"), grid_img)    
            
            curr_frame += 1
            time.sleep(0.01)


# Git pull and client run
pc_list = list(pc_info.keys())
git_pull("merging", pc_list)

run_script(f"python src/inference_/yolo.py --rgb_framerate {args.rgb_framerate}", pc_list)


# Connect pc with 5566 TCP
for pc_name, info in pc_info.items():
    ip = info["ip"]
    sock = context.socket(zmq.DEALER)
    sock.identity = b"server"
    sock.connect(f"tcp://{ip}:5556")
    socket_dict[pc_name] = sock

# Check socker registered
for pc_name, info in pc_info.items():
    socket_dict[pc_name].send_string("register")
    if socket_dict[pc_name].recv_string() == "registered":
        print(f"[{pc_name}] Registered.")
    
    # socket_dict[pc_name].send_string("filename:" + filename)

# Start per-socket listener
for pc_name, sock in socket_dict.items():
    threading.Thread(target=listen_socket, args=(pc_name, sock), daemon=True).start()

wait_for_cameras()
print("All cameras are ready.")
# Main UI loop
# while True:
#     time.sleep(0.1)  # Prevent busy-waiting
main_ui_loop()

# except Exception as e:
#     print(e)
#     for pc_name, sock in socket_dict.items():
#         sock.send_string("quit")
#         sock.close()
        
for pc_name, sock in socket_dict.items():
    sock.send_string("quit")
    sock.close()
        