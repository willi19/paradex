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

BOARD_COLORS = [
    (255, 0, 0), (0, 255, 0), (0, 0, 255),
    (255, 255, 0), (255, 0, 255), (0, 255, 255),
    (128, 128, 255)
]
hide_list = ['22641005','22645021','23280594','23180202','22641023','23029839','22640993']
device = torch.device("cuda:0")

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--root_path', type=str, required=True)

args = parser.parse_args()

obj_name = args.root_path.split("/")[-2]
mask_sub_dir = 'mask_hq/%s/%05d'%(obj_name, 0)
mask_root = Path(args.root_path)/mask_sub_dir
org_scene = Scene(root_path=Path(args.root_path), rescale_factor=1.0, mask_dir_nm=mask_sub_dir, device=device)

# === SETUP ===
pc_info = json.load(open(os.path.join(config_dir, "environment", "pc.json"), "r"))
serial_list = []
for pc in pc_info.keys():
    serial_list.extend(pc_info[pc]['cam_list'])

context = zmq.Context()
socket_dict = {}
terminate_dict = {pc: False for pc in pc_info.keys()}
start_dict = {pc: False for pc in pc_info.keys()}

saved_corner_img = {serial_num:np.zeros((1536, 2048, 3), dtype=np.uint8) for serial_num in serial_list}
cur_state = {serial_num:(np.array([]), np.array([]), 0) for serial_num in serial_list}
capture_idx = 0
capture_state = {pc: False for pc in pc_info.keys()}

filename = time.strftime("%Y%m%d_%H%M%S", time.localtime())

detection_results = {}

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

        if data.get("type") == "demo":
            detections_mask = data["detections.mask"]
            detections_xyxy = data["detections.xyxy"]
            detections_confidence = data["detections.confidence"]

            if data["frame"] not in detection_results:
                detection_results[data["frame"]] = {}
            # print(serial_num, data["frame"])
            
            
            detection_results[data["frame"]][serial_num] = {}
            detection_results[data["frame"]][serial_num]["mask"] = np.array(detections_mask, dtype=bool)
            detection_results[data["frame"]][serial_num]["xyxy"] = np.array(detections_xyxy, dtype=float)
            detection_results[data["frame"]][serial_num]["confidence"] = np.array(detections_confidence)
            detection_results[data["frame"]][serial_num]["frame_num"] = data["frame"]
            
            # if detection_results[serial_num]["mask"].size > 0:
            #     print(data["frame"])
            #     print(detection_results[serial_num]["mask"])
            #     print(detection_results[serial_num]["xyxy"])
            #     print(detection_results[serial_num]["confidence"])

        else:
            print(f"[{pc_name}] Unknown JSON type: {data.get('type')}")

def main_ui_loop():
    curr_frame = 3
    while True:
        
        if curr_frame in detection_results: print(len(detection_results[curr_frame]))
        if curr_frame in detection_results and len(detection_results[curr_frame]) > 12:
            
            detection_results_curr = detection_results[curr_frame]
            confidence_dict = {cam_id: detection_results_curr[cam_id]["confidence"][0] for cam_id in detection_results_curr if detection_results_curr[cam_id]["confidence"].size > 0 and detection_results_curr[cam_id]["confidence"][0] > 0.001 and cam_id not in hide_list}
            # print(confidence_dict)
            cam_N = 10
            top_n_cams2confidence = sorted(confidence_dict.items(), key=lambda x: x[1], reverse=True)[:cam_N]
            top_n_cams = [cam_id for cam_id, confidence in top_n_cams2confidence]

            mask_dict_org = {cam_id: detection_results[cam_id]["mask"][0] for cam_id in top_n_cams}
            # print(mask_dict_org)

            in_mask_points, initial_translate = get_visualhull_ctr(org_scene, mask_dict=mask_dict_org) # Set Initial translation as center of visual hull

        

        # frame_num = detection_results[detection_results[0]]
            print(f"{initial_translate}")
            time.sleep(0.01)


    # num_images = len(serial_list)
    # grid_cols = math.ceil(math.sqrt(num_images))
    # grid_rows = math.ceil(num_images / grid_cols)
    # border_px = 20
    # new_W = 2048 // grid_rows
    # new_H = 1536 // grid_rows

    # while True:
    #     all_disconnected = True
    #     for pc_name, terminated in terminate_dict.items():
    #         if not terminated:
    #             all_disconnected = False
    #     if all_disconnected:
    #         break
        
    #     grid_image = np.ones((1536+border_px*(grid_rows-1), (2048//grid_rows)*grid_cols+border_px*(grid_cols-1), 3), dtype=np.uint8) * 255
    #     for idx, serial_num in enumerate(serial_list):
    #         img = saved_corner_img[serial_num].copy()
    #         corners, ids, frame = cur_state[serial_num]
    #         # if corners.shape[0] > 0:
    #         #     draw_charuco_corners_custom(img, corners, BOARD_COLORS[1], 5, -1, ids)
    #         img = cv2.putText(img, f"{serial_num} {frame}", (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 6, (255, 255, 0), 12)

    #         resized_img = cv2.resize(img, (new_W, new_H))
            
    #         r_idx = idx // grid_cols
    #         c_idx = idx % grid_cols

    #         r_start = r_idx * (new_H + border_px)
    #         c_start = c_idx * (new_W + border_px)
    #         grid_image[r_start:r_start+resized_img.shape[0], c_start:c_start+resized_img.shape[1]] = resized_img


    #     grid_image = cv2.resize(grid_image, (int(2048//1.5), int(1536//1.5)))
    #     cv2.imshow("Grid", grid_image)
        # key = cv2.waitKey(1)
        # if key == ord('q'):
        #     print("[Server] Quitting...")
        #     for socket in socket_dict.values():
        #         socket.send_string("quit")
        #     break
        # elif key == ord('c'):
        #     print("[Server] Sending capture command.")
        #     send_capture = True
        #     for pc in pc_info.keys():
        #         if capture_state[pc]:
        #             send_capture = False
        #             break
        #     if send_capture:
        #         global capture_idx, filename
        #         os.makedirs(os.path.join(shared_dir, "extrinsic", filename, str(capture_idx)), exist_ok=True)
        #         for pc, socket in socket_dict.items():
        #             socket.send_string(f"capture:{capture_idx}")
        #             capture_state[pc] = True
        #         capture_idx += 1

# Git pull and client run
pc_list = list(pc_info.keys())
git_pull("merging", pc_list)
# run_script("python src/demo_250618/client.py", pc_list)

# try:
for pc_name, info in pc_info.items():
    ip = info["ip"]
    sock = context.socket(zmq.DEALER)
    sock.identity = b"server"
    sock.connect(f"tcp://{ip}:5556")
    socket_dict[pc_name] = sock

for pc_name, info in pc_info.items():
    socket_dict[pc_name].send_string("register")
    if socket_dict[pc_name].recv_string() == "registered":
        print(f"[{pc_name}] Registered.")
    
    socket_dict[pc_name].send_string("filename:" + filename)

# Start per-socket listener
for pc_name, sock in socket_dict.items():
    threading.Thread(target=listen_socket, args=(pc_name, sock), daemon=True).start()

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
        