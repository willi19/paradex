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

BOARD_COLORS = [
    (255, 0, 0), (0, 255, 0), (0, 0, 255),
    (255, 255, 0), (255, 0, 255), (0, 255, 255),
    (128, 128, 255)
]

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

def draw_charuco_corners_custom(image, corners, color=(0, 255, 255), radius=4, thickness=2, ids=None):
    for i in range(len(corners)):
        corner = tuple(int(x) for x in corners[i][0])
        cv2.circle(image, corner, radius, color, thickness)
        if ids is not None:
            cv2.putText(image, str(int(ids[i])), (corner[0] + 5, corner[1] - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1, lineType=cv2.LINE_AA)

def main_ui_loop():
    num_images = len(serial_list)
    grid_cols = math.ceil(math.sqrt(num_images))
    grid_rows = math.ceil(num_images / grid_cols)
    border_px = 20

    new_W = 2048 // grid_rows
    new_H = 1536 // grid_rows

    while True:
        all_disconnected = True
        for pc_name, terminated in terminate_dict.items():
            if not terminated:
                all_disconnected = False
        if all_disconnected:
            break
        
        grid_image = np.ones((1536+border_px*(grid_rows-1), (2048//grid_rows)*grid_cols+border_px*(grid_cols-1), 3), dtype=np.uint8) * 255
        for idx, serial_num in enumerate(serial_list):
            img = saved_corner_img[serial_num].copy()
            corners, ids, frame = cur_state[serial_num]
            if corners.shape[0] > 0:
                draw_charuco_corners_custom(img, corners, BOARD_COLORS[1], 5, -1, ids)
            img = cv2.putText(img, f"{serial_num} {frame}", (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 6, (255, 255, 0), 12)

            resized_img = cv2.resize(img, (new_W, new_H))
            
            r_idx = idx // grid_cols
            c_idx = idx % grid_cols

            r_start = r_idx * (new_H + border_px)
            c_start = c_idx * (new_W + border_px)
            grid_image[r_start:r_start+resized_img.shape[0], c_start:c_start+resized_img.shape[1]] = resized_img

        grid_image = cv2.resize(grid_image, (int(2048//1.5), int(1536//1.5)))
        cv2.imshow("Grid", grid_image)
        key = cv2.waitKey(1)
        if key == ord('q'):
            print("[Server] Quitting...")
            for socket in socket_dict.values():
                socket.send_string("quit")
            break
        elif key == ord('c'):
            print("[Server] Sending capture command.")
            send_capture = True
            for pc in pc_info.keys():
                if capture_state[pc]:
                    send_capture = False
                    break
            if send_capture:
                global capture_idx, filename
                os.makedirs(os.path.join(shared_dir, "extrinsic", filename, str(capture_idx)), exist_ok=True)
                for pc, socket in socket_dict.items():
                    socket.send_string(f"capture:{capture_idx}")
                    capture_state[pc] = True
                capture_idx += 1

num_images = len(serial_list)
grid_cols = math.ceil(math.sqrt(num_images))
grid_rows = math.ceil(num_images / grid_cols)
border_px = 20

new_W = 2048 // grid_rows
new_H = 1536 // grid_rows


keypoint_path = os.path.join(f"{shared_dir}/extrinsic")
keypoint_file = os.listdir(keypoint_path)[-1]
keypoint_list = os.listdir(os.path.join(keypoint_path, keypoint_file))

output_path = f"extrinsic_demo.mp4"
# os.makedirs("extrinsic_demo")

frame_size = ((2048//grid_rows)*grid_cols+border_px*(grid_cols-1), 1536+border_px*(grid_rows-1))  # (W, H)
fps = 5  # 10 frames per second

# fourcc: 코덱 설정 (mp4는 'mp4v' 또는 'avc1' 추천)
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
video_writer = cv2.VideoWriter(output_path, fourcc, fps, frame_size)

img_dict = {serial_name :np.zeros((1536, 2048, 3), dtype=np.uint8) for serial_name in serial_list}

for fidx in keypoint_list:
    keypoint_file_list = os.listdir(os.path.join(keypoint_path, keypoint_file, fidx))
    for keypoint_name in keypoint_file_list:
        if "cor" not in keypoint_name:
            continue
        serial_num = keypoint_name.split("_")[0]
        
        corners = np.load(os.path.join(keypoint_path, keypoint_file, fidx, keypoint_name))
        ids = np.load(os.path.join(keypoint_path, keypoint_file, fidx, f"{serial_num}_id.npy"))
        if corners.shape[0] == 0:
            continue
        
        draw_charuco_corners_custom(img_dict[serial_num], corners, BOARD_COLORS[2], 5, -1, ids)

    grid_image = np.ones((1536+border_px*(grid_rows-1), (2048//grid_rows)*grid_cols+border_px*(grid_cols-1), 3), dtype=np.uint8) * 255
    for idx, serial_num in enumerate(serial_list):
        img = img_dict[serial_num].copy()
        img = cv2.putText(img, f"{serial_num} {fidx}", (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 6, (255, 255, 0), 12)

        resized_img = cv2.resize(img, (new_W, new_H))
        
        r_idx = idx // grid_cols
        c_idx = idx % grid_cols

        r_start = r_idx * (new_H + border_px)
        c_start = c_idx * (new_W + border_px)
        grid_image[r_start:r_start+resized_img.shape[0], c_start:c_start+resized_img.shape[1]] = resized_img

    video_writer.write(grid_image)
video_writer.release()