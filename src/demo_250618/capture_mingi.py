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

# def draw_charuco_corners_custom(image, corners, color=(0, 255, 255), radius=4, thickness=2, ids=None):
#     for i in range(len(corners)):
#         corner = tuple(int(x) for x in corners[i][0])
#         cv2.circle(image, corner, radius, color, thickness)
#         if ids is not None:
#             cv2.putText(image, str(int(ids[i])), (corner[0] + 5, corner[1] - 5),
#                         cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1, lineType=cv2.LINE_AA)

capture_state = {}

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

        elif msg == "save_finish":
            capture_state[pc_name] = False
            print(f"[{pc_name}] Save finished.")
            continue

        try:
            data = json.loads(msg)
        except json.JSONDecodeError:
            print(f"[{pc_name}] Non-JSON message: {msg}")
            continue
        
        serial_num = data["serial_num"]
        if data.get("type") == "charuco":
            result = data["detect_result"]
            # corners = np.array(result["checkerCorner"], dtype=np.float32)
            # ids = np.array(result["checkerIDs"], dtype=np.int32).reshape(-1, 1)
            frame = data["frame"]
            cur_state[serial_num] = frame
            # cur_state[serial_num] = (corners, ids, frame)

            # if result["save"]:
            #     draw_charuco_corners_custom(saved_corner_img[serial_num], corners, BOARD_COLORS[2], 5, -1, ids)

        else:
            print(f"[{pc_name}] Unknown JSON type: {data.get('type')}")

def main_loop():
    while True:
        current_idx = 0
        cur_cnt = 0
        
        for serial_num in serial_list:
            if serial_num in cur_state:
                if cur_state[serial_num] >= current_idx:
                    current_idx = cur_state[serial_num]
                    cur_cnt += 1
        if cur_cnt == len(serial_list):
            # DO YOLO
            # plot grid image
        current_idx += 1
    
# def main_ui_loop():
#     num_images = len(serial_list)
#     grid_cols = math.ceil(math.sqrt(num_images))
#     grid_rows = math.ceil(num_images / grid_cols)
#     border_px = 20

#     new_W = 2048 // grid_rows
#     new_H = 1536 // grid_rows

#     while True:
#         all_disconnected = True
#         for pc_name, terminated in terminate_dict.items():
#             if not terminated:
#                 all_disconnected = False
#         if all_disconnected:
#             break
        
#         grid_image = np.ones((1536+border_px*(grid_rows-1), (2048//grid_rows)*grid_cols+border_px*(grid_cols-1), 3), dtype=np.uint8) * 255
#         for idx, serial_num in enumerate(serial_list):
#             img = saved_corner_img[serial_num].copy()
#             corners, ids, frame = cur_state[serial_num]
#             if corners.shape[0] > 0:
#                 draw_charuco_corners_custom(img, corners, BOARD_COLORS[1], 5, -1, ids)
#             img = cv2.putText(img, f"{serial_num} {frame}", (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 6, (255, 255, 0), 12)

#             resized_img = cv2.resize(img, (new_W, new_H))
            
#             r_idx = idx // grid_cols
#             c_idx = idx % grid_cols

#             r_start = r_idx * (new_H + border_px)
#             c_start = c_idx * (new_W + border_px)
#             grid_image[r_start:r_start+resized_img.shape[0], c_start:c_start+resized_img.shape[1]] = resized_img

#         grid_image = cv2.resize(grid_image, (int(2048//1.5), int(1536//1.5)))
#         cv2.imshow("Grid", grid_image)
#         key = cv2.waitKey(1)
#         if key == ord('q'):
#             print("[Server] Quitting...")
#             for socket in socket_dict.values():
#                 socket.send_string("quit")
#             break
#         elif key == ord('c'):
#             print("[Server] Sending capture command.")
#             send_capture = True
#             for pc in pc_info.keys():
#                 if capture_state[pc]:
#                     send_capture = False
#                     break
#             if send_capture:
#                 global capture_idx, filename
#                 os.makedirs(os.path.join(shared_dir, "extrinsic", filename, str(capture_idx)), exist_ok=True)
#                 for pc, socket in socket_dict.items():
#                     socket.send_string(f"capture:{capture_idx}")
#                     capture_state[pc] = True
#                 capture_idx += 1

def wait_for_camera_ready():
    while True:
        all_ready = True
        for pc_name, ready in start_dict.items():
            if not ready:
                all_ready = False
                break
        if all_ready:
            break
        time.sleep(0.1)

# Git pull and client run
pc_list = list(pc_info.keys())
git_pull("merging", pc_list)
run_script("python src/demo_250618/client_mingi.py", pc_list)

try:
    
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
    wait_for_camera_ready()
    # Main UI loop
    main_loop()

except Exception as e:
    print(e)
    for pc_name, sock in socket_dict.items():
        sock.send_string("quit")
        sock.close()