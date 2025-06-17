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

capture_idx = 0
capture_state = {pc: False for pc in pc_info.keys()}

filename = time.strftime("%Y%m%d_%H%M%S", time.localtime())

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
        
def wait_for_capture():
    while True:
        all_captured = True
        for pc_name, captured in capture_state.items():
            if not captured:
                all_captured = False
                break
        if all_captured:
            break
        time.sleep(0.1)


# Git pull and client run
pc_list = list(pc_info.keys())
git_pull("merging", pc_list)
# run_script("python src/calibration/extrinsic/client.py", pc_list)

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

for i in range(5):
    for pc_name, sock in socket_dict.items():
        sock.send_string(f"capture:{i}")
        print(f"[{pc_name}] Start capture {i+1}/5")
    wait_for_capture()
    time.sleep(0.5)  # Wait for cameras to stabilize
        

# except Exception as e:
#     print(e)
#     for pc_name, sock in socket_dict.items():
#         sock.send_string("quit")
#         sock.close()