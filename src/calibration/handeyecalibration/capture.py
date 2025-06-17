import threading
import queue
import numpy as np
import cv2
import json
import time
import zmq
import os
from paradex.utils.file_io import config_dir, shared_dir, find_latest_directory
from paradex.io.capture_pc.connect import git_pull, run_script
import math
from xarm.wrapper import XArmAPI

class DexArmControl:
    def __init__(self, xarm_ip_address="192.168.1.221"):

        # self.allegro = AllegroController()
        self.arm = XArmAPI(xarm_ip_address, report_type="devlop")

        self.max_hand_joint_vel = 100.0 / 360.0 * 2 * math.pi  # 100 degree / sec
        self.last_xarm_command = None
        self.last_allegro_command = None

        self.arm.motion_enable(enable=False)
        self.arm.motion_enable(enable=True)
        self.arm.set_mode(0)
        self.arm.set_state(state=0)

        self.reset()

        print("init complete")

    def reset(self):
        if self.arm.has_err_warn:
            self.arm.clean_error()

        self.arm.motion_enable(enable=False)
        self.arm.motion_enable(enable=True)
        self.arm.set_mode(0)  # 0: position control, 1: servo control
        self.arm.set_state(state=0)


    def move_arm(self, target_action):
        self.arm.set_position_aa(axis_angle_pose=target_action, wait=True, is_radian=True, motion_type=1)

    def move_hand(self, allegro_angles):
        num_steps = 10
        fps = 100
        for i in range(num_steps):
            self.allegro.hand_pose(allegro_angles)
            time.sleep(1 / fps)

    def get_joint_values(self):
        is_error = 1
        while is_error != 0:
            is_error, arm_joint_states = self.arm.get_joint_states(is_radian=True)
            xarm_angles = np.array(arm_joint_states[0])

        return xarm_angles

    def quit(self):
        self.arm.motion_enable(enable=False)
        self.arm.disconnect()


def copy_calib_files(save_path):
    camparam_dir = os.path.join(shared_dir, "cam_param")
    camparam_name = find_latest_directory(camparam_dir)
    camparam_path = os.path.join(shared_dir, "cam_param", camparam_name)

    shutil.copytree(camparam_path, os.path.join(save_path, "cam_param"))
    
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
        for pc_name, in_capture in capture_state.items():
            if in_capture:
                all_captured = False
                break
        if all_captured:
            break
        time.sleep(0.1)

dex_arm = DexArmControl()

# Git pull and client run
pc_list = list(pc_info.keys())
git_pull("merging", pc_list)
# run_script("python src/calibration/handeyecalibration/client.py", pc_list)

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

try:
    for i in range(20):
        # move robot
        target_action = np.load(f"hecalib/{i+10}.npy")
        dex_arm.move_arm(target_action)
        time.sleep(0.5)
        
        xarm_angles = dex_arm.get_joint_values()
        os.makedirs(f"{shared_dir}/handeye_calibration/{filename}/{i}/image", exist_ok=True)
        np.save(f"{shared_dir}/handeye_calibration/{filename}/{i}/robot", xarm_angles[:6])
        
        for pc_name, sock in socket_dict.items():
            capture_state[pc_name] = True
            sock.send_string(f"capture:{i}")
            print(f"[{pc_name}] Start capture {i}")
        wait_for_capture()
        
except Exception as e:
    print(e)
    
    for pc_name, sock in socket_dict.items():
        sock.send_string("quit")
        sock.close()
    
    dex_arm.quit()    

dex_arm.quit()
copy_calib_files(f"/home/temp_id/shared_data/handeye_calibration/{filename}/0")
for pc_name, sock in socket_dict.items():
    sock.send_string("quit")
    sock.close()
    