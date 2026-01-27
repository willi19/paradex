from threading import Event
import time
import argparse
import os
import datetime
import numpy as np

from paradex.io.camera_system.remote_camera_controller import remote_camera_controller
from paradex.io.robot_controller import get_arm, get_hand

from paradex.utils.keyboard_listener import listen_keyboard
from paradex.utils.path import shared_dir

from paradex.calibration.utils import save_current_camparam, save_current_C2R

parser = argparse.ArgumentParser()
parser.add_argument('--save_path', required=True)
parser.add_argument("--arm", type=str, default="xarm")
parser.add_argument("--hand", type=str, default="inspire")

args = parser.parse_args()

rcc = remote_camera_controller("image_main.py")

# stop_event = Event()
# save_event = Event()

# listen_keyboard({"c":save_event, "q":stop_event})

save_dir = os.path.join(shared_dir, args.save_path)

arm_ctrl = get_arm(args.arm)
hand_ctrl = get_hand(args.hand, ip=True)

# print("ready to capture images. Press 'c' to capture, 'q' to quit.")

try:
    # date_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    for pos in range(0, 1001, 1):
        hand_action = np.array([pos, 1000, 1000, 1000, 1000, 1000])
        hand_ctrl.move(hand_action)
        time.sleep(0.5)
        hand_state = hand_ctrl.get_qpos()
        arm_state = arm_ctrl.get_data()
        state_str = str(hand_state[0])
        if os.path.exists(os.path.join(shared_dir, args.save_path, state_str)):
            print(f"Skip capture, already exists: {args.save_path}/{state_str}")
            continue
        save_current_camparam(os.path.join(shared_dir, args.save_path, state_str))
        save_current_C2R(os.path.join(shared_dir, args.save_path, state_str))
        print(f"Capturing image to {args.save_path}/{state_str}")
        raw_dir = os.path.join(shared_dir, args.save_path, state_str, "raw")
        hand_dir = os.path.join(raw_dir, "hand")
        arm_dir = os.path.join(raw_dir, "arm")
        os.makedirs(hand_dir, exist_ok=True)
        os.makedirs(arm_dir, exist_ok=True)
        np.save(os.path.join(hand_dir, "state.npy"), hand_state)
        np.save(os.path.join(arm_dir, "state.npy"), arm_state.get("qpos"))
        rcc.start("image", False, f'shared_data/{args.save_path}/{state_str}/raw')
        rcc.stop()
        
finally:
    rcc.end()
