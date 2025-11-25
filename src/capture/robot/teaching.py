import os
import time
from threading import Event
import argparse

from xarm.wrapper import XArmAPI
import numpy as np

from paradex.utils.keyboard_listener import listen_keyboard
from paradex.geometry.conversion import aa2mtx

stop_event = Event()
save_event = Event()
listen_keyboard({'q':stop_event, 'c':save_event})

parser = argparse.ArgumentParser()
parser.add_argument('--save_path')
args = parser.parse_args()

if args.save_path is not None:
    os.makedirs(args.save_path, exist_ok=True)
     
ip = "192.168.1.221"
arm = XArmAPI(ip, is_radian=True)
if arm.has_err_warn:
    arm.clean_error()
    
arm.motion_enable(enable=True)
arm.set_mode(0)
arm.set_state(state=0)

# Turn on manual mode before recording
arm.set_mode(2)
arm.set_state(0)

idx = 0
try:
    while not stop_event.is_set():
        if save_event.is_set() and not args.save_path is None:
            _, pos_aa = arm.get_position_aa(is_radian=True)
            _, qpos = arm.get_joint_states()
            qpos = qpos[0][:6]
            
            pos_aa = np.array(pos_aa)
            wrist_pos = aa2mtx(pos_aa)
            
            np.save(os.path.join(args.save_path, f'{idx}_qpos.npy'), qpos)
            np.save(os.path.join(args.save_path, f'{idx}_aa.npy'), wrist_pos)
            
            print(f"Saved pose {idx}: {wrist_pos}")
            idx += 1
            save_event.clear()
        time.sleep(0.1)
        
except KeyboardInterrupt:
    print("Interrupted by user.")

# Turn off manual mode after recording
arm.set_mode(0)
arm.set_state(0)
arm.motion_enable(enable=False)
arm.disconnect()
print("Recording session ended.")
