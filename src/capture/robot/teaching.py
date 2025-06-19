import os
import sys
import time
import threading

sys.path.append(os.path.join(os.path.dirname(__file__), '../../..'))

from xarm.wrapper import XArmAPI
import numpy as np

stop_event = threading.Event()
save_event = threading.Event()
save_dir = 'demo_asdf'
os.makedirs(save_dir, exist_ok=True)

def listen_keyboard():
    print("Press 'c' to capture, 'q' to quit.")
    while not stop_event.is_set():
        key = input().strip().lower()
        if key == 'q':
            stop_event.set()
        elif key == 'c':
            save_event.set()
            
input_thread = threading.Thread(target=listen_keyboard, daemon=True)
input_thread.start()
     
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
        if save_event.is_set():
            _, pos_aa = arm.get_position_aa(is_radian=True)
            np.save(os.path.join(save_dir, f'{idx}.npy'), pos_aa)
            print(f"Saved pose {idx}: {pos_aa}")
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
