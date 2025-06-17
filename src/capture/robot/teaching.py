import os
import sys
import time

sys.path.append(os.path.join(os.path.dirname(__file__), '../../..'))

from xarm.wrapper import XArmAPI
import numpy as np

ip = "192.168.1.221"
arm = XArmAPI(ip, is_radian=True)
if arm.has_err_warn:
    arm.clean_error()
    
arm.motion_enable(enable=True)
arm.set_mode(0)
arm.set_state(state=0)

# # Turn on manual mode before recording
arm.set_mode(2)
arm.set_state(0)

# arm.start_record_trajectory()

# # Analog recording process, here with delay instead
os.makedirs('robot_traj', exist_ok=True)
start_time = time.time()
idx = 0
while time.time() - start_time < 30:
    _, pos_aa = arm.get_position_aa(is_radian=True)
    np.save(f'robot_traj/{idx}.npy', pos_aa)
    idx += 1
    time.sleep(1)
# arm.stop_record_trajectory()
# arm.save_record_trajectory('test.traj')

# Turn off manual mode after recording
arm.set_mode(0)
arm.set_state(0)

arm.motion_enable(enable=False)
arm.disconnect()