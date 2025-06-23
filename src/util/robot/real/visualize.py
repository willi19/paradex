import os
import sys
import time
from paradex.simulator.isaac import simulator

sys.path.append(os.path.join(os.path.dirname(__file__), '../../..'))

from xarm.wrapper import XArmAPI
import numpy as np

# Viewer setting
obj_name = "bottle"
save_video = False
save_state = False
view_physics = False
view_replay = True
headless = False

sim = simulator(
    obj_name,
    view_physics,
    view_replay,
    headless,
    save_video,
    save_state,
    fixed=True
)


ip = "192.168.1.221"
arm = XArmAPI(ip, is_radian=True)
if arm.has_err_warn:
    arm.clean_error()
    
arm.motion_enable(enable=True)
arm.set_mode(0)
arm.set_state(state=0)

# arm.start_record_trajectory()
start_time = time.time()
while time.time() - start_time < 30:
    # _, pos_aa = arm.get_position_aa(is_radian=True)
    current_arm_angles = np.asarray(arm.get_joint_states(is_radian=True)[1][0][:6])
    action = np.zeros(22)
    action[:6] = current_arm_angles
    
    sim.step(action, action, np.eye(4, dtype=np.float32))

# arm.stop_record_trajectory()
# arm.save_record_trajectory('test.traj')

# Turn off manual mode after recording
sim.terminate()
arm.set_mode(0)
arm.set_state(0)

arm.motion_enable(enable=False)
arm.disconnect()