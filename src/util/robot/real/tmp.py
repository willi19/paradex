import os
import sys
from threading import Event
from paradex.robot import RobotWrapper
from scipy.spatial.transform import Rotation as R

sys.path.append(os.path.join(os.path.dirname(__file__), '../../..'))

from xarm.wrapper import XArmAPI
import numpy as np

if __name__ == "__main__":
    robot = RobotWrapper(os.path.join("rsc","robot","xarm.urdf"))
    
    ip = "192.168.1.221"
    arm = XArmAPI(ip, is_radian=True)
    if arm.has_err_warn:
        arm.clean_error()
        
    arm.motion_enable(enable=True)
    arm.set_mode(0)
    arm.set_state(state=0)

    current_arm_angles = np.asarray(arm.get_joint_states(is_radian=True)[1][0][:6])
    robot.compute_forward_kinematics(current_arm_angles)
    link6_id = robot.get_link_index("link6")
    
    position = arm.get_position(is_radian=True)

    print(position[1][3:], "current")
    
    arm.set_mode(0)
    arm.set_state(0)

    arm.motion_enable(enable=False)
    arm.disconnect()