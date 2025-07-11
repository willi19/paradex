import os
import sys
import time
from paradex.simulator import IsaacSimulator
import argparse
from paradex.utils.keyboard_listener import listen_keyboard
from threading import Event

sys.path.append(os.path.join(os.path.dirname(__file__), '../../..'))

from xarm.wrapper import XArmAPI
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("--arm", type=str, required=True)
parser.add_argument("--hand", type=str, required=True)
args = parser.parse_args()

if __name__ == "__main__":
    stop_event = Event()
    listen_keyboard({"q":stop_event})    
    
    sim = IsaacSimulator(
        headless=False
    )

    sim.load_robot_asset(args.arm, args.hand)
    sim.add_env({"robot":{}, "robot_vis":{"robot":(args.arm, args.hand)}, "object":{}, "object_vis":{}})

    hand_dof = 12 if args.hand == "inspire" else 16
    
    ip = "192.168.1.221"
    arm = XArmAPI(ip, is_radian=True)
    if arm.has_err_warn:
        arm.clean_error()
        
    arm.motion_enable(enable=True)
    arm.set_mode(0)
    arm.set_state(state=0)

    current_arm_angles = np.asarray(arm.get_joint_states(is_radian=True)[1][0][:6])
    action = np.zeros(6+hand_dof)
    action[:6] = current_arm_angles
    
    sim.reset(0, {"robot":{}, "robot_vis":{"robot":action.copy()}, "object":{}})
    sim.tick()
        
    while not stop_event.is_set():
       
        current_arm_angles = np.asarray(arm.get_joint_states(is_radian=True)[1][0][:6])
        action = np.zeros(6+hand_dof)
        action[:6] = current_arm_angles
        
        sim.step(0, {"robot":{}, "robot_vis":{"robot":action.copy()}, "object_vis":{}})
        sim.tick()

    # arm.stop_record_trajectory()
    # arm.save_record_trajectory('test.traj')

    # Turn off manual mode after recording
    sim.terminate()
    arm.set_mode(0)
    arm.set_state(0)

    arm.motion_enable(enable=False)
    arm.disconnect()