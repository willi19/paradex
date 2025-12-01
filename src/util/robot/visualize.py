import os
import sys
import time
import argparse
from threading import Event
from xarm.wrapper import XArmAPI
import numpy as np

from paradex.simulator import IsaacSimulator
from paradex.utils.keyboard_listener import listen_keyboard
from paradex.robot.utils import get_robot_urdf_path
from paradex.utils.system import network_info
from paradex.io.robot_controller import get_arm

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
    sim.add_env("view", {"robot":{}, "robot_vis":{"robot":(args.arm, args.hand)}, "object":{}, "object_vis":{}})

    hand_dof = 12 if args.hand == "inspire" else 16
    arm = get_arm(args.arm)
    qpos = arm.get_data()["qpos"]

    action = np.concatenate([qpos, np.zeros(hand_dof)])

    sim.reset("view", {"robot":{}, "robot_vis":{"robot":action.copy()}, "object":{}})
    sim.tick()
        
    while not stop_event.is_set():   
        sim.step("view", {"robot":{}, "robot_vis":{"robot":action.copy()}, "object_vis":{}})
        sim.tick()

    # Turn off manual mode after recording
    sim.terminate()
    arm.end(set_break=True)