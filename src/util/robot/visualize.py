import os
import sys
import time
import argparse
from threading import Event
import numpy as np

from paradex.robot.utils import get_robot_urdf_path
from paradex.utils.system import network_info
from paradex.io.robot_controller import get_arm
from paradex.visualization.visualizer.viser import ViserViewer

parser = argparse.ArgumentParser()
parser.add_argument("--arm", type=str, required=True)
parser.add_argument("--hand", type=str, required=True)
args = parser.parse_args()

if __name__ == "__main__":
    stop_event = Event()
    vis = ViserViewer()
    hand_dof = 6 if args.hand == "inspire" else 16
    arm = get_arm(args.arm)
    qpos = arm.get_data()["qpos"]
    print(qpos)
    action = np.concatenate([qpos, np.zeros(hand_dof)])

    vis.add_robot("robot", get_robot_urdf_path(args.arm, args.hand))
    vis.add_traj("traj", {"robot":np.array([action])})
    arm.end(set_break=True)
    vis.start_viewer()