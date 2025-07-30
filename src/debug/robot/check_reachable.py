import numpy as np
import os
from threading import Event
import time
import tqdm

from paradex.robot.robot_wrapper import RobotWrapper
from paradex.utils.file_io import rsc_path
from paradex.simulator import IsaacSimulator
from paradex.utils.keyboard_listener import listen_keyboard
from paradex.geometry.coordinate import DEVICE2WRIST

arm_name = "xarm"
hand_name = "inspire"
robot = RobotWrapper(os.path.join(rsc_path, "robot", f"{arm_name}_{hand_name}.urdf"))

wrist_pos_list = [
    np.array([[0.0,  0.0, 1.0],
        [1.0, 0.0,  0.0],
        [ 0.0, 1.0,  0.0]]), # Front view
    np.array([[0.0,  1.0, 0.0], # Looking left
        [0.0, 0.0,  1.0],
        [ 1.0,  0.0,  0.0]]),
    np.array([[0.0,  -1.0, 0.0],
        [0.0, 0.0,  -1.0],
        [ 1.0,  0.0,  0.0]]), # Looking Right
    np.array([[1.0,  0.0, 0.0], # Looking up
        [0.0, 1.0,  0.0],
        [ 0.0, 0.0,  1.0]]),
    np.array([[0.0,  1.0, 0.0], # Looking Down
        [1.0, 0.0,  0.0],
        [ 0.0, 0.0,  -1.0]])
]

result = {}

for d in tqdm.tqdm(range(len(wrist_pos_list))):
    result[d] = {"pos":wrist_pos_list[d].copy()}
    for tx in tqdm.tqdm(np.arange(0.2, 0.7, 0.05)):
        for ty in tqdm.tqdm(np.arange(-0.7, 0.7, 0.05)):
            # for theta in np.arange(0, np.pi * 2, 0.01):
            for tz in tqdm.tqdm(np.arange(0.1, 0.7, 0.05)):
                wrist_position = np.eye(4)
                wrist_position[0, 3] = tx
                wrist_position[1, 3] = ty
                wrist_position[2, 3] = tz
                wrist_position[:3, :3] = wrist_pos_list[d].copy()
                # wrist_position[:3, :3] = wrist_position[:3, :3] @ np.array([[np.cos(theta), np.sin(theta) , 0],[-np.sin(theta), np.cos(theta), 0],[0, 0, 1]])
                action, succ = robot.solve_ik(wrist_position, "link6")
                result[d][(tx,ty,tz)] = [action, succ]

np.save("data/ik.npy", result)
                
