import numpy as np
import os
import tqdm

from paradex.robot.robot_wrapper_updating import RobotWrapper
from paradex.utils.file_io import rsc_path

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


for d in tqdm.tqdm(range(1, len(wrist_pos_list))):
    result = {"pos":wrist_pos_list[d].copy()}
    for tx in tqdm.tqdm(np.arange(0.1, 0.7, 0.02)):
        for ty in tqdm.tqdm(np.arange(-0.7, 0.7, 0.02)):
            # for theta in np.arange(0, np.pi * 2, 0.01):
            for tz in np.arange(0.1, 0.7, 0.02):
                if tx ** 2 + ty ** 2 + tz ** 2 > 0.8 ** 2:
                    continue
                wrist_position = np.eye(4)
                wrist_position[0, 3] = tx
                wrist_position[1, 3] = ty
                wrist_position[2, 3] = tz
                wrist_position[:3, :3] = wrist_pos_list[d].copy()
                # wrist_position[:3, :3] = wrist_position[:3, :3] @ np.array([[np.cos(theta), np.sin(theta) , 0],[-np.sin(theta), np.cos(theta), 0],[0, 0, 1]])
                action, succ = robot.solve_ik(wrist_position, "link6")
                recom_succ = True

                robot.compute_forward_kinematics(action)
                link6_pos = robot.get_link_pose(robot.get_link_index("link6"))

                if np.linalg.norm((link6_pos - wrist_position)[:,3]) > 0.01:
                    recom_succ = False
                
                diff = link6_pos[:3,:3] @ np.linalg.inv(wrist_position[:3,:3]) - np.eye(3)
                if np.linalg.norm(diff) > 0.015:
                    recom_succ = False
                result[(tx,ty,tz)] = [action, succ, recom_succ]

    np.save(f"data/ik_{d}.npy", result)
                
