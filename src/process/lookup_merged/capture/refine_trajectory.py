import numpy as np
import os
import time
from scipy.spatial.transform import Rotation

from paradex.utils.file_io import shared_dir, download_dir, eef_calib_path, load_latest_eef, get_robot_urdf_path
from paradex.inference.lookup_table import refine_trajectory
from paradex.simulator import IsaacSimulator
from paradex.robot.mimic_joint import parse_inspire
from paradex.geometry.coordinate import DEVICE2WRIST
from paradex.robot.robot_wrapper import RobotWrapper
from paradex.process.lookup import normalize

root_dir = os.path.join(shared_dir, "capture", "lookup")
obj_list = os.listdir(root_dir)
for obj_name in obj_list:
    index_list = os.listdir(os.path.join(root_dir, obj_name))
    for index in index_list:
        demo_path = os.path.join(root_dir, obj_name, index)
        
        for type in ["pick", "place"]:
            action = np.load(os.path.join(demo_path, f"{type}_action.npy"))
            state = np.load(os.path.join(demo_path, f"{type}_state.npy"))
            hand_action = np.load(os.path.join(demo_path, f"{type}_hand.npy"))
            
            action, state, hand_action = refine_trajectory(action, state, hand_action)
            
            np.save(os.path.join(demo_path, f"refined_{type}_action.npy"), action)
            np.save(os.path.join(demo_path, f"refined_{type}_state.npy"), state)
            np.save(os.path.join(demo_path, f"refined_{type}_hand.npy"), hand_action)