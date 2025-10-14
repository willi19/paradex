import numpy as np
import pickle
import os
import open3d as o3d
from curobo.wrap.reacher.motion_gen import MotionGen, MotionGenConfig, MotionGenPlanConfig, PoseCostMetric
from curobo.geom.types import WorldConfig
from curobo.types.base import TensorDeviceType
import trimesh
from scipy.spatial.transform import Rotation as R
# Pick Target 

from paradex.utils.file_io import find_latest_directory, shared_dir, home_path, load_yaml, load_latest_C2R, get_robot_urdf_path, rsc_path
from paradex.robot.curobo import CuroboPlanner
from paradex.visualization.visualizer.viser import ViserViewer
from paradex.inference.util import get_linear_path
from paradex.robot.robot_wrapper import RobotWrapper
from paradex.robot.mimic_joint import parse_inspire

def merge_detection():
    obj_6d_path = os.path.join(shared_dir, 'object_6d', 'data', 'obj_output')
    latest_dir = find_latest_directory(obj_6d_path)
    source_dir = '20251014-211030'

    with open(os.path.join(obj_6d_path, latest_dir, 'obj_T.pkl'), 'rb') as f:
        obj_output = pickle.load(f)
    
    with open(os.path.join(obj_6d_path, source_dir, 'obj_T.pkl'), 'rb') as f:
        source_output = pickle.load(f)
    
    source_name = ["red_19", "brown_3", "brown_2"]
    obj_idx = 0
    for obj_type, obj_list in source_output.items():
        obj_type_idx = max([int(obj_name) for obj_name in obj_output[obj_type].keys()])+1
        
        color = obj_type.split('_')[0]  # brown_ramen_1 -> brown
        if color not in ["brown", "red", "yellow"]:
            continue

        for obj_name, obj_se3 in obj_list.items():
            if f"{color}_{obj_idx}" in source_name:
                obj_output[obj_type][obj_type_idx] = obj_se3
                obj_type_idx += 1
            obj_idx += 1
    pickle.dump(obj_output, open(os.path.join(obj_6d_path, latest_dir, 'obj_T_tmp.pkl'), 'wb'))
merge_detection()