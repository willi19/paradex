# Third Party
import torch
# cuRobo
from curobo.geom.sdf.world import CollisionCheckerType
from curobo.types.base import TensorDeviceType
from curobo.wrap.model.robot_world import RobotWorld, RobotWorldConfig
from curobo.wrap.reacher.trajopt import TrajOptSolver, TrajOptSolverConfig
from curobo.geom.types import Capsule, Cuboid, Cylinder, Mesh, Sphere, WorldConfig
from curobo.util_file import get_assets_path, join_path
from curobo.rollout.rollout_base import Goal
from curobo.types.base import TensorDeviceType
from curobo.types.math import Pose
from curobo.types.robot import JointState, RobotConfig
from curobo.util_file import get_robot_configs_path, get_world_configs_path, join_path, load_yaml
from curobo.wrap.reacher.motion_gen import MotionGen, MotionGenConfig, MotionGenPlanConfig
from curobo.wrap.reacher.ik_solver import IKSolver, IKSolverConfig
"""Trajectory Optimization with Custom URDF
Basic Trajectory Optimization using PyRoKi with custom URDF file.
Robot going over a wall, while avoiding world-collisions.
"""
import time
from typing import Literal, Optional
import numpy as np
import trimesh
import yourdfpy as ydf
import trimesh
from scipy.spatial.transform import Rotation as R
import os

from paradex.utils.file_io import rsc_path, shared_dir, get_robot_urdf_path
from paradex.robot.curobo import load_world_config
from paradex.inference.pickplace import PickPlaceTraj

def load_object(obj_name, obj_pose_list):
    mesh_path = f"{rsc_path}/object/{obj_name}/{obj_name}.obj"

    move = {"0":("6", "0"), "1":("9", "3")}
    obj_list = {}
    
    for obj_name, (pick_id, place_id) in move.items():
        os.makedirs(f"pickplace/object/{obj_name}", exist_ok=True)
        obj_list[obj_name] = {"start":{}, "end":{}}

        obj_list[obj_name]["start"]["pose"] = obj_pose_list[pick_id]
        obj_list[obj_name]["start"]["file_path"] = mesh_path
        
        obj_list[obj_name]["end"]["pose"] = obj_pose_list[place_id]
        obj_list[obj_name]["end"]["file_path"] = mesh_path
    return obj_list

start_pos= np.array([[0, 0, 1, 0.3],
                    [1, 0, 0, -0.3],
                    [0, 1, 0, 0.4], 
                    [0, 0, 0, 1]])

obj_pose_list = np.load("pickplace_position.npy", allow_pickle=True).item()
obj_dict = load_object("pringles", obj_pose_list)
obj_list = sorted(list(obj_dict.keys()))

os.makedirs("pickplace/traj", exist_ok=True)

traj_generator = PickPlaceTraj(start_pos, obj_dict)

cnt = 0
for pick_id in obj_list:
    os.makedirs(f"pickplace/traj/{pick_id}", exist_ok=True)

    ret = traj_generator.update(pick_id, obj_dict[pick_id]["end"]["pose"]) # eef_se3, hand_qpos, qpos, state, obj_T
    for name, val in ret.items():
        np.save(f"pickplace/traj/{pick_id}/{name}.npy", val) # what is the point? ans 
            
    