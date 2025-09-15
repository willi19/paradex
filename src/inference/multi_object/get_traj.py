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
from paradex.inference.pickplace import CollisionAvoidanceTraj

def load_object(obj_name, obj_pose_list):
    mesh_path = f"{rsc_path}/object/{obj_name}/{obj_name}.obj"

    move = {"0":("6", "0"), "1":("9", "3")}
    obj_list = {}
    
    for obj_id, (pick_id, place_id) in move.items():
        obj_list[obj_id] = {"start":{}, "end":{}}

        obj_list[obj_id]["start"]["pose"] = obj_pose_list[pick_id]
        obj_list[obj_id]["start"]["file_path"] = mesh_path

        obj_list[obj_id]["end"]["pose"] = obj_pose_list[place_id]
        obj_list[obj_id]["end"]["file_path"] = mesh_path

        obj_list[obj_id]["name"] = "pringles"
    return obj_list

start_pos = np.zeros(22)
start_pos[:6] = np.array([-48.7, -14.2, -44.9, 114.1, 55.7, 142.6]) / 180 * 3.14159
start_pos[18] = 0.36

obj_pose_list = np.load("pickplace_position.npy", allow_pickle=True).item()
obj_dict = load_object("pringles", obj_pose_list)
obj_list = sorted(list(obj_dict.keys()))
traj_generator = CollisionAvoidanceTraj(obj_dict, start_pos)

for obj_id in obj_list:
    obj_info = obj_dict[obj_id]
    action = traj_generator.pickplace(obj_id, obj_info["start"]["pose"], obj_info["end"]["pose"])
    start_pos = action[-1]
    np.save(f"pickplace/traj/{obj_id}/start_qpos.npy", action[0])
    np.save(f"pickplace/traj/{obj_id}/pick_qpos.npy", action[len(action)//3])
    np.save(f"pickplace/traj/{obj_id}/end_qpos.npy", action[2*len(action)//3])
    np.save(f"pickplace/traj/{obj_id}/place_qpos.npy", action[-1])


np.save("test_traj.npy", action)