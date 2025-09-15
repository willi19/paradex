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
from paradex.inference.lookup_table import LookupTable


class CollisionAvoidanceTraj():
    def __init__(self, obj_info_list, cur_pose, arm_name = "xarm", hand_name="allegro"): # obj_info_list : [{"name":str, "start":np.ndarray(4,4), "end":np.ndarray(4,4)}] order is pick order
        self.obj_info_list = obj_info_list
        self.obj_list = list(set([obj_info["name"] for obj_info in obj_info_list.values()]))
        self.cur_state = JointState.from_position(torch.from_numpy(cur_pose).float().to('cuda').unsqueeze(0))

        self.arm_name = arm_name
        self.hand_name = hand_name
        
        urdf_path = get_robot_urdf_path(arm_name, hand_name)
        yml_path = os.path.join(f"{rsc_path}/robot/{arm_name}_{hand_name}.yml")
        
        self.robot_cfg = load_yaml(yml_path)['robot_cfg']
        self.robot_cfg["kinematics"]["extra_collision_spheres"] = {"attached_object": 100}

        self.world_cfg_dict = load_world_config({obj_name:obj_info["start"] for obj_name, obj_info in obj_info_list.items()})
        self.world_config = WorldConfig().from_dict(self.world_cfg_dict)
        
        self.tensor_args = TensorDeviceType()
        self.motion_gen_config = MotionGenConfig.load_from_robot_config(
                self.robot_cfg,
                self.world_config,
                self.tensor_args,
                trajopt_tsteps=200,
                use_cuda_graph=False,
                num_trajopt_seeds=4,
                num_ik_seeds=50,
                num_batch_ik_seeds=30,
                evaluate_interpolated_trajectory=True,
                interpolation_dt=0.03,
                interpolation_steps=500,
                grad_trajopt_iters=30,
                collision_checker_type=CollisionCheckerType.MESH,
                self_collision_check=False,
                self_collision_opt=False,
            )
        
        self.motion_gen_env = MotionGen(self.motion_gen_config)
        self.motion_gen_env.reset()
        self.motion_gen_env.warmup(
            enable_graph=True, warmup_js_trajopt=True, batch_env_mode=False
        )
        
        self.m_config = MotionGenPlanConfig(
            True, True, enable_finetune_trajopt=True
        )

        self.ik_config = IKSolverConfig.load_from_robot_config(
            self.robot_cfg,
            None,
            rotation_threshold=0.05,
            position_threshold=0.005,
            num_seeds=20,
            self_collision_check=True,
            self_collision_opt=True,
            tensor_args=self.tensor_args,
            use_cuda_graph=True,
        )
        self.ik_solver = IKSolver(self.ik_config)
        
        self.lookup_table = {obj_name:LookupTable(obj_name, self.hand_name, None) for obj_name in self.obj_list}
    
    @ staticmethod
    def eef_se3_to_Pose(eef_se3):
        position = torch.from_numpy(eef_se3[..., :3, 3]).float().to('cuda').unsqueeze(0)
        quat = R.from_matrix(eef_se3[..., :3, :3]).as_quat() # x, y, z, w
        quaternion = torch.from_numpy(quat[..., [3, 0, 1, 2]]).float().to('cuda').unsqueeze(0) # w, x, y, z
        return Pose(position, quaternion)
    
    # def get_pickplace_traj(self, obj_name, pick_6d, place_6d):
    #     pickplace_traj = self.get_lookup_pickplace_traj(obj_name, "59")
    #     # move from self.cur_pos to pick_T
        
    #     # grab object 
        
    #     # move from pick_T to place_T
        
    #     # release object
    #     return eef_se3, hand_qpos, qpos, state
    
    def move(self, goal_pos, update=True, start_pos=None):
        if start_pos is None:
            start_pos = self.cur_pos.copy()

        start_pose = self.eef_se3_to_Pose(start_pos)
        start_qpos = self.motion_gen_env.solve_ik(start_pose).solution[0]
        start_state = JointState.from_position(start_qpos)
        
        goal_pose = Pose(torch.from_numpy(goal_pos[:3, 3]), quaternion=torch.from_numpy(se3_to_quat(goal_pos)[0]))
        result = self.motion_gen_env.plan_single(start_state, goal_pose, self.m_config)
            
        qpos = result.optimized_plan.position.detach().cpu().numpy()
        if update:
            self.cur_pos = qpos[-1]

    def update_world(self, obj_name, obj_pose):
        self.world_cfg_dict[obj_name] = obj_pose
        self.world_config = WorldConfig().from_dict(self.world_cfg_dict)
        self.motion_gen_env.update_world(self.world_config)

    def get_world_mesh(self):
        return self.world_config.save_world_as_mesh()
    
    def pickplace(self, obj_name, place_6d):
        # Start : Cur pos -> pick pos
        # Pick : pick pos trajectory
        # Move : pick pos -> place pos
        # Place : place pos trajectory
        # End : place pos -> rest pos
        obj_type = self.obj_info_list[obj_name]["name"]

        start_state = self.cur_state
        pick_6D = self.obj_info_list[obj_name]["start"]["pose"]
        place_6D = self.obj_info_list[obj_name]["end"]["pose"]

        pickplace_traj = self.lookup_table[obj_type].get_trajs(pick_6D, place_6D) # Multiple pick traj to search
        grasp_poses_se3 = [pickplace_traj[idx]["pick"]["eef_se3"][0] for idx in range(len(pickplace_traj))]
        grasp_poses = self.eef_se3_to_Pose(np.array(grasp_poses_se3))
        goal_motion_gen_result = self.motion_gen_env.plan_goalset(start_state, grasp_poses, self.m_config)
        action = goal_motion_gen_result.get_interpolated_plan()
        
        return action.position.detach().cpu().numpy()
        
        # Pick
        # we simply follow the pick traj
        # Then we need to update the world+
        
        # Add object to the robot collision
        # Change the object pose in the world
        # Change the robot pose to the last of pick traj
        
        # Move
        # We need to move from the last of pick traj to the first of place traj
        
        # Place
        # we simply follow the place traj
        # Then we need to update the world
        # Remove object from the robot collision
        # Change the object pose in the world
        # Change the robot pose to the last of place traj       
        
    