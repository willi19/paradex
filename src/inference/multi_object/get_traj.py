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
from paradex.inference.lookup_table import get_traj

class PickPlaceTraj():
    
    def __init__(self, obj_info_list, init_pose, arm_name = "xarm", hand_name="allegro"): # obj_info_list : [{"name":str, "start":np.ndarray(4,4), "end":np.ndarray(4,4)}] order is pick order
        self.obj_info_list = obj_info_list
        self.init_pose = init_pose

        self.robot_cfg = RobotConfig.from_basic(get_robot_urdf_path(arm_name, hand_name), "world", "link6")
        self.robot_cfg["kinematics"]["extra_collision_spheres"] = {"attached_object": 100}
        
        self.init_world_cfg_dict = load_world_config({obj_info["name"]:obj_info["start"] for obj_info in obj_info_list})
        self.world_config = WorldConfig().from_dict(self.init_world_cfg_dict)
        
        self.tensor_args = TensorDeviceType()
        motion_gen_config = MotionGenConfig.load_from_robot_config(
                get_robot_configs_path(arm_name, hand_name),
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
        
        self.motion_gen_env = MotionGen(motion_gen_config)
        self.motion_gen_env.reset()
        self.motion_gen_env.warmup(
            enable_graph=True, warmup_js_trajopt=True, batch_env_mode=False
        )
        
        self.m_config = MotionGenPlanConfig(
            True, True, max_attempts=5, enable_graph_attempt=3, enable_finetune_trajopt=True
        )

        ik_config = IKSolverConfig.load_from_robot_config(
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
        self.ik_solver = IKSolver(ik_config)
    
    def se3_to_quat(obj_pose):
        rotation_matrix = obj_pose[:3, :3]
        r = R.from_matrix(rotation_matrix)
        quat_xyzw = r.as_quat()  # scipy는 xyzw 순서로 반환
        quat_wxyz = np.array([quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]])  # wxyz로 변환
        
        return quat_wxyz
    
    def load_urdf(urdf_path, default_joint_config = None):
        if urdf_path is None:
            raise ValueError("URDF path must be provided for custom robot")
        # Load custom URDF from file using yourdfpy (올바른 방법)
        urdf = ydf.URDF.load(urdf_path)
        return urdf

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

    def get_lookup_pickplace_traj(self, obj_name, index):
        pick_lookup_traj = np.load(f"{shared_dir}/capture/lookup/{obj_name}/{index}/refined_pick_action.npy")
        place_lookup_traj = np.load(f"{shared_dir}/capture/lookup/{obj_name}/{index}/refined_place_action.npy")
        pick_obj_T = np.load(f"{shared_dir}/capture/lookup/{obj_name}/{index}/pick_objT.npy")
        place_obj_T = np.load(f"{shared_dir}/capture/lookup/{obj_name}/{index}/place_objT.npy")
        pick_hand_lookup_traj = np.load(f"{shared_dir}/capture/lookup/{obj_name}/{index}/refined_pick_hand.npy")
        place_hand_lookup_traj = np.load(f"{shared_dir}/capture/lookup/{obj_name}/{index}/refined_place_hand.npy")
        
        ret = {
            "pick_traj": pick_lookup_traj,
            "place_traj": place_lookup_traj,
            "pick_obj_T": pick_obj_T,
            "place_obj_T": place_obj_T,
            "pick_hand_lookup_traj": pick_hand_lookup_traj,
            "place_hand_lookup_traj": place_hand_lookup_traj,
        }
        return ret
    
    def get_pickplace_traj(self, obj_name, pick_6d, place_6d):
        pickplace_traj = self.get_lookup_pickplace_traj(obj_name, "59")
        # move from self.cur_pos to pick_T
        
        # grab object 
        
        # move from pick_T to place_T
        
        # release object
        return eef_se3, hand_qpos, qpos, state
    
    def update(self, index, place_6d):
        obj_name = self.obj_info_list[index]["name"]
        
        pick_obj_pose = self.obj_info_list[index]["start"]["pose"]
        place_obj_pose = self.obj_info_list[index]["end"]["pose"]
        
        
        start_pose = Pose(torch.from_numpy(start_pos[:3, 3]).float().to('cuda'), quaternion=torch.from_numpy(self.se3_to_quat(start_pos)).float().to('cuda'))
        
        
        
        pickplace_traj = self.get_pickplace_traj(pick_obj_pose, place_obj_pose)
    
start_pos= np.array([[0, 0, 1, 0.3],
                    [1, 0, 0, -0.3],
                    [0, 1, 0, 0.4], 
                    [0, 0, 0, 1]])


obj_pose_list = np.load("pickplace_position.npy", allow_pickle=True).item()

# obj_dict = load_object("pringles", obj_pose_list)
# obj_list = sorted(list(obj_dict.keys()))

# os.makedirs("pickplace/traj", exist_ok=True)

# cnt = 0
# for pick_id in obj_list:
#     os.makedirs(f"pickplace/traj/{pick_id}", exist_ok=True)
#     pick_traj = obj_dict[pick_id]["start"]["pose"] @ pick_lookup_traj
#     place_traj = obj_dict[pick_id]["end"]["pose"] @ place_lookup_traj

#     for state in ["start", "end"]:
#         motion_gen_env.update_world(WorldConfig().from_dict(world_cfg[cnt]))
#         cnt += 1
#         start_pose = Pose(torch.from_numpy(start_pos[:3, 3]).float().to('cuda'), quaternion=torch.from_numpy(se3_to_quat(start_pos)[0]).float().to('cuda'))
#         start_qpos = motion_gen_env.solve_ik(start_pose).solution[0]
#         start_state = JointState.from_position(start_qpos)
#         if state == "start":
#             goal_pose = Pose(torch.from_numpy(pick_traj[0, :3, 3]), quaternion=torch.from_numpy(se3_to_quat(pick_traj[0])[0]))
#             start_pos = pick_traj[-1]
#         if state == "end":
#             goal_pose = Pose(torch.from_numpy(place_traj[0, :3, 3]), quaternion=torch.from_numpy(se3_to_quat(place_traj[0])[0]))
#             start_pos = place_traj[-1]

#         # goal_pose_batch.append(goal_pose)
#         result = motion_gen_env.plan_single(start_state, goal_pose, m_config)
        
#         qpos = result.optimized_plan.position.detach().cpu().numpy()
#         hand_qpos = qpos[:, -16:]
        
#         ee_traj = motion_gen_env.compute_kinematics(result.optimized_plan)
#         pos = ee_traj.ee_pos_seq.detach().cpu().numpy()
#         quat = ee_traj.ee_quat_seq.detach().cpu().numpy()
        
#         eef_se3 = np.zeros((pos.shape[0], 4, 4))
#         eef_se3[:, :3, 3] = pos
#         eef_se3[:, :3, :3] = R.from_quat(quat[:, [1, 2, 3, 0]]).as_matrix()
#         eef_se3[:, 3, 3] = 1.0
        
#         np.save(f"pickplace/traj/{pick_id}/{state}.npy", eef_se3)
#         np.save(f"pickplace/traj/{pick_id}/{state}_hand.npy", hand_qpos)
#         np.save(f"pickplace/traj/{pick_id}/{state}_qpos.npy", qpos)


# # total_traj = []
# for i, pick_id in enumerate(obj_list):
    
#     pick_q_traj = ik_solver.solve_batch(Pose(torch.from_numpy(pick_traj[:, :3, 3]).float().to('cuda'), \
#                                                         quaternion=torch.from_numpy(np.array([se3_to_quat(pick_traj[i])[0] for i in range(len(pick_traj))])).float().to('cuda'))).solution.detach().cpu().numpy().squeeze(1)
#     place_q_traj = ik_solver.solve_batch(Pose(torch.from_numpy(place_traj[:, :3, 3]).float().to('cuda'), \
#                                                         quaternion=torch.from_numpy(np.array([se3_to_quat(place_traj[i])[0] for i in range(len(place_traj))])).float().to('cuda'))).solution.detach().cpu().numpy().squeeze(1)
#     pick_traj_q = np.concatenate([pick_q_traj, pick_hand_lookup_traj], axis=1)
#     place_traj_q = np.concatenate([place_q_traj, place_hand_lookup_traj], axis=1)

#     np.save(f"pickplace/traj/{pick_id}/pick.npy", pick_traj)
#     np.save(f"pickplace/traj/{pick_id}/place.npy", place_traj)

#     np.save(f"pickplace/traj/{pick_id}/pick_hand.npy", pick_hand_lookup_traj)
#     np.save(f"pickplace/traj/{pick_id}/place_hand.npy", place_hand_lookup_traj)

#     np.save(f"pickplace/traj/{pick_id}/pick_qpos.npy", pick_traj_q)
#     np.save(f"pickplace/traj/{pick_id}/place_qpos.npy", place_traj_q)
