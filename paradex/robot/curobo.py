import os
import numpy as np
from scipy.spatial.transform import Rotation as R
import torch

from curobo.geom.sdf.world import CollisionCheckerType
from curobo.geom.sphere_fit import SphereFitType
from curobo.geom.types import WorldConfig, Cuboid, Sphere
from curobo.rollout.rollout_base import Goal
from curobo.types.base import TensorDeviceType
from curobo.types.math import Pose
from curobo.types.robot import JointState
from curobo.util.logger import setup_curobo_logger
# from curobo.util.usd_helper import UsdHelper
from curobo.util_file import get_robot_configs_path, get_world_configs_path, join_path, load_yaml
from curobo.wrap.reacher.motion_gen import MotionGen, MotionGenConfig, MotionGenPlanConfig, PoseCostMetric
from curobo.wrap.reacher.ik_solver import IKSolver, IKSolverConfig
from curobo.wrap.reacher.mpc import MpcSolver, MpcSolverConfig

from paradex.utils.file_io import home_path

def to_quat(obj_pose):
    ret = np.zeros(7)   
    rotation_matrix = obj_pose[:3, :3]
    r = R.from_matrix(rotation_matrix)
    quat_xyzw = r.as_quat()  # scipy는 xyzw 순서로 반환
    ret[3] = quat_xyzw[3]
    ret[4:7] = quat_xyzw[0:3]
    ret[:3] = obj_pose[:3, 3]
    return ret

def load_world_config(obstacle_dict, obj_dict):
    world_config_dict = {'mesh':{}, 'cuboid':{}}
    for obstacle_name, obstacle_info in obstacle_dict['cuboid'].items():
        world_config_dict["cuboid"][str(obstacle_name)] = {
                "pose": obstacle_info["pose"],
                "dims": obstacle_info["dims"],
                "color": obstacle_info.get("color", [0.5, 0.5, 0.5, 1.0])
            }

    for obj_name, obj_info in obj_dict.items():
        world_config_dict["mesh"][str(obj_name)] = {
                "pose": to_quat(obj_info["pose"]).tolist(),
                "file_path": obj_info["file_path"],
            }
        
    return world_config_dict

class CuroboPlanner:
    def __init__(self,
                obstacle_dict,
                obj_dict,
                robot_cfg,
                tensor_args,):
        
        n_obstacle_cuboids = 30
        n_obstacle_mesh = 100
        self.obstacle_dict = obstacle_dict
        self.world_cfg = WorldConfig().from_dict(load_world_config(obstacle_dict, obj_dict))
        
        motion_gen_config = MotionGenConfig.load_from_robot_config(
            robot_cfg,
            self.world_cfg,
            tensor_args,
            collision_checker_type=CollisionCheckerType.MESH,
            num_ik_seeds=64, #128, 
            num_trajopt_seeds=64, #32, #64, # 32
            num_graph_seeds=32,#32,
            use_cuda_graph = False,
            interpolation_dt=0.01,
            collision_cache={"obb": n_obstacle_cuboids, "mesh": n_obstacle_mesh},
            particle_ik_file= os.path.join(home_path, 'curobo/src/curobo/content/assets/myrobot/my_particle_ik.yml'),
            collision_activation_distance=0.006,
            optimize_dt=True,
            trajopt_dt=0.05, #0.1, #0.1, #0.1,
            trajopt_tsteps=60# 40, #30, #70, #30, # 50 #70, (demo : 70, robothome : 30)
            
        )
        self.motion_gen = MotionGen(motion_gen_config)
        self.motion_gen.warmup(enable_graph=True, warmup_js_trajopt=False)

        self.plan_config = MotionGenPlanConfig(
            enable_graph=True,
            enable_graph_attempt=2,
            max_attempts=7,
            enable_finetune_trajopt=True, # just a rough trajectory
            time_dilation_factor=0.5,#1.0, 
        )

        self.batch_plan_config = MotionGenPlanConfig(
            enable_graph=True,
            enable_opt=True,
        )

        ik_config = IKSolverConfig.load_from_robot_config(
            robot_cfg,
            self.world_cfg,
            rotation_threshold=0.05,
            position_threshold=0.005,
            num_seeds=20,
            self_collision_check=False,
            self_collision_opt=True,
            tensor_args=tensor_args,
            use_cuda_graph=True,
            collision_checker_type=CollisionCheckerType.MESH,
            collision_cache={"obb": 0, "mesh": 0},
            # use_fixed_samples=True,
        )
        self.ik_solver = IKSolver(ik_config)      
    
    
    def get_robot_mesh(self, joint_state):
        js = JointState.from_position(torch.tensor(joint_state, device=self.motion_gen.tensor_args.device).float()).unsqueeze(0)
        return self.motion_gen.kinematics.get_visual_meshes(js)
    

    def update_world(self, obj_dict):
        # update
        self.world_cfg = WorldConfig().from_dict(load_world_config(self.obstacle_dict, obj_dict))
        self.motion_gen.clear_world_cache()
        self.motion_gen.update_world(self.world_cfg)
            

    def plan_goalset(self, init_state, goal_pose):
        init_js_state = JointState.from_position(torch.tensor(init_state, device=self.motion_gen.tensor_args.device).float()).unsqueeze(0)

        position = torch.tensor(goal_pose[:, :3, 3].astype(np.float32), device=self.motion_gen.tensor_args.device).unsqueeze(0)
        xyzw = R.from_matrix(goal_pose[:, :3, :3]).as_quat()  # xyzw
        wxyz = torch.tensor(xyzw[:, [3, 0, 1, 2]].astype(np.float32), device=self.motion_gen.tensor_args.device).unsqueeze(0)  # wxyz
        goal_pose = Pose(
            position=position,
            quaternion=wxyz,
        )
        
        result = self.motion_gen.plan_goalset(
            start_state=init_js_state,
            goal_pose=goal_pose,
            plan_config=self.plan_config,
        )
        return result.goalset_index, result.get_interpolated_plan().position.cpu().numpy()

    def plan_full_step(self, 
                        current_state : JointState, 
                        target_positions : torch.Tensor, 
                        target_quats : torch.Tensor,
                        num_grasp : int
                        ):
        """
        Plan for entire target object, provided with plausible grasping target 
        full_trajectory : generated trajectory for full object
        """

        num_batch = target_positions.shape[0]
        full_indices = self.tensor_args.to_device(torch.arange(num_batch)).to(int)

        batched_current_state = current_state.unsqueeze(0).repeat_seeds(num_batch).unsqueeze(1)

        target_states = Pose(
                position=self.tensor_args.to_device(target_positions),
                quaternion=self.tensor_args.to_device(target_quats),            
        )
        # self.motion_gen
        batch_results = self.motion_gen.plan_batch_goalset(
                                           start_state = batched_current_state[:,:,:6], # #.unsqueeze(0),
                                           goal_pose = target_states,
                                           plan_config = self.batch_plan_config)

        # for bid in range(num_batch):
        #     ik_goal = Pose(
        #         position=self.tensor_args.to_device(target_positions[[bid]]),
        #         quaternion=self.tensor_args.to_device(target_quats[[bid]]),
        #     )
        #     result = self.motion_gen.plan_single(current_state.unsqueeze(0), ik_goal, self.plan_config)
        # batch_results = result

        batch_success = batch_results.success
        # interp_plans = batch_results.get_interpolated_plan() # single result error
        plan_success = full_indices[batch_success]

        plan_success_place_ids = plan_success // num_grasp
        
        plan_tensor_ids = []
        
        # self.motion_gen.ik_solver.reset_cuda_graph()
        # self.motion_gen.ik_solver.reset_shape()
        if plan_success.shape[0] > 0:
            unique_ids = torch.unique(plan_success_place_ids)
            for uid in unique_ids:
                mask = (plan_success_place_ids == uid)
                check_ids = plan_success[mask]
                if check_ids.shape[0] > 1:
                    summed_error = batch_results.position_error[check_ids] + batch_results.rotation_error[check_ids]
                    best_id = torch.argmin(summed_error)

                    plan_tensor_ids.append(check_ids[best_id])
                else:
                    plan_tensor_ids.append(check_ids)

            plan_tensor_ids = [val.reshape(-1) for val in plan_tensor_ids]
            pos, vel, acc, jerk = [],[],[],[]
            all_plans = batch_results.get_paths()
            minlength = min([val.shape[0] for pid, val in enumerate(all_plans) if pid in plan_tensor_ids]) 
            for val in plan_tensor_ids:
                # success_js_lst = batch_results.get_successful_paths() # list of only successful paths
                cur_plan = all_plans[val]
                curlen = cur_plan.shape[0]
                if curlen > minlength:
                    sampledid = torch.linspace(0, curlen-1, minlength).long()
                    pos.append(cur_plan.position[None,sampledid])
                    vel.append(cur_plan.velocity[None,sampledid])
                    acc.append(cur_plan.acceleration[None,sampledid])
                    jerk.append(cur_plan.jerk[None,sampledid])
                else:
                    pos.append(cur_plan.position[None,:])
                    vel.append(cur_plan.velocity[None,:])
                    acc.append(cur_plan.acceleration[None,:])
                    jerk.append(cur_plan.jerk[None,:])
            if len(plan_tensor_ids) == 1:
                pos = pos[0]
                vel = vel[0]
                acc = acc[0]
                jerk = jerk[0]
            else:
                pos = torch.cat(pos, dim=0)
                vel = torch.cat(vel, dim=0)
                acc = torch.cat(acc, dim=0)
                jerk = torch.cat(jerk, dim=0)
                
            cmd_plan = JointState(
                    position=pos,
                    velocity=vel,
                    acceleration=acc,
                    jerk=jerk,
                    joint_names=all_plans[0].joint_names
                )
            cmd_plan = self.motion_gen.get_full_js(cmd_plan) # self.motion_gen.get_full_js(interp_plans)
                
            # Sample cmd_plan to rough_steps with equal intervals
            plan_length = cmd_plan.position.shape[1]

            if plan_length > self.rough_steps:
                sample_indices = torch.linspace(0, plan_length-1, self.rough_steps).long()
                cmd_plan_sampled = JointState(
                    position=cmd_plan.position[:, sample_indices],
                    velocity=cmd_plan.velocity[:, sample_indices],
                    acceleration=cmd_plan.acceleration[:, sample_indices],
                    jerk=cmd_plan.jerk[:, sample_indices],
                    joint_names=cmd_plan.joint_names
                )
            else:
                cmd_plan_sampled = cmd_plan

            self.batched_plans = cmd_plan_sampled #interp_plans
            self.batch_idx = 0
            self.mpc_idx = 0
            self.waypoint_start_time = 0

            return True, torch.cat(plan_tensor_ids).detach().cpu().numpy()
        else:
            return False, None
        
    def plan_to_joint_target(self, init_state, goal_joint_state):
        """
        Plan trajectory to reach a target joint configuration
        
        Args:
            init_state: Initial joint positions (numpy array or list)
            goal_joint_state: Target joint positions (numpy array or list)
        
        Returns:
            success: bool
            trajectory: numpy array of joint positions
        """
        # Convert to JointState
        init_js_state = JointState.from_position(
            torch.tensor(init_state, device=self.motion_gen.tensor_args.device).float()
        ).unsqueeze(0)
        
        goal_js_state = JointState.from_position(
            torch.tensor(goal_joint_state, device=self.motion_gen.tensor_args.device).float()
        ).unsqueeze(0)
        
        # Plan using trajopt (joint space planning)
        result = self.motion_gen.plan_single_js(
            start_state=init_js_state,
            goal_state=goal_js_state,
            plan_config=self.plan_config,
        )
        
        if result.success:
            trajectory = result.get_interpolated_plan().position.cpu().numpy()
            return True, trajectory
        else:
            print(f"Joint target planning failed: {result.status}")
            return False, None