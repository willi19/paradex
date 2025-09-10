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

from paradex.utils.file_io import rsc_path, shared_dir
from paradex.robot.curobo import load_world_config

def se3_to_quat(obj_pose):
    rotation_matrix = obj_pose[:3, :3]
    r = R.from_matrix(rotation_matrix)
    quat_xyzw = r.as_quat()  # scipy는 xyzw 순서로 반환
    quat_wxyz = np.array([quat_xyzw[1], quat_xyzw[2], quat_xyzw[3], quat_xyzw[0]])  # wxyz로 변환
    position = obj_pose[:3, 3]
    
    return quat_wxyz, position

def load_urdf(urdf_path, default_joint_config = None):
    if urdf_path is None:
        raise ValueError("URDF path must be provided for custom robot")
    # Load custom URDF from file using yourdfpy (올바른 방법)
    urdf = ydf.URDF.load(urdf_path)
    return urdf

def load_object(obj_name):
    mesh_path = f"{rsc_path}/object/{obj_name}/{obj_name}.obj"
    os.makedirs(f"pickplace/object/{obj_name}", exist_ok=True)
    
    num_row = 2
    num_col = 3
    obj_list = {}
    
    for i in range(num_row):
        for j in range(num_col):
            ind  = i * num_col + j
            os.makedirs(f"pickplace/object/{ind}", exist_ok=True)
            obj_list[ind] = {"start":{}, "end":{}}
            z = 0.096 - 0.041
            obj_list[ind]["start"]["pose"] = np.eye(4)
            obj_list[ind]["start"]["pose"][:3, 3] = [0.15 * i + 0.25, 0.15 * j - 0.3, z]
            obj_list[ind]["start"]["file_path"] = mesh_path
            obj_list[ind]["end"]["pose"]  = np.eye(4)
            obj_list[ind]["end"]["pose"][:3, 3] = [0.15 * i + 0.25, 0.15 * j + 0.3, z]
            obj_list[ind]["end"]["file_path"] = mesh_path
    return obj_list

start_pos= np.array([[0, 0, 1, 0.3],
                    [1, 0, 0, -0.3],
                    [0, 1, 0, 0.4], 
                    [0, 0, 0, 1]])
robot_cfg = RobotConfig.from_basic(os.path.join(f"{rsc_path}/robot/xarm_allegro.urdf"), "world", "link6")
obj_dict = load_object("pringles")
obj_list = sorted(list(obj_dict.keys()))

world_cfg = []


lookup_table_path = os.path.join(shared_dir, "capture", "lookup")
index = "1"
index_path = os.path.join(lookup_table_path, "pringles", index)
pick_lookup_traj = np.load(f"{index_path}/refined_pick_action.npy")
place_lookup_traj = np.load(f"{index_path}/refined_place_action.npy")


for pick_id in obj_list:
    for state in ["start", "end"]:
        scene_obj_dict = {}
        
        for obj_name in obj_list:
            # if obj_name < pick_id:
            #     scene_obj_dict[obj_name] = obj_dict[obj_name]["end"]
            # elif obj_name > pick_id:
            #     scene_obj_dict[obj_name] = obj_dict[obj_name]["start"]
            # if obj_name == pick_id and state == "start":
            #     scene_obj_dict[obj_name] = obj_dict[obj_name]["start"]
            pass
        
        world_cfg.append(load_world_config(scene_obj_dict))

    
tensor_args = TensorDeviceType()
motion_gen_config = MotionGenConfig.load_from_robot_config(
        os.path.join(f"{rsc_path}/robot/xarm_allegro.yml"),
        world_cfg,
        tensor_args,
        trajopt_tsteps=30,
        use_cuda_graph=False,
        num_trajopt_seeds=4,
        num_ik_seeds=30,
        num_batch_ik_seeds=30,
        evaluate_interpolated_trajectory=True,
        interpolation_dt=0.03,
        interpolation_steps=500,
        grad_trajopt_iters=30,
        collision_checker_type=CollisionCheckerType.MESH,
        self_collision_check=False,
        self_collision_opt=False,
    )

world = WorldConfig().from_dict(world_cfg[0])
world.save_world_as_mesh("scene.obj")
    
    
motion_gen_batch_env = MotionGen(motion_gen_config)
motion_gen_batch_env.reset()
motion_gen_batch_env.warmup(
    enable_graph=False, batch=len(obj_list)*2, warmup_js_trajopt=False, batch_env_mode=True
)
n_envs =  len(obj_list)*2

start_state_batch = []
goal_quat_batch = []
goal_pose_batch = []
for pick_id in obj_list:
    pick_traj = obj_dict[pick_id]["start"]["pose"] @ pick_lookup_traj
    place_traj = obj_dict[pick_id]["end"]["pose"] @ place_lookup_traj
    
    for state in ["start", "end"]:
        start_pose = Pose(torch.from_numpy(start_pos[:3, 3]).float().to('cuda'), quaternion=torch.from_numpy(se3_to_quat(start_pos)[0]).float().to('cuda'))
        start_qpos = motion_gen_batch_env.solve_ik(start_pose).solution
        start_state_batch.append(start_qpos)# JointState.from_position(start_qpos))
        
        if state == "start":
            goal_pose_batch.append(torch.from_numpy(pick_traj[0, :3, 3]).float().to('cuda'))
            goal_quat_batch.append(torch.from_numpy(se3_to_quat(pick_traj[0])[0]).float().to('cuda'))
            start_pos = pick_traj[-1]
            # goal_pose = Pose(torch.from_numpy(pick_traj[0, :3, 3]), quaternion=torch.from_numpy(se3_to_quat(pick_traj[0])[0]))
        if state == "end":
            goal_pose_batch.append(torch.from_numpy(place_traj[0, :3, 3]).float().to('cuda'))
            goal_quat_batch.append(torch.from_numpy(se3_to_quat(place_traj[0])[0]).float().to('cuda'))
            start_pos = place_traj[-1]
            # goal_pose = Pose(torch.from_numpy(place_traj[0, :3, 3]), quaternion=torch.from_numpy(se3_to_quat(place_traj[0])[0]))
        
        # goal_pose_batch.append(goal_pose)

goal_pose = Pose(torch.stack(goal_pose_batch), quaternion=torch.stack(goal_quat_batch))
start_state = JointState.from_position(torch.stack(start_state_batch))

m_config = MotionGenPlanConfig(
    False, True, max_attempts=1, enable_graph_attempt=None, enable_finetune_trajopt=False
)
result = motion_gen_batch_env.plan_batch_env(start_state, goal_pose, m_config)
q = result.optimized_plan.position.detach().cpu().numpy()
print(q.shape)
# print(n_envs, result.total_time, result.total_time / n_envs)
# np.save("pickplace/traj.npy", q)
ik_config = IKSolverConfig.load_from_robot_config(
    robot_cfg,
    None,
    rotation_threshold=0.05,
    position_threshold=0.005,
    num_seeds=20,
    self_collision_check=True,
    self_collision_opt=True,
    tensor_args=tensor_args,
    use_cuda_graph=True,
)
ik_solver = IKSolver(ik_config)

total_traj = []
for i, pick_id in enumerate(obj_list):
    pick_traj = obj_dict[pick_id]["start"]["pose"] @ pick_lookup_traj
    place_traj = obj_dict[pick_id]["end"]["pose"] @ place_lookup_traj
    
    pick_q_traj = ik_solver.solve_batch(Pose(torch.from_numpy(pick_traj[:, :3, 3]).float().to('cuda'), \
                                                        quaternion=torch.from_numpy(np.array([se3_to_quat(pick_traj[i])[0] for i in range(len(pick_traj))])).float().to('cuda'))).solution.detach().cpu().numpy().squeeze(1)
    place_q_traj = ik_solver.solve_batch(Pose(torch.from_numpy(place_traj[:, :3, 3]).float().to('cuda'), \
                                                        quaternion=torch.from_numpy(np.array([se3_to_quat(place_traj[i])[0] for i in range(len(place_traj))])).float().to('cuda'))).solution.detach().cpu().numpy().squeeze(1)
    pick_q_traj = np.concatenate([pick_q_traj, np.zeros((pick_q_traj.shape[0], 16))], axis=1)
    place_q_traj = np.concatenate([place_q_traj, np.zeros((place_q_traj.shape[0], 16))], axis=1)
    # print(pick_q
    total_traj.append(q[2 * i])
    total_traj.append(pick_q_traj)
    total_traj.append(q[2 * i + 1])
    total_traj.append(place_q_traj)
    # print(q[2 * i].shape, pick_q_traj.shape, q[2 * i + 1].shape, place_q_traj.shape)
total_traj = np.concatenate(total_traj, axis=0)
np.save("pickplace/traj.npy", total_traj)
# for pick_id in obj_list:
#     for state in ["start", "end"]:
#         scene_obj_dict = {}
        
#         for obj_name in obj_list:
#             if obj_name < pick_id:
#                 scene_obj_dict[obj_name] = obj_dict[obj_name]["end"]
#             elif obj_name > pick_id:
#                 scene_obj_dict[obj_name] = obj_dict[obj_name]["start"]
#             if obj_name == pick_id and state == "start":
#                 scene_obj_dict[obj_name] = obj_dict[obj_name]["start"]
#         world_cfg = load_world_config(scene_obj_dict)
#         motion_gen.update_world_config(world_cfg)
#         target_link_name = "link6"
#         start_pose = np.eye(4)
#         start_pose[:3, 3] = [0.3, -0.3, 0.4]
#         end_pose = np.eye(4)
#         end_pose[:3, 3] = [0.3, 0.3, 0.4]
#         start_quat, start_pos = se3_to_quat(start_pose)
#         end_quat, end_pos = se3_to_quat(end_pose)
#         timesteps, dt = 50, 0.04
#         # Solve trajectory optimization
#         start_time = time.time()
#         traj = motion_gen.plan(
#             target_link_name,
#             start_pos,
#             start_quat,
#             end_pos,
#             end_quat,
#             timesteps,
#             dt,
#         )
#         print(time.time()-start_time, "cost")
#         traj = np.array(traj)
#         os.makedirs(f"pickplace/{pick_id}/{state}", exist_ok=True)
#         np.save(f"pickplace/{pick_id}/{state}/traj.npy", traj)
# pick_traj = np.load(f"{rsc_path}/lookup/pringles/stand_allegro/1/pick.npy")
# place_traj = np.load(f"{rsc_path}/lookup/pringles/stand_allegro/1/place.npy")
# pick_pose = Pose(state.ee_pos_seq.squeeze(), quaternion=state.ee_quat_seq.squeeze())
# place_pose = Pose(state.ee_pos_seq.squeeze(), quaternion=state.ee_quat_seq.squeeze())

# result = motion_gen.plan(
# os.makedirs("pickplace", exist_ok=True)
# obj_list = load_object(data_path)
# pickplace_order = list(obj_list.keys())
# pickplace_order.sort()
# robot_coll = pk.collision.RobotCollision.from_urdf(base_urdf)
# ground_coll = pk.collision.HalfSpace.from_point_and_normal(
#     np.array([0.0, 0.0, 0.0]), np.array([0.0, 0.0, 1.0])
# )
# init_pose = np.array([[0, 0, 1, 0.3],
#                 [1, 0, 0, 0.35],
#                 [0, 1, 0, 0.10],
#                 [0, 0, 0, 1]])
# term_pose = np.array([[0, 0, 1, 0.3],
#                 [1, 0, 0, -0.35],
#                 [0, 1, 0, 0.10],
#                 [0, 0, 0, 1]])
# cur_pose = init_pose
# timesteps, dt = 50, 0.04
# demo_path = os.path.join("data", "lookup", "pringles", "stand_allegro", str(1))
# pick_traj = np.load(f"{demo_path}/pick.npy")
# place_traj = np.load(f"{demo_path}/place.npy")
# state = {i:"start" for i in pickplace_order}
# for pick_id in [1,2,3,4,5,6]:
#     os.makedirs(f"pickplace/{pick_id}/pick", exist_ok=True)
#     os.makedirs(f"pickplace/{pick_id}/place", exist_ok=True)
#     start_pose, end_pose = get_traj_point(pick_traj, obj_list[pick_id]["start"]["pose"], place_traj, \
#                                           obj_list[pick_id]["end"]["pose"])
#     # pick
#     obj_coll_list = [obj_list[ind][state[ind]]["coll"] for ind in list(obj_list.keys())]
#     world_coll = [ground_coll] + obj_coll_list
#     np.save(f"pickplace/{pick_id}/pick/start_pose.npy", cur_pose)
#     np.save(f"pickplace/{pick_id}/pick/end_pose.npy", start_pose)
#     start_quat, start_pos = se3_to_quat(cur_pose)
#     end_quat, end_pos = se3_to_quat(start_pose)
#     start_time = time.time()
#     traj = pks.solve_trajopt(
#         robot,
#         robot_coll,
#         world_coll,
#         target_link_name,
#         start_pos,
#         start_quat,
#         end_pos,
#         end_quat,
#         timesteps,
#         dt,
#     )
#     print(time.time()-start_time, "cost")
#     traj = np.array(traj)
#     np.save(f"pickplace/{pick_id}/pick/traj.npy", traj)
#     cur_pose = start_pose.copy()
#     #place
#     obj_coll_list = [obj_list[ind][state[ind]]["coll"] for ind in list(obj_list.keys()) if ind != pick_id]
#     world_coll = [ground_coll] + obj_coll_list
#     np.save(f"pickplace/{pick_id}/place/start_pose.npy", cur_pose)
#     np.save(f"pickplace/{pick_id}/place/end_pose.npy", end_pose)
#     start_quat, start_pos = se3_to_quat(cur_pose)
#     end_quat, end_pos = se3_to_quat(end_pose)
#     # Solve trajectory optimization
#     start_time = time.time()
#     traj = pks.solve_trajopt(
#         robot,
#         robot_coll,
#         world_coll,
#         target_link_name,
#         start_pos,
#         start_quat,
#         end_pos,
#         end_quat,
#         timesteps,
#         dt,
#     )
#     print(time.time()-start_time, "cost")
#     traj = np.array(traj)
#     np.save(f"pickplace/{pick_id}/place/traj.npy", traj)
#     cur_pose = end_pose.copy()
    
# start_quat, start_pos = se3_to_quat(cur_pose)
# end_quat, end_pos = se3_to_quat(term_pose)
# traj = pks.solve_trajopt(
#         robot,
#         robot_coll,
#         world_coll,
#         target_link_name,
#         start_pos,
#         start_quat,
#         end_pos,
#         end_quat,
#         timesteps,
#         dt,
#     )
# os.makedirs("pickplace/term", exist_ok=True)
# np.save(f"pickplace/term/traj.npy", traj)

# robot_file = "franka.yml"

# # create a world from a dictionary of objects
# # cuboid: {} # dictionary of objects that are cuboids
# # mesh: {} # dictionary of objects that are meshes
# world_config = {
#     "cuboid": {
#         "table": {"dims": [2, 2, 0.2], "pose": [0.4, 0.0, -0.1, 1, 0, 0, 0]},
#         "cube_1": {"dims": [0.1, 0.1, 0.2], "pose": [0.4, 0.0, 0.5, 1, 0, 0, 0]},
#     },
#     "mesh": {
#         "scene": {
#             "pose": [1.5, 0.080, 1.6, 0.043, -0.471, 0.284, 0.834],
#             "file_path": "scene/nvblox/srl_ur10_bins.obj",
#         }
#     },
# }
# tensor_args = TensorDeviceType()
# config = RobotWorldConfig.load_from_config(robot_file, world_config,
#                                           collision_activation_distance=0.0)
# curobo_fn = RobotWorld(config)
# print("done loading")

# # create spheres with shape batch, horizon, n_spheres, 4.
# q_sph = torch.randn((10, 1, 1, 4), device=tensor_args.device, dtype=tensor_args.dtype)
# q_sph[...,3] = 0.2 # radius of spheres
# d = curobo_fn.get_collision_distance(q_sph)
# print(d)

# q_s = curobo_fn.sample(5, mask_valid=False)
# d_world, d_self = curobo_fn.get_world_self_collision_distance_from_joints(q_s)
# print(d_world, d_self)