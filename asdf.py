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

from paradex.utils.file_io import rsc_path
from paradex.robot.curobo import load_world_config

def se3_to_quat(obj_pose):
    rotation_matrix = obj_pose[:3, :3]
    r = R.from_matrix(rotation_matrix)
    quat_xyzw = r.as_quat()  # scipy는 xyzw 순서로 반환
    quat_wxyz = np.array([quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]])  # wxyz로 변환
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
for pick_id in obj_list:
    for state in ["start", "end"]:
        scene_obj_dict = {}
        
        for obj_name in obj_list:
            if obj_name < pick_id:
                scene_obj_dict[obj_name] = obj_dict[obj_name]["end"]
            elif obj_name > pick_id:
                scene_obj_dict[obj_name] = obj_dict[obj_name]["start"]
            if obj_name == pick_id and state == "start":
                scene_obj_dict[obj_name] = obj_dict[obj_name]["start"]
            world_cfg.append(load_world_config(scene_obj_dict))

tensor_args = TensorDeviceType()
motion_gen_config = MotionGenConfig.load_from_robot_config(
        robot_cfg,
        world_cfg,
        tensor_args,
        trajopt_tsteps=30,
        use_cuda_graph=False,
        num_trajopt_seeds=4,
        num_ik_seeds=30,
        num_batch_ik_seeds=30,
        evaluate_interpolated_trajectory=True,
        interpolation_dt=0.05,
        interpolation_steps=500,
        grad_trajopt_iters=30,
        collision_checker_type=CollisionCheckerType.MESH
    )
motion_gen = MotionGen(motion_gen_config)
world = WorldConfig().from_dict(world_cfg[0])
world.save_world_as_mesh("scene.obj")
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