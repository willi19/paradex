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

xarm_init_pose = np.array([-0.8048279285430908, 0.2773207128047943, -1.4464116096496582, 2.0092501640319824, 0.7059974074363708, -2.361839532852173]) # Initial xarm pose for every grasp 
robot = RobotWrapper(get_robot_urdf_path(arm_name="xarm", hand_name=None))
robot.compute_forward_kinematics(xarm_init_pose)
xarm_init_se3 = robot.get_link_pose(robot.get_link_index("link6"))
    
place_origin = np.array([-0.55, -0.45, 0.24]) # 25cm : floor + safety margin, 10cm: ramen height

Z_OFFSET = np.array([0.0, 0.0, 0.11])
Z_NUM = 2

X_OFFSET = np.array([0.13, 0.0, 0.0])
X_NUM = 2

Y_OFFSET = np.array([0.0, 0.13, 0.0])
Y_NUM = 5

LINEAR_START = 0.8
LINEAR_DIRECTION = np.array([0.0, -1.0, 0.0])
# [[ 0.04099059 -0.99915576 -0.00274577]
#  [-0.99902283 -0.04103029  0.01642854]
#  [-0.01652733  0.00206967 -0.99986127]]
C2R = load_latest_C2R()
OBSTACLE = {'cuboid': 
                {# 'base': {'dims': [0.6, 0.6, 2.07], 'pose': [-0.12577951395665932, -0.0055037614535835555, -1.041670779638955, 0.7082079218969054, -0.00040869377511796283, -0.006448498134229638, 0.7059743544943244]}, 
                 # 'baseback': {'dims': [2.0, 0.5, 2.0], 'pose': [-1.0857807924568896, -0.011288158965621076, -0.015956839973832793, 0.7082079218969054, -0.00040869377511796283, -0.006448498134229638, 0.7059743544943244]}, 
                 # 'basetop': {'dims': [5.0, 5.0, 0.2], 'pose': [-0.16635518943256555, 0.026355011122211947, 1.093385105617831, 0.7082079218969054, -0.00040869377511796283, -0.006448498134229638, 0.7059743544943244]}, 
                 'shelf0': {'dims': [0.8, 0.33, 2.02], 'pose': [-0.68+0.33/2, -0.6+0.8/2, -0.76, 0.70710678, 0, 0, 0.70710678]}, 
                 'shelf1': {'dims': [0.8, 0.03, 2.06], 'pose': [-0.68+0.03/2+0.33, -0.6+0.8/2, -0.75, 0.70710678, 0, 0, 0.70710678]}, # + 0.1
                 'shelf2': {'dims': [0.8, 0.1, 1.0], 'pose': [-0.68-0.1/2, -0.6+0.8/2, 0.2541, 0.70710678, 0, 0, 0.70710678]},  # + 1.0141
                 'shelf3': {'dims': [0.8, 0.33, 0.05], 'pose': [-0.68+0.33/2, -0.6+0.8/2, 0.574, 0.70710678, 0, 0, 0.70710678]},  #+ 1.334

                 # 'table': {'dims': [5.0, 5.0, 5.0], 'pose': [-0.07808975157119691, -0.5062144110803565, -2.584682669305668, 0.6999402146008835, 0.004682160214565101, -0.0007793753508808123, -0.7141856662901159]}}
                 }
            }

def precalculate_put_traj(desired_theta, grasp_se3):
    put_traj_dict = {}
    put_init_qpos = np.array([2.033782482147217, 0.34809282422065735, -1.556875467300415, 3.2306017875671387, 0.3351057767868042, -3.2026853561401367])
    
    for X in range(X_NUM):
        for Z in range(Z_NUM):
            place_position = place_origin + Z * Z_OFFSET + X * X_OFFSET
            put_init_obj_position = place_position - LINEAR_START * LINEAR_DIRECTION
            put_init_obj_pose = np.eye(4)
            put_init_obj_pose[:3, 3] = put_init_obj_position

            wrist_direction = grasp_se3[:3, 2][: 2] # project z axis to x-y plane
            cur_degree = np.arctan2(wrist_direction[1], wrist_direction[0])
            delta_degree = desired_theta - cur_degree

            put_init_obj_pose[:3,:3] = np.array([[np.cos(delta_degree), -np.sin(delta_degree), 0],
                                                [np.sin(delta_degree), np.cos(delta_degree), 0],
                                                [0, 0, 1]])
            put_init_pose = put_init_obj_pose @ grasp_se3
            
            place_init_obj_pose = put_init_obj_pose.copy()
            place_init_obj_pose[:3, 3] = place_position
            place_wrist_pose = place_init_obj_pose @ grasp_se3

            num_step = int(LINEAR_START * 3 * 100)
            put_xarm_traj, _ = get_linear_path(put_init_pose, place_wrist_pose, np.zeros(6), np.zeros(6), length=num_step)
            put_xarm_qpos = [put_init_qpos]
        
            for put_xarm in put_xarm_traj:
                qpos, succ = robot.solve_ik(put_xarm, "link6", put_xarm_qpos[-1])
                if not succ:
                    qpos = put_xarm_qpos[-1]
                put_xarm_qpos.append(qpos)
            put_xarm_qpos = np.array(put_xarm_qpos)[1:]
            put_traj_dict[(X, Z)] = put_xarm_qpos
    return put_traj_dict

def precalculate_return_traj(put_traj_dict):
    return_traj_dict = {}
    j1_degree = -np.pi / 3 * 2 # 120 degree
    rot_step = 180

    for key, put_qpos in put_traj_dict.items():
        init_qpos = put_qpos[0]
        ret_qpos = []

        for step in range(rot_step):
            delta_degree = j1_degree / rot_step
            target_qpos = init_qpos.copy()
            target_qpos[0] += delta_degree * (step + 1)
            ret_qpos.append(target_qpos)

        ret_qpos.extend(linear_trajectory(ret_qpos[-1], xarm_init_se3, length=50))

        return_traj_dict[key] = np.array(ret_qpos)

    return return_traj_dict

# We use object coordinate as it's center is in the bottom, middle of the object, with z-axis pointing up
# However center of ramen mesh is not as it is so we need to adjust it
ramen_offset = {
    "brown":np.array([[1, 0, 0, 0], 
                      [0, 0, 1, -0.055], 
                      [0, -1, 0, 0], 
                      [0, 0, 0, 1]]),
    "red":np.array([[1, 0, 0, 0], 
                      [0, 0, 1, -0.055], 
                      [0, -1, 0, 0], 
                      [0, 0, 0, 1]]),
    "yellow":np.array([[1, 0, 0, 0], 
                      [0, 0, 1, -0.055], 
                      [0, -1, 0, 0], 
                      [0, 0, 0, 1]])
}

def load_pick_position():
    obj_6d_path = os.path.join(shared_dir, 'object_6d', 'data', 'obj_output')
    latest_dir = find_latest_directory(obj_6d_path)
    obj_T = {}

    with open(os.path.join(obj_6d_path, latest_dir, 'obj_T.pkl'), 'rb') as f:
        obj_output = pickle.load(f)
    
    obj_idx = 0
    for obj_type, obj_list in obj_output.items():
        obj_type = obj_type.split('_')[0]  # brown_ramen_1 -> brown
        for obj_name, obj_se3 in obj_list.items():
            obj_se3 = np.linalg.inv(C2R) @ obj_se3 @ ramen_offset[obj_type]
            if obj_se3[2, 2] < 0.7:
                continue

            obj_T[f"{obj_type}_{obj_idx}"] = obj_se3
            obj_idx += 1

    return obj_T

def load_pick_traj():
    demo_path = os.path.join("data", "ramen")
    demo_dict = {}
    for idx in os.listdir(demo_path):
        wristSe3 = np.load(os.path.join(demo_path, idx, "grasp", "obj_wristSe3.npy"))
        hand_traj = np.load(os.path.join(demo_path, idx, "grasp", "inspire", "position.npy"))
        demo_dict[idx] = (wristSe3, hand_traj)
    return demo_dict

def load_visualizer(pick_position):
    visualizer = ViserViewer(up_direction=np.array([0,0,1]))

    visualizer.add_floor()
    visualizer.add_robot("xarm", get_robot_urdf_path(arm_name="xarm", hand_name="inspire"))

    mesh_dict = {}
    for color in ["brown", "red", "yellow"]:
        mesh_path = os.path.join(rsc_path, "object", f"{color}_ramen_von", f"{color}_ramen_von.obj")
        mesh = trimesh.load(mesh_path)
        mesh_dict[color] = mesh

    # for obj_name, obj_pose in pick_position.items():
    #     visualizer.add_object(obj_name, mesh_dict[obj_name.split('_')[0]], obj_pose)
    
    for obj_type, obstacles in OBSTACLE.items():
        if obj_type == 'cuboid':
            for obs_name, obs_data in obstacles.items():
                dims = obs_data['dims']
                pose = obs_data['pose']
                
                # Create box mesh
                box = trimesh.creation.box(extents=dims)
                
                # Create transformation matrix
                obs_T = np.eye(4)
                obs_T[:3, 3] = pose[:3]
                wxyz = [pose[6], pose[3], pose[4], pose[5]]  # Convert to wxyz format
                obs_T[:3, :3] = R.from_quat(wxyz).as_matrix()
                print(obs_T)
                # 그냥 add_object 호출!
                visualizer.add_object(f"obstacle_{obs_name}", box, obs_T)

    return visualizer

######################################################## utils ########################################################

def linear_trajectory(init_qpos, target_se3, length=50):
    robot.compute_forward_kinematics(init_qpos)
    init_xarm_se3 = robot.get_link_pose(robot.get_link_index("link6"))

    xarm_se3_traj, _ = get_linear_path(init_xarm_se3, target_se3, np.zeros(6), np.zeros(6), length=length)
    xarm_qpos_traj = [init_qpos]
    for xarm_se3 in xarm_se3_traj:
        qpos, succ = robot.solve_ik(xarm_se3, "link6", xarm_qpos_traj[-1])
        if not succ:
            qpos = xarm_qpos_traj[-1]
            print("IK failed")
        xarm_qpos_traj.append(qpos)
    return np.array(xarm_qpos_traj)[1:]

def get_lift_traj(init_qpos, height, length=50):
    robot.compute_forward_kinematics(init_qpos)
    init_xarm_se3 = robot.get_link_pose(robot.get_link_index("link6"))
    target_se3 = init_xarm_se3.copy()
    target_se3[2, 3] += height

    xarm_qpos_traj = linear_trajectory(init_qpos, target_se3, length=length)
    return xarm_qpos_traj

def get_obj_traj(qpos_traj, grasp_se3):
    obj_pose = []
    for qpos in qpos_traj:
        robot.compute_forward_kinematics(qpos)
        xarm_se3 = robot.get_link_pose(robot.get_link_index("link6"))
        obj_se3 = xarm_se3 @ np.linalg.inv(grasp_se3)
        obj_pose.append(obj_se3)

    return obj_pose

def merge_qpos(xarm, inspire_qpos):
    if len(xarm.shape) == 1:
        xarm = np.repeat(xarm[None, :], repeats=inspire_qpos.shape[0], axis=0)
    if len(inspire_qpos.shape) == 1:
        inspire_qpos = np.repeat(inspire_qpos[None, :], repeats=xarm.shape[0], axis=0)
    return np.concatenate([xarm, inspire_qpos], axis=1)
#########################################################################################################################

pick_position = load_pick_position()
grasp_policy_dict = load_pick_traj()

grasp_idx = "7"
inspire_traj = parse_inspire(grasp_policy_dict[grasp_idx][1], joint_order = ['right_thumb_1_joint', 'right_thumb_2_joint', 'right_index_1_joint', 'right_middle_1_joint', 'right_ring_1_joint', 'right_little_1_joint', ])[::9]

orig_inspire_traj = grasp_policy_dict[grasp_idx][1].copy()[::3]
orig_inspire_traj_pre = []
for i in range(15):
    orig_inspire_traj_pre.append(orig_inspire_traj[0] * i / 15 + np.array([1000, 1000, 1000, 1000, 1000 ,800]) * (1 - i / 15))
orig_inspire_traj = np.concatenate([np.array(orig_inspire_traj_pre), orig_inspire_traj], axis=0)


grasp_se3 = grasp_policy_dict[grasp_idx][0]

visualizer = load_visualizer(pick_position)

desired_theta = np.pi * (1) # degree between x-axis and z-axis of wrist np.pi ~ np.pi * 3 / 2
put_traj_dict = precalculate_put_traj(desired_theta, grasp_se3)
ret_traj_dict = precalculate_return_traj(put_traj_dict)

robot.compute_forward_kinematics(xarm_init_pose)
init_xarm_se3 = robot.get_link_pose(robot.get_link_index("link6"))

for step in range(20):
    tot_traj = []
    # move to initial of put trajectory
    put_xarm_traj = put_traj_dict[((step // Z_NUM) % X_NUM, step % Z_NUM)].copy()
    
    put_traj_len = int((LINEAR_START - (step // (X_NUM * Z_NUM)) * np.linalg.norm(Y_OFFSET)) * 100 * 3)
    put_xarm_traj = put_xarm_traj[:put_traj_len]
    # put 

    visualizer.add_traj(f"put_{step}", {"xarm":merge_qpos(put_xarm_traj, inspire_traj[-1])}, {"brown_0":np.array(get_obj_traj(put_xarm_traj, grasp_se3))})
    tot_traj.append(merge_qpos(put_xarm_traj, orig_inspire_traj[-1]))
    # down
    down_xarm_traj = get_lift_traj(put_xarm_traj[-1], height=-0.02, length=30)
    down_obj_pose = get_obj_traj(down_xarm_traj[:, :6], grasp_se3)
    down_traj = merge_qpos(down_xarm_traj, np.repeat(inspire_traj[-1][None, :], repeats=down_xarm_traj.shape[0], axis=0))
    visualizer.add_traj(f"down_{step}", {"xarm":down_traj}, {"brown_0":np.array(down_obj_pose)})
    tot_traj.append(merge_qpos(down_xarm_traj, orig_inspire_traj[-1]))
    # release
    release_inspire_traj = inspire_traj[::-1]
    release_traj = merge_qpos(down_xarm_traj[-1], release_inspire_traj)
    visualizer.add_traj(f"release_{step}", {"xarm":release_traj})
    tot_traj.append(merge_qpos(down_xarm_traj[-1], orig_inspire_traj[::-1]))
    # up
    up_xarm_traj = get_lift_traj(down_xarm_traj[-1, :6], height=0.02, length=30)
    up_traj = merge_qpos(up_xarm_traj, np.repeat(inspire_traj[0][None, :], repeats=up_xarm_traj.shape[0], axis=0))
    visualizer.add_traj(f"up_{step}", {"xarm":up_traj})
    tot_traj.append(merge_qpos(up_xarm_traj, orig_inspire_traj[0]))
    # out of shelf
    out_xarm_traj = put_xarm_traj[::-1]
    out_traj = merge_qpos(out_xarm_traj, np.repeat(inspire_traj[0][None, :], repeats=out_xarm_traj.shape[0], axis=0))
    visualizer.add_traj(f"out_{step}", {"xarm":out_traj})
    tot_traj.append(merge_qpos(out_xarm_traj, orig_inspire_traj[0]))
    # back to init
    ret_xarm_traj = ret_traj_dict[((step // Z_NUM) % X_NUM, step % Z_NUM)].copy()
    ret_traj = merge_qpos(ret_xarm_traj, np.repeat(inspire_traj[0][None, :], repeats=ret_xarm_traj.shape[0], axis=0))
    visualizer.add_traj(f"init_{step}", {"xarm":ret_traj})
    tot_traj.append(merge_qpos(ret_xarm_traj, orig_inspire_traj[0]))
    
    os.makedirs(os.path.join("data", "place_traj"), exist_ok=True)
    np.save(os.path.join("data", "place_traj", f"{step}.npy"), np.concatenate(tot_traj, axis=0))

print("Done")
visualizer.start_viewer()