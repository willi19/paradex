import numpy as np
import pickle
import os
import trimesh
from scipy.spatial.transform import Rotation as R
from threading import Event
import time
# Pick Target 

from paradex.utils.file_io import find_latest_directory, shared_dir, home_path, load_yaml, load_latest_C2R, get_robot_urdf_path, rsc_path
from paradex.visualization.visualizer.viser import ViserViewer
from paradex.robot.mimic_joint import parse_inspire
from paradex.robot.robot_wrapper import RobotWrapper
from paradex.io.robot_controller import get_arm, get_hand
from paradex.utils.keyboard_listener import listen_keyboard

import argparse
argparser = argparse.ArgumentParser()
argparser.add_argument("--use_simplemesh", action='store_true', help="use simple mesh for visualization")
args = argparser.parse_args()

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
NUM_GRASP = 9

def load_visualizer():
    visualizer = ViserViewer(up_direction=np.array([0,0,1]))

    for i in range(NUM_GRASP):
        visualizer.add_robot(f"inspire_{i}", get_robot_urdf_path(hand_name="inspire"))
    visualizer.robot_dict["inspire_0"].change_color([], [0.8, 0.2, 0.2])
    mesh_dict = {}
    for color in ["brown", "red", "yellow"]:
        if args.use_simplemesh:
            mesh_path = os.path.join(shared_dir, "object_6d", "data", "mesh", f"{color}_ramen_von", f"{color}_ramen_von_transformed_remesh.obj")
        else:
            mesh_path = os.path.join(rsc_path, "object", f"{color}_ramen_von", f"{color}_ramen_von.obj")
        mesh = trimesh.load(mesh_path)
        mesh_dict[color] = mesh

    visualizer.add_object("ramen", mesh_dict["brown"], np.eye(4))

    return visualizer

def load_grasp_traj():
    demo_data = os.path.join(shared_dir, "object_6d", "demo_data")
    demo_path = os.path.join(demo_data, "ramen")
    demo_dict = {}
    for idx in os.listdir(demo_path):
        wristSe3 = np.load(os.path.join(demo_path, idx, "grasp", "obj_wristSe3.npy"))
        hand_traj = np.load(os.path.join(demo_path, idx, "grasp", "inspire", "position.npy"))
        demo_dict[idx] = (wristSe3, hand_traj)
    return demo_dict

def merge_qpos(xarm, inspire_qpos):
    if len(xarm.shape) == 1:
        xarm = np.repeat(xarm[None, :], repeats=inspire_qpos.shape[0], axis=0)
    if len(inspire_qpos.shape) == 1:
        inspire_qpos = np.repeat(inspire_qpos[None, :], repeats=xarm.shape[0], axis=0)
    return np.concatenate([xarm, inspire_qpos], axis=1)

def load_eef_offset():
    robot_tmp = RobotWrapper(get_robot_urdf_path(arm_name="xarm", hand_name="inspire"))
    robot_tmp.compute_forward_kinematics(np.zeros(robot_tmp.dof))
    link6_se3 = robot_tmp.get_link_pose(robot_tmp.get_link_index("link6"))
    inspire_se3 = robot_tmp.get_link_pose(robot_tmp.get_link_index("wrist"))
    return np.linalg.inv(link6_se3) @ inspire_se3
    
grasp_policy_dict = load_grasp_traj()
grasp_idx = "7"

inspire_traj = grasp_policy_dict[grasp_idx][1].copy()[300:][::3]
vis_inspire_traj = parse_inspire(inspire_traj, joint_order = ['right_thumb_1_joint', 'right_thumb_2_joint', 'right_index_1_joint', 'right_middle_1_joint', 'right_ring_1_joint', 'right_little_1_joint', ])
grasp_se3 = grasp_policy_dict[grasp_idx][0] @ load_eef_offset()

visualizer = load_visualizer()

# arm_controller = get_arm("xarm")
# hand_controller = get_hand("inspire")

quit_event = Event()
start_event = Event()
# listen_keyboard({"q": quit_event, "y": start_event})

action_dict = {}
theta_list = np.linspace(0, np.pi*2, NUM_GRASP)
rot_list = [np.eye(4) for _ in range(NUM_GRASP)]

for i in range(NUM_GRASP):
    rot_list[i][:3, :3] = R.from_euler('z', theta_list[i]).as_matrix()
    # print(rot_list[i])
    wrist_pose = rot_list[i] @ grasp_se3  # @ 

    action = np.zeros(6)
    action[:3] = wrist_pose[:3,3]
    rpy = R.from_matrix(wrist_pose[:3,:3]).as_euler("zyx")
    
    action[5] = rpy[0]
    action[4] = rpy[1]
    action[3] = rpy[2]

    action_dict[f"inspire_{i}"] = merge_qpos(action, vis_inspire_traj)

visualizer.add_traj(f"grasping_demo", action_dict)
visualizer.start_viewer()