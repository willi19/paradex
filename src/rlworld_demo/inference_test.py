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
from paradex.io.robot_controller import get_arm, get_hand
from paradex.utils.keyboard_listener import listen_keyboard

import argparse
argparser = argparse.ArgumentParser()
argparser.add_argument("--use_simplemesh", action='store_true', help="use simple mesh for visualization")
args = argparser.parse_args()

xarm_init_pose = np.array([-0.8048279285430908, 0.2773207128047943, -1.4464116096496582, 2.0092501640319824, 0.7059974074363708, -2.361839532852173]) # Initial xarm pose for every grasp 
C2R = load_latest_C2R()

OBSTACLE = {'cuboid': 
                { # xyz, quaternion
                 'table': {'dims': [2, 2, 0.2], 'pose': [0,0,-0.12, \
                                           0,0,0,1]}, 
                 # 'baseback': {'dims': [2.0, 0.5, 2.0], 'pose': [-1.0857807924568896, -0.011288158965621076, -0.015956839973832793, 0.7082079218969054, -0.00040869377511796283, -0.006448498134229638, 0.7059743544943244]}, 
                 'basetop': {'dims': [5.0, 5.0, 0.2], 'pose': [0, 0, 1.0, 0, 0, 0, 1]}, 
                 'shelf0': {'dims': [0.8, 0.33, 2.02], 'pose': [-0.68+0.33/2, -0.6+0.8/2, -0.76, 0.70710678, 0, 0, 0.70710678]}, 
                 'shelf1': {'dims': [0.8, 0.03, 2.06], 'pose': [-0.68+0.03/2+0.33, -0.6+0.8/2, -0.75, 0.70710678, 0, 0, 0.70710678]}, # + 0.1
                 'shelf2': {'dims': [0.8, 0.1, 1.0], 'pose': [-0.68-0.1/2, -0.6+0.8/2, 0.2541, 0.70710678, 0, 0, 0.70710678]},  # + 1.0141
                 'shelf3': {'dims': [0.8, 0.33, 0.05], 'pose': [-0.68+0.33/2, -0.6+0.8/2, 0.574, 0.70710678, 0, 0, 0.70710678]},  #+ 1.334
                 # 'table': {'dims': [5.0, 5.0, 5.0], 'pose': [-0.07808975157119691, -0.5062144110803565, -2.584682669305668, 0.6999402146008835, 0.004682160214565101, -0.0007793753508808123, -0.7141856662901159]}}
                 }
            }

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
PICK_ORDER = [
    'brown_1', 
    'red_12', 
    'red_13', 
    'red_14', 
    'brown_0', 
    'yellow_6', 
    'red_16', 
    'yellow_5', 
    'yellow_8', 
    'yellow_9', 
    'red_18', 
    'yellow_10', 
    'yellow_11', 
    'brown_2', 
    'yellow_7', 
    'red_19', 
    'brown_3', 
    "red_15", 
    "red_17", 
    "brown_4"
]

def load_pick_position():
    obj_6d_path = os.path.join(shared_dir, 'object_6d', 'data', 'obj_output')
    latest_dir = find_latest_directory(obj_6d_path)
    print(f"Loading pick position from {os.path.join(obj_6d_path, latest_dir)}")
    obj_T = {}

    with open(os.path.join(obj_6d_path, latest_dir, 'obj_T.pkl'), 'rb') as f:
        obj_output = pickle.load(f)
    obj_idx = 0
    for obj_type, obj_list in obj_output.items():
        obj_type = obj_type.split('_')[0]  # brown_ramen_1 -> brown
        for obj_name, obj_se3 in obj_list.items():
            if f"{obj_type}_{obj_idx}" in PICK_ORDER[10:]:
                obj_se3 = np.linalg.inv(C2R) @ obj_se3 @ ramen_offset[obj_type]

                obj_T[f"{obj_type}_{obj_idx}"] = obj_se3
            obj_idx += 1

    return obj_T

def load_visualizer(pick_position):
    print(pick_position)
    visualizer = ViserViewer(up_direction=np.array([0,0,1]))

    # visualizer.add_floor()
    visualizer.add_robot("xarm", get_robot_urdf_path(arm_name="xarm", hand_name="inspire"))

    mesh_dict = {}
    for color in ["brown", "red", "yellow"]:
        if args.use_simplemesh:
            mesh_path = os.path.join(shared_dir, "object_6d", "data", "mesh", f"{color}_ramen_von", f"{color}_ramen_von_transformed_remesh.obj")
        else:
            mesh_path = os.path.join(rsc_path, "object", f"{color}_ramen_von", f"{color}_ramen_von.obj")
        mesh = trimesh.load(mesh_path)
        mesh_dict[color] = mesh

    for obj_name, obj_pose in pick_position.items():
        visualizer.add_object(obj_name, mesh_dict[obj_name.split('_')[0]], obj_pose)
    
    # make trimesh objects
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
                # 그냥 add_object 호출!
                visualizer.add_object(f"obstacle_{obs_name}", box, obs_T)

    return visualizer

pick_position = load_pick_position()
visualizer = load_visualizer(pick_position)

# arm_controller = get_arm("xarm")
# hand_controller = get_hand("inspire")

quit_event = Event()
start_event = Event()
listen_keyboard({"q": quit_event, "y": start_event})

for step in range(len(pick_position)):#len(pick_position)):
    pick_traj = np.load(os.path.join("data", "refine_pick_traj", f"{step}.npy"))
    vis_pick_traj = pick_traj.copy()
    vis_pick_traj[:, 6:] = parse_inspire(vis_pick_traj[:,6:], joint_order = ['right_thumb_1_joint', 'right_thumb_2_joint', 'right_index_1_joint', 'right_middle_1_joint', 'right_ring_1_joint', 'right_little_1_joint', ])
    visualizer.add_traj(f"pick_{step}", {"xarm":vis_pick_traj})

    place_traj = np.load(os.path.join("data", "place_traj", f"{step}.npy"))
    vis_place_traj = place_traj.copy()
    vis_place_traj[:, 6:] = parse_inspire(vis_place_traj[:,6:], joint_order = ['right_thumb_1_joint', 'right_thumb_2_joint', 'right_index_1_joint', 'right_middle_1_joint', 'right_ring_1_joint', 'right_little_1_joint', ])
    visualizer.add_traj(f"place_{step}", {"xarm":vis_place_traj})

    tot_traj = np.concatenate([pick_traj, place_traj], axis=0)
    # arm_controller.home_robot(xarm_init_pose)
    # hand_controller.home_robot(pick_traj[0, 6:])

    # print(f"Press 'y' to start step {step}")
    # while not start_event.is_set() and not quit_event.is_set():
    #     time.sleep(0.01)
    # if quit_event.is_set():
    #     break
    # start_event.clear()

    # for qpos in tot_traj:
    #     if quit_event.is_set():
    #         break
    #     arm_controller.set_action(qpos[:6])
    #     hand_controller.set_target_action(qpos[6:])
    #     time.sleep(0.02)
        
    # if quit_event.is_set():
    #     break

# arm_controller.quit()
# hand_controller.quit()
visualizer.start_viewer()