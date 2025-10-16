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

Z_OFFSET = np.array([0.0, 0.0, 0.12])
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

demo_data = os.path.join(shared_dir, "object_6d", "demo_data")
C2R = load_latest_C2R()

# PICK_ORDER = ['brown_1', 'red_12', 'red_13', 'red_16', 'yellow_6', 'red_18', 'yellow_10', 'red_19', 'yellow_9', 'yellow_11', 'brown_4', 'yellow_8', 'brown_2', 'red_15', 'brown_3', 'yellow_5', 'red_14']
PICK_ORDER = ['brown_1', 'red_12', 'red_13', 'red_14', 'brown_0', 'yellow_6', 'red_16', 'yellow_5', 'yellow_8', 'yellow_9', 'red_18', 'yellow_10', 'yellow_11', 'brown_2', 'yellow_7', 'red_19', 'brown_3', "red_15", "red_17", "brown_4"]
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
OBSTACLE = {'cuboid': 
                { # xyz, quaternion
                 'table': {'dims': [2, 2, 0.2], 'pose': [0,0,-0.12,0,0,0,1]}, 
                 # 'baseback': {'dims': [2.0, 0.5, 2.0], 'pose': [-1.0857807924568896, -0.011288158965621076, -0.015956839973832793, 0.7082079218969054, -0.00040869377511796283, -0.006448498134229638, 0.7059743544943244]}, 
                 'basetop': {'dims': [5.0, 5.0, 0.2], 'pose': [0, 0, 1.0, 0, 0, 0, 1]}, 
                 'shelf0': {'dims': [0.8, 0.33, 2.02], 'pose': [-0.68+0.33/2, -0.6+0.8/2, -0.76, 0.70710678, 0, 0, 0.70710678]}, 
                 'shelf1': {'dims': [0.8, 0.03, 2.06], 'pose': [-0.68+0.03/2+0.33, -0.6+0.8/2, -0.75, 0.70710678, 0, 0, 0.70710678]}, # + 0.1
                 'shelf2': {'dims': [0.8, 0.1, 1.0], 'pose': [-0.68-0.1/2, -0.6+0.8/2, 0.2541, 0.70710678, 0, 0, 0.70710678]},  # + 1.0141
                 'shelf3': {'dims': [0.8, 0.33, 0.05], 'pose': [-0.68+0.33/2, -0.6+0.8/2, 0.574, 0.70710678, 0, 0, 0.70710678]},  #+ 1.334
                 # 'table': {'dims': [5.0, 5.0, 5.0], 'pose': [-0.07808975157119691, -0.5062144110803565, -2.584682669305668, 0.6999402146008835, 0.004682160214565101, -0.0007793753508808123, -0.7141856662901159]}}
                 }
            }

# TODO:
# [X] make floor
# [X] add object to OBSTACLE
# [X] change linear path planning to planning
# [] make order of pick

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
            if f"{obj_type}_{obj_idx}" in PICK_ORDER[10:]:
                obj_se3 = np.linalg.inv(C2R) @ obj_se3 @ ramen_offset[obj_type]

                obj_T[f"{obj_type}_{obj_idx}"] = obj_se3
            obj_idx += 1

    return obj_T

# shared_data/object_6d/demo_data/myrobot
def load_grasp_traj():
    demo_path = os.path.join(demo_data, "ramen")
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

def merge_qpos(xarm, inspire_qpos):
    if len(xarm.shape) == 1:
        xarm = np.repeat(xarm[None, :], repeats=inspire_qpos.shape[0], axis=0)
    if len(inspire_qpos.shape) == 1:
        inspire_qpos = np.repeat(inspire_qpos[None, :], repeats=xarm.shape[0], axis=0)
    return np.concatenate([xarm, inspire_qpos], axis=1)


robot = RobotWrapper(get_robot_urdf_path(arm_name="xarm", hand_name=None))
def linear_trajectory(init_qpos, target_se3, length=50):

    robot.compute_forward_kinematics(init_qpos)
    init_xarm_se3 = robot.get_link_pose(robot.get_link_index("link6"))
    
    xarm_se3_traj, _ = get_linear_path(init_xarm_se3, target_se3, np.zeros(6), np.zeros(6), length=length)
    xarm_qpos_traj = [init_qpos]
    for xarm_se3 in xarm_se3_traj:
        qpos, succ = robot.solve_ik(xarm_se3, "link6", xarm_qpos_traj[-1])
        if not succ:
            qpos = xarm_qpos_traj[-1]
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

def get_move_traj(init_qpos, target_qpos, length=50):
    qpos_traj = []
    for i in range(length):
        ratio = (i+1) / length
        qpos = init_qpos * (1-ratio) + target_qpos * ratio
        qpos_traj.append(qpos)
    qpos_traj = np.array(qpos_traj)
    return qpos_traj

def extract_approach_trajectory(traj, threshold=0.001, min_static_frames=10):
    """
    Extract approach trajectory by finding where xarm joints stop moving
    
    Args:
        traj: Full trajectory array (N, 12) where first 6 columns are xarm joints
        threshold: Maximum joint movement to consider as "static" (radians)
        min_static_frames: Minimum number of consecutive static frames to detect stop
    
    Returns:
        approach_traj: Extracted approach trajectory
        split_idx: Index where approach ends
    """
    xarm_traj = traj[:, :6]
    
    # Calculate joint velocities (differences between consecutive frames)
    joint_diffs = np.abs(np.diff(xarm_traj, axis=0))
    
    # Check if all joints are below threshold (robot is static)
    is_static = np.all(joint_diffs < threshold, axis=1)
    
    # Find first occurrence of min_static_frames consecutive static frames
    static_count = 0
    split_idx = None
    
    for i in range(len(is_static)):
        if is_static[i]:
            static_count += 1
            if static_count >= min_static_frames:
                # Go back to the start of static period
                split_idx = i - min_static_frames + 2  # +2 because diff reduces length by 1
                break
        else:
            static_count = 0
    
    # If no static period found, return full trajectory
    if split_idx is None:
        print("Warning: No static period found. Returning full trajectory.")
        return traj, len(traj)
    
    approach_traj = traj[:split_idx, :6]
    return approach_traj
    

pick_position = load_pick_position()

###### Load grasp policy #######
grasp_policy_dict = load_grasp_traj()
grasp_idx = "7"
inspire_traj = grasp_policy_dict[grasp_idx][1].copy()

# start from zero state
inspire_traj_pre = []
for i in range(15):
    inspire_traj_pre.append(inspire_traj[0] * i / 15 + np.array([1000, 1000, 1000, 1000, 1000 ,1000]) * (1 - i / 15))
inspire_traj = np.concatenate([np.array(inspire_traj_pre), inspire_traj], axis=0)
grasp_se3 = grasp_policy_dict[grasp_idx][0]
####################################

for step in range(10):# len(PICK_ORDER)):
    tot_traj = []
    # approach
    pick_traj = np.load(os.path.join("data", "pick_traj", f"{step+10}.npy"))
    
    approach_xarm_traj = extract_approach_trajectory(pick_traj, min_static_frames=inspire_traj.shape[0] - 20)
    approach_traj = merge_qpos(approach_xarm_traj, inspire_traj[0])
    os.makedirs(os.path.join("data", "approach"), exist_ok=True)
    np.save(os.path.join("data", "approach", f"{step}.npy"), approach_xarm_traj)

    grasp_traj = merge_qpos(approach_xarm_traj[-1, :6], inspire_traj[::3])

    lift_xarm_traj = get_lift_traj(approach_traj[-1, :6], height=0.2, length=50)
    lift_traj = merge_qpos(lift_xarm_traj, inspire_traj[-1])

    place_traj = np.load(os.path.join("data", "place_traj", f"{step}.npy"))
    move_xarm_traj = get_move_traj(lift_xarm_traj[-1, :6], place_traj[0, :6], length=100)
    move_traj = merge_qpos(move_xarm_traj, inspire_traj[-1])

    
    tot_traj = np.concatenate([approach_traj, grasp_traj, lift_traj, move_traj], axis=0)
    os.makedirs(os.path.join("data", "refine_pick_traj"), exist_ok=True)
    np.save(os.path.join("data", "refine_pick_traj", f"{step}.npy"), tot_traj)
