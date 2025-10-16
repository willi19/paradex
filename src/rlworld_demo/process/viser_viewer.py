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
obj_color = ["red", "red", "red", "red", "yellow", "yellow", "yellow", "yellow", "brown", "brown"]
new_obj_order = [f"ramen_{i}" for i in range(len(PICK_ORDER)-10)]

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
    latest_dir = '20251014-233540'#find_latest_directory(obj_6d_path)
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

    visualizer.add_floor(height=-0.02)
    visualizer.add_robot("xarm", get_robot_urdf_path(arm_name="xarm", hand_name="inspire"))

    mesh_dict = {}
    for color in ["brown", "red", "yellow"]:
        mesh_path = os.path.join(rsc_path, "object", f"{color}_ramen_von", f"{color}_ramen_von.obj")
        mesh = trimesh.load(mesh_path)
        mesh_dict[color] = mesh

    for oi, (obj_name, obj_pose) in enumerate(pick_position.items()):
        pick_index = PICK_ORDER[10:].index(obj_name)
        color = obj_color[pick_index]
        print(color, obj_name)
        visualizer.add_object(obj_name, mesh_dict[color], obj_pose)

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

######################################################## utils ########################################################
def get_obj_traj(qpos_traj, grasp_se3):
    obj_pose = []
    for qpos in qpos_traj:
        robot.compute_forward_kinematics(qpos)
        xarm_se3 = robot.get_link_pose(robot.get_link_index("link6"))
        obj_se3 = xarm_se3 @ np.linalg.inv(grasp_se3)
        obj_pose.append(obj_se3)

    return obj_pose

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

#########################################################################################################################

pick_position = load_pick_position()
visualizer = load_visualizer(pick_position)

###### Get grasp pose ######
grasp_policy_dict = load_pick_traj()
grasp_idx = "7"
grasp_se3 = grasp_policy_dict[grasp_idx][0]
###########################

for step in range(10):
    obj_name = PICK_ORDER[10+step]
    pick_traj = np.load(os.path.join("data", "refine_pick_traj", f"{step}.npy"))
    pick_traj[:, 6:] = parse_inspire(pick_traj[:,6:], joint_order = ['right_thumb_1_joint', 'right_thumb_2_joint', 'right_index_1_joint', 'right_middle_1_joint', 'right_ring_1_joint', 'right_little_1_joint', ])

    place_traj = np.load(os.path.join("data", "place_traj", f"{step}.npy"))
    place_traj[:, 6:] = parse_inspire(place_traj[:,6:], joint_order = ['right_thumb_1_joint', 'right_thumb_2_joint', 'right_index_1_joint', 'right_middle_1_joint', 'right_ring_1_joint', 'right_little_1_joint', ])
    
    approach_xarm_traj = extract_approach_trajectory(pick_traj)
    approach_T = approach_xarm_traj.shape[0]

    put_xarm_traj = extract_approach_trajectory(place_traj)
    put_T = put_xarm_traj.shape[0]

    # approach
    visualizer.add_traj(f"approach_{step}", {"xarm":pick_traj[:approach_T]})
    
    # pick
    obj_traj = get_obj_traj(pick_traj[approach_T:, :6], grasp_se3)
    visualizer.add_traj(f"pick_obj_{step}", {"xarm":pick_traj[approach_T:]}, {f"{obj_name}":np.array(obj_traj)})

    # put
    obj_traj = get_obj_traj(place_traj[:put_T, :6], grasp_se3)
    visualizer.add_traj(f"put_{step}", {"xarm":place_traj[:put_T]}, {f"{obj_name}":np.array(obj_traj)})

    # release
    visualizer.add_traj(f"place_obj_{step}", {"xarm":place_traj[put_T:]})

visualizer.start_viewer()