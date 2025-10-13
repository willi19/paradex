import numpy as np
import pickle
import os
import open3d as o3d
from curobo.wrap.reacher.motion_gen import MotionGen, MotionGenConfig, MotionGenPlanConfig, PoseCostMetric
from curobo.geom.types import WorldConfig
from curobo.types.base import TensorDeviceType
import trimesh
# Pick Target 

from paradex.utils.file_io import find_latest_directory, shared_dir, home_path, load_yaml, load_latest_C2R, get_robot_urdf_path, rsc_path
from paradex.robot.curobo import CuroboPlanner
from paradex.visualization.visualizer.viser import ViserViewer
from paradex.inference.util import get_linear_path
from paradex.robot.robot_wrapper import RobotWrapper
from paradex.robot.mimic_joint import parse_inspire

xarm_init_pose = np.array([-0.8048279285430908, 0.2773207128047943, -1.4464116096496582, 2.0092501640319824, 0.7059974074363708, -2.361839532852173]) # Initial xarm pose for every grasp 

place_origin = np.array([-0.55, -0.45, 0.251 + 0.1]) # 25cm : floor, 10cm: ramen height

Z_OFFSET = np.array([0.0, 0.0, 0.10])
Z_NUM = 2

X_OFFSET = np.array([0.13, 0.0, 0.0])
X_NUM = 2

Y_OFFSET = np.array([0.0, 0.13, 0.0])
Y_NUM = 5

LINEAR_START = 0.4
LINEAR_DIRECTION = np.array([0.0, 1.0, 0.0])

C2R = load_latest_C2R()

# We use object coordinate as it's center is in the bottom, middle of the object, with z-axis pointing up
# However center of ramen mesh is not as it is so we need to adjust it
ramen_offset = {
    "brown":np.array([[1, 0, 0, 0], 
                      [0, 0, 1, -0.055], 
                      [0, -1, 0, 0], 
                      [0, 0, 0, 1]]),
    "red":None,
    "yellow":None,
}

def get_place_position(index):
    """ Get the place position based on the index 
        ORDER: Y -> X -> Z
    """
    z_index = index % Z_NUM
    x_index = (index // Z_NUM) % X_NUM
    y_index = (index // (Z_NUM * X_NUM)) % Y_NUM
    place_position = place_origin + z_index * Z_OFFSET + x_index * X_OFFSET + y_index * Y_OFFSET
    return place_position

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
            obj_T[f"{obj_type}_{obj_idx}"] = np.linalg.inv(C2R) @ obj_se3 @ ramen_offset[obj_type]
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

def get_pick_traj(obj_6D, grasp_pose):
    # Special normalization for ramen
    obj_6D[:3, :3] = np.eye(3)

def load_planner():
    cfg_path = os.path.join(home_path, "robothome", "robothome_collision/1001_rlwrld_collision_v8_grasp2/robothome_scene_collision_rlwrld.yaml")
    # print(load_yaml(cfg_path))
    # import pdb; pdb.set_trace()

    env_cfg = WorldConfig.from_dict(load_yaml(cfg_path))

    world_cfg = WorldConfig(
        cuboid=env_cfg.cuboid, #[boxs_obstacle, boxs_obstacle2],
        # sphere=[sphere_obstacle],
        cylinder=env_cfg.cylinder,
    )
    robot_cfg = load_yaml(os.path.join(home_path, "robothome", "curobo_rsc", "content/assets/myrobot/xarm_inspire.yml"))["robot_cfg"]
    tensor_args = TensorDeviceType()

    return CuroboPlanner(world_cfg, robot_cfg, tensor_args)

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
    return visualizer

def get_grasp_pose():
    pass

def get_linear_start_position(theta, object_position, ):
    pass

pick_position = load_pick_position()
grasp_policy = load_pick_traj()
print(pick_position)

planner = load_planner()
visualizer = load_visualizer(pick_position)

robot = RobotWrapper(get_robot_urdf_path(arm_name="xarm", hand_name=None))
robot.compute_forward_kinematics(xarm_init_pose)
init_xarm_se3 = robot.get_link_pose(robot.get_link_index("link6"))

for obj_name in pick_position.keys():
    # approach
    pick_start_pose = init_xarm_se3
    target_obj_pose = pick_position[obj_name]
    target_obj_pose[:3, :3] = np.eye(3) # Special normalization for ramen
    
    pick_end_pose = target_obj_pose @ grasp_policy["7"][0]
    
    pick_xarm_traj, pick_inspire_traj = get_linear_path(pick_start_pose, pick_end_pose, np.zeros(6), np.zeros(6), length=50)
    pick_xarm_qpos = [xarm_init_pose]
    for pick_xarm in pick_xarm_traj:
        qpos, succ = robot.solve_ik(pick_xarm, "link6", pick_xarm_qpos[-1])
        print(succ)
        if False:# not succ:
            qpos = pick_xarm_qpos[-1]
        pick_xarm_qpos.append(qpos)
    pick_xarm_qpos = np.array(pick_xarm_qpos)[1:]

    pick_traj = np.concatenate([pick_xarm_qpos, pick_inspire_traj], axis=1)
    visualizer.add_traj(f"pick_{obj_name}", {"xarm":pick_traj})
    # grasp
    inspire_traj = parse_inspire(grasp_policy["7"][1], joint_order = ['right_thumb_1_joint', 'right_thumb_2_joint', 'right_index_1_joint', 'right_middle_1_joint', 'right_ring_1_joint', 'right_little_1_joint', ])[::3]
    xarm_traj = np.repeat(pick_xarm_qpos[-1][None, :], repeats=inspire_traj.shape[0], axis=0)
    grasp_traj = np.concatenate([xarm_traj, inspire_traj], axis=1)
    visualizer.add_traj(f"grasp_{obj_name}", {"xarm":grasp_traj})

    # lift
    target_pose = pick_end_pose.copy()
    target_pose[2, 3] += 0.2
    
    lift_xarm_traj, _ = get_linear_path(pick_end_pose, target_pose, np.zeros(6), np.zeros(6), length=50)
    lift_xarm_qpos = [pick_xarm_qpos[-1]]
    lift_obj_pose = []

    for lift_xarm in lift_xarm_traj:
        qpos, succ = robot.solve_ik(lift_xarm, "link6", lift_xarm_qpos[-1])
        lift_xarm_qpos.append(qpos)
        lift_obj_pose.append(lift_xarm @ np.linalg.inv(grasp_policy["7"][0]))

    lift_xarm_qpos = np.array(lift_xarm_qpos)[1:]
    lift_inspire_traj = np.repeat(inspire_traj[-1][None, :], repeats=lift_xarm_qpos.shape[0], axis=0)
    lift_traj = np.concatenate([lift_xarm_qpos, lift_inspire_traj], axis=1)

    visualizer.add_traj(f"lift_{obj_name}", {"xarm":lift_traj}, {obj_name:np.array(lift_obj_pose)})

    # move to linear path initial
    wrist_direction = lift_obj_pose[-1][:3, 2]
    
    # move to place position linearly
    
    # lay down

    # retreat

    # back to init

visualizer.start_viewer()