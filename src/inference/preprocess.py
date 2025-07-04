import os
import time
import pinocchio as pin
from paradex.simulator.isaac import simulator
from paradex.utils.file_io import rsc_path
import numpy as np
from scipy.spatial.transform import Rotation as R
import pickle
import transforms3d as t3d  
from paradex.robot.robot_wrapper import RobotWrapper
from paradex.utils.metric import get_pickplace_timing, compute_mesh_to_ground_distance
import open3d as o3d

demo_path = "data_Icra/teleoperation/bottle"
demo_path_list = os.listdir(demo_path)

robot = RobotWrapper(
    os.path.join(rsc_path, "xarm6", "xarm6_allegro_wrist_mounted_rotate.urdf")
)
link_index = robot.get_link_index("palm_link")

obj_mesh = o3d.io.read_triangle_mesh(os.path.join(rsc_path, "bottle", "bottle.obj"))
os.makedirs("data_Icra/pose/bottle", exist_ok=True)

for demo_name in demo_path_list:
    obj_T = pickle.load(open(os.path.join(demo_path, demo_name, "obj_traj.pickle"), "rb"))['bottle']
    robot_traj = np.load(os.path.join(demo_path, demo_name, "robot_qpos.npy"))
    target_traj = np.load(os.path.join(demo_path, demo_name, "target_qpos.npy"))

    T = obj_T.shape[0]

    height_list = []
    for step in range(T):
        h = compute_mesh_to_ground_distance(obj_T[step], obj_mesh)    
        height_list.append(h)
    pick, place = get_pickplace_timing(height_list)
    t = (pick + place) // 2

    robot_pose = robot_traj[t]
    robot.compute_forward_kinematics(robot_pose)
    wrist_pose = robot.get_link_pose(link_index)

    obj_pos = np.linalg.inv(wrist_pose) @ obj_T[t]
    np.save(os.path.join("data_Icra/pose/bottle", f"{demo_name}.npy"), obj_pos)
 