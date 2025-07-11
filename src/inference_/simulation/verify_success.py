import os
import pinocchio as pin
from paradex.utils.file_io import rsc_path
import numpy as np
from scipy.spatial.transform import Rotation as R
import pickle
from paradex.utils.metric import get_pickplace_timing, compute_mesh_to_ground_distance
from paradex.robot import RobotWrapper
import open3d as o3d

# Viewer setting
obj_name = "bottle"
save_video = True
save_state = True
view_physics = True
view_replay = False
headless = False


robot = RobotWrapper(
    os.path.join(rsc_path, "xarm6", "xarm6_allegro_wrist_mounted_rotate.urdf")
)
link_index = robot.get_link_index("palm_link")

demo_path = "state"
demo_path_list = os.listdir(demo_path)
demo_path_list.sort()
obj_mesh = o3d.io.read_triangle_mesh(os.path.join(rsc_path, "bottle", "bottle.obj"))

for demo_name in demo_path_list:
    traj = pickle.load(open(os.path.join(demo_path, demo_name), "rb"))
    
    robot_traj = traj['robot']
    obj_traj = traj['object']

    T = len(robot_traj)

    h_last = compute_mesh_to_ground_distance(obj_traj[-1], obj_mesh)    
    h_init = compute_mesh_to_ground_distance(obj_traj[0], obj_mesh)

    if h_last - h_init > 0.05:
        pass
        # print(f"Demo {demo_name} is successful, height change: {h_last - h_init:.2f} m")
    else:
        print(f"Demo {demo_name} is not successful, height change: {h_last - h_init:.2f} m")