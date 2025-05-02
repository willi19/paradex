import os
import time
import pinocchio as pin
from paradex.simulator.isaac import simulator
from paradex.utils.file_io import shared_path, load_c2r
import numpy as np
from scipy.spatial.transform import Rotation as R
import pickle

# Viewer setting
obj_name = "clock"
save_video = False
save_state = False
view_physics = False
view_replay = True
headless = False

simulator = simulator(
    obj_name,
    view_physics,
    view_replay,
    headless,
    save_video,
    save_state,
    fixed=True
)

demo_path = f"{shared_path}/processed/{obj_name}"
demo_path_list = os.listdir(demo_path)

dof_names_list = simulator.get_dof_names()

for demo_name in ["0"]:
    hand_pose = np.load(os.path.join(demo_path, demo_name, 'hand', 'state.npy'))
    arm_pose = np.load(os.path.join(demo_path, demo_name, 'arm', 'state.npy'))
    T = hand_pose.shape[0]

    c2r = load_c2r(os.path.join(demo_path, demo_name))
    
    robot_action = np.zeros((T, 22))
    
    robot_action[:, 6:] = hand_pose
    robot_action[:, :6] = arm_pose

    obj_traj = pickle.load(open(os.path.join(demo_path, demo_name, 'object_tracking', 'trajectory.pickle'), 'rb'))
    for step in range(T):
        obj_T = np.linalg.inv(c2r) @ obj_traj[step]
        simulator.step(robot_action[step], robot_action[step], obj_T)#robot_traj[step], obj_traj[step])
        

