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
view_physics = True
view_replay = True
headless = False

simulator = simulator(
    obj_name,
    view_physics,
    view_replay,
    headless,
    save_video,
    save_state,
    fixed=False
)

demo_path = f"{shared_path}/processed/{obj_name}"
demo_path_list = os.listdir(demo_path)

dof_names_list = simulator.get_dof_names()
learning_rate = 0.05#[0.01, 0.05, 0.1, 0.5]
for demo_name in ["0"]:
    wrist_T = np.load(os.path.join(shared_path, 'contact_map', obj_name, demo_name, "wrist_T.npy"))
    obj_T = np.load(os.path.join(shared_path, 'contact_map', obj_name, demo_name, "object_pose.npy"))
    
    logs_basedir = os.path.join(shared_path, "inference", f'{obj_name}', f'{demo_name}', 'grasp_pose')
    q_target = np.load(os.path.join(logs_basedir, f'3_{int(learning_rate*1000)}_squeeze.npy'))
    q_pose = np.load(os.path.join(logs_basedir, f'3.npy'))
    q_pregrasp = 2 * q_pose - q_target
    
    T = 100

    c2r = load_c2r(os.path.join(demo_path, demo_name))
    # obj_T = np.linalg.inv(c2r) @ obj_T
    
    robot_action = np.zeros((22))
    
    robot_action[:3] = wrist_T[:3, 3]
    robot_action[3:6] = R.from_matrix(wrist_T[:3, :3]).as_euler('XYZ')

    robot_pose = np.zeros((22))
    robot_pose[:3] = wrist_T[:3, 3]
    robot_pose[3:6] = R.from_matrix(wrist_T[:3, :3]).as_euler('XYZ')

    for step in range(200):
        simulator.step(robot_pose, robot_action, obj_T)
        

    for step in range(T):
        action = (q_pregrasp * (1 - step / T) 
                 + q_target * (step / T))
        robot_action[6:] = action
        robot_pose[6:] = action
        if step >= T // 2:
            robot_pose[6:] = q_pose

        simulator.step(robot_action, robot_pose, obj_T)#robot_traj[step], obj_traj[step])
        
    for step in range(200):
        robot_action[2] += 0.001
        robot_pose[2] += 0.001
        simulator.step(robot_action, robot_pose, obj_T)

