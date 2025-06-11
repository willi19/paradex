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
from paradex.geometry.ik import inverse_kinematics

LINK62PALM = np.array(
    [
        [0, 1, 0, 0],
        [0, 0, 1, 0],
        [1, 0, 0, 0],
        [0, 0, 0, 1],
    ]
)


# Viewer setting
obj_name = "bottle"
save_video = True
save_state = True
view_physics = True
view_replay = False
headless = True

demo_path = "data_Icra/teleoperation/bottle"
demo_path_list = os.listdir(demo_path)
demo_path_list.sort()

robot = RobotWrapper(
    os.path.join(rsc_path, "xarm6", "xarm6_allegro_wrist_mounted_rotate.urdf")
)
link_index = robot.get_link_index("link6")

for demo_name in demo_path_list:  # demo_path_list:
    if demo_name < "85":
        continue
    sim = simulator(
        obj_name,
        view_physics,
        view_replay,
        headless,
        save_video,
        save_state,
        fixed=True
    )

    sim.load_camera()
    sim.set_savepath(f"replay/video/{demo_name}.mp4", f"replay/state/{demo_name}.pickle")

    obj_T = pickle.load(open(os.path.join(demo_path, demo_name, "obj_traj.pickle"), "rb"))['bottle']
    robot_traj = np.load(os.path.join(demo_path, demo_name, "robot_qpos.npy"))
    target_traj = np.load(os.path.join(demo_path, demo_name, "target_qpos.npy"))

    T = obj_T.shape[0]
    

    for step in range(T):
        robot_action = target_traj[step]
        
        t = robot_action[:3]
        axis_angle = robot_action[3:6]
        angle = np.linalg.norm(axis_angle)
        if angle > 1e-6:
            axis = axis_angle / angle
        else:
            axis = np.zeros(3)
            
        link6_mat = t3d.axangles.axangle2mat(axis, angle)
        R_mat = link6_mat @ LINK62PALM[:3,:3]
        e = R.from_matrix(R_mat).as_euler('XYZ')
        robot_action[3:6] = e

        robot_T = np.eye(4)
        robot_T[:3, :3] = link6_mat
        robot_T[:3, 3] = t

        robot_pose = robot_traj[step]
        robot.compute_forward_kinematics(robot_pose)
        R_mat_pose = robot.get_link_pose(link_index)

        # print(np.linalg.norm(R_mat_pose[:3, 3] - robot_action[:3]))

        q_ik = inverse_kinematics(
            robot,
            robot_T,
            "link6",
            q_init=robot_pose,
            max_iter=100,
            tol=1e-4,
            alpha=1e-1
        )

        robot.compute_forward_kinematics(q_ik)
        R_mat_ik = robot.get_link_pose(link_index)
        # print(np.linalg.norm(R_mat_ik[:3, 3] - R_mat_pose[:3, 3]), "error")
        # print(np.linalg.norm(R_mat_ik[:3, 3] - robot_action[:3]), "IK error")

        q_ik[:6] = robot_pose[:6]  # Set the first 6 joints to the action
        q_ik[6:] = robot_action[6:]  # Set the wrist joint angles
        
        sim.step(q_ik, robot_pose, obj_T[step])#robot_traj[step], obj_traj[step])
    sim.terminate()
        