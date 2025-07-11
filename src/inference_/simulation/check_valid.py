import os
import time
import pinocchio as pin
from paradex.simulator.isaac import simulator
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

demo_path = "data_Icra/teleoperation/bottle"
demo_path_list = os.listdir(demo_path)
demo_path_list.sort()
obj_mesh = o3d.io.read_triangle_mesh(os.path.join(rsc_path, "bottle", "bottle.obj"))

grasp_T = 50
lift_T = 100
approach_T = 50

for demo_name in demo_path_list:
    if demo_name < "81":
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
    sim.set_savepath(f"video/{demo_name}.mp4", f"state/{demo_name}.pickle")

    obj_T = pickle.load(open(os.path.join(demo_path, demo_name, "obj_traj.pickle"), "rb"))['bottle']
    robot_traj = np.load(os.path.join(demo_path, demo_name, "robot_qpos.npy"))
    target_traj = np.load(os.path.join(demo_path, demo_name, "target_qpos.npy"))

    dist = 0.07

    T = obj_T.shape[0]
    height_list = []

    pre_grasp_pose = None
    start_T = -1

    for step in range(T):
        h = compute_mesh_to_ground_distance(obj_T[step], obj_mesh)    
        height_list.append(h)

    for step in range(T):
        robot_pose = robot_traj[step]
        robot.compute_forward_kinematics(robot_pose)
        wrist_pose = robot.get_link_pose(link_index)
    
        obj_wrist_pose = np.linalg.inv(obj_T[step]) @ wrist_pose
        obj_pos = obj_wrist_pose[:2, 3]

        d = np.linalg.norm(obj_pos[:2])
        if d < dist:
            pre_grasp_pose = robot_pose[6:]
            start_T = step
            break

    pick, place = get_pickplace_timing(height_list)
    
    robot_pose = robot_traj[pick]
    robot.compute_forward_kinematics(robot_pose)
    wrist_pose_pick = robot.get_link_pose(link_index)


    obj_pose = obj_T[start_T].copy()
    # obj_pose[:2, 3] = obj_T[pick, :2, 3] # move to same x, y

    wrist_pose = wrist_pose_pick.copy()
    wrist_pose[:2, 3] -= 0.01

    squeeze_grasp_pose = target_traj[pick, 6:]
    grasp_pose = robot_traj[pick, 6:]
    
    # approach
    v = wrist_pose[:3, 0]
    wrist_start_pose = wrist_pose.copy()
    wrist_start_pose[:3, 3] -= v * dist

    # q_pos_init = robot.solve_ik(wrist_pose, "palm_link", q_init=robot_pose)
    last_qpos = robot.solve_ik(wrist_start_pose, "palm_link", q_init=robot_pose, max_iter=50000, alpha=0.01)
    # approach_T = (pick - start_T) // 2
    # grasp_T = (pick - start_T) // 2

    for i in range(approach_T):
        cur_wrist_pose = wrist_start_pose.copy()
        cur_wrist_pose[:3, 3] += v * (i / approach_T) * dist
        last_qpos = robot.solve_ik(cur_wrist_pose, "palm_link", q_init=last_qpos)

        q_pos = last_qpos.copy()
        q_pos_vis = last_qpos.copy()

        # q_pos[6:] = np.zeros(16) * (1 - i / grasp_T) + pre_grasp_pose * (i / grasp_T)
        q_pos[6:] = target_traj[max(0, start_T-approach_T+i+1), 6:]
        q_pos_vis[6:] = pre_grasp_pose
        # q_pos_vis = robot_traj[pick-approach_T-grasp_T+i+1, :]
        sim.step(q_pos, q_pos_vis, obj_pose)

    # grasp
    for i in range(grasp_T):
        last_qpos = robot.solve_ik(wrist_pose, "palm_link", q_init=last_qpos)
        q_pos = last_qpos.copy()
        q_pos_vis = last_qpos.copy()

        # q_pos[6:] = pre_grasp_pose * (1 - i / grasp_T) + squeeze_grasp_pose * (i / grasp_T)
        q_pos[6:] = target_traj[max(start_T, pick-grasp_T+i+1), 6:]
        # q_pos_vis = robot_traj[pick-grasp_T+i+1, :]
        q_pos_vis[6:] = squeeze_grasp_pose
        sim.step(q_pos, q_pos_vis, obj_pose)


    # grasp
    # for i in range(grasp_T):
    #     q_pos = last_qpos.copy()
    #     q_pos[6:] = np.zeros(16) * (1 - i / grasp_T) + squeeze_grasp_pose * (i / grasp_T)
        
    #     q_pos_vis = last_qpos.copy()
    #     q_pos_vis[6:] = np.zeros(16) * (1 - i / grasp_T) + grasp_pose * (i / grasp_T)

    #     sim.step(q_pos, q_pos_vis, obj_pose)

    # last_qpos = q_pos_init.copy()
    wrist_pose_lift = wrist_pose.copy()

    for i in range(lift_T):
        wrist_pose_lift[2, 3] += 0.001

        last_qpos = robot.solve_ik(wrist_pose_lift, "palm_link", q_init=last_qpos)

        q_pos = last_qpos.copy()
        q_pos_vis = last_qpos.copy()

        q_pos[6:] = squeeze_grasp_pose
        q_pos_vis[6:] = grasp_pose

        obj_pose[2, 3] += 0.001

        sim.step(q_pos, q_pos_vis, obj_pose)
    sim.terminate()