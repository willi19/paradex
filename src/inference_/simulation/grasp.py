import os
import time
import pinocchio as pin
from paradex.simulator.isaac_deprecated import simulator
from paradex.utils.file_io import rsc_path
import numpy as np
from scipy.spatial.transform import Rotation as R
import pickle
from paradex.utils.metric import get_pickplace_timing, compute_mesh_to_ground_distance
from paradex.robot import RobotWrapper
import open3d as o3d


demo_path = "data_Icra/teleoperation/bottle"
demo_path_list = os.listdir(demo_path)
demo_path_list.sort()

grasp_T = 50
lift_T = 100
approach_T = 50

def load_demo(demo_name):
    obj_T = pickle.load(open(os.path.join(demo_path, demo_name, "obj_traj.pickle"), "rb"))['bottle']
    robot_traj = np.load(os.path.join(demo_path, demo_name, "robot_qpos.npy"))
    target_traj = np.load(os.path.join(demo_path, demo_name, "target_qpos.npy"))

    T = obj_T.shape[0]

    height_list = []

    for step in range(T):
        h = compute_mesh_to_ground_distance(obj_T[step], obj_mesh)    
        height_list.append(h)

    end_T, _ = get_pickplace_timing(height_list)
    
    dist = 0.07
    for step in range(T):
        robot_pose = robot_traj[step]
        robot.compute_forward_kinematics(robot_pose)
        wrist_pose = robot.get_link_pose(link_index)
    
        obj_wrist_pose = np.linalg.inv(obj_T[step]) @ wrist_pose
        obj_pos = obj_wrist_pose[:2, 3]

        d = np.linalg.norm(obj_pos[:2])
        if d < dist:
            start_T = step
            break

    robot_pose = []
    hand_action = []

    for i in range(start_T, end_T + 1 + lift_T):
        if i > end_T:
            robot_pose.append(robot_pose[-1].copy())
            robot_pose[-1][2, 3] += 0.001
            hand_action.append(target_traj[end_T, 6:])
            continue

        robot.compute_forward_kinematics(robot_traj[i])
        wrist_pose = robot.get_link_pose(link_index)

        hand_action.append(target_traj[i, 6:])
        robot_pose.append(wrist_pose)
    
    robot_pose = np.array(robot_pose)
    hand_action = np.array(hand_action)
    obj_pose = obj_T[start_T].copy()
    
    tx, ty = obj_pose[:2, 3]

    desired_x = np.array([0, 1, 0])
    current_x = robot_pose[-1, :3, 0][:2]  # x축 방향 (2D)
    theta = np.arctan2(desired_x[1], desired_x[0]) - np.arctan2(current_x[1], current_x[0])

    rot_mat = np.array([
        [np.cos(theta), -np.sin(theta), 0, 0],
        [np.sin(theta),  np.cos(theta), 0, 0],
        [0,              0,             1, 0],
        [0,              0,             0, 1],
    ])
    print(tx, ty)
    obj_pose[:2, 3] -= np.array([tx, ty])
    robot_pose[:, :2, 3] -= np.array([tx, ty])
    robot_pose[:, 2, 3] += (0.1 - robot_pose[end_T-start_T+1, 2, 3])

    robot_pose = np.einsum('ij, kjl -> kil', rot_mat, robot_pose)
    return obj_pose, robot_pose, hand_action

# def load_recon_demo(index):
#     obj_T = pickle.load(open(os.path.join(demo_path, demo_name, "obj_traj.pickle"), "rb"))['bottle']
#     robot_traj = np.load(os.path.join(demo_path, demo_name, "robot_qpos.npy"))
#     target_traj = np.load(os.path.join(demo_path, demo_name, "target_qpos.npy"))

#     dist = 0.07

#     T = obj_T.shape[0]
    
#     height_list = []
#     start_T = -1

#     for step in range(T):
#         h = compute_mesh_to_ground_distance(obj_T[step], obj_mesh)    
#         height_list.append(h)

#     pick, place = get_pickplace_timing(height_list)
    
#     for step in range(T):
#         robot_pose = robot_traj[step]
#         robot.compute_forward_kinematics(robot_pose)
#         wrist_pose = robot.get_link_pose(link_index)
    
#         obj_wrist_pose = np.linalg.inv(obj_T[step]) @ wrist_pose
#         obj_pos = obj_wrist_pose[:2, 3]

#         d = np.linalg.norm(obj_pos[:2])
#         if d < dist:
#             pre_grasp_pose = robot_pose[6:]
#             start_T = step
#             break

    
#     robot_pose = robot_traj[pick]
#     robot.compute_forward_kinematics(robot_pose)
#     wrist_pose_pick = robot.get_link_pose(link_index)


#     obj_pose = obj_T[start_T].copy()
#     # obj_pose[:2, 3] = obj_T[pick, :2, 3] # move to same x, y

#     wrist_pose = wrist_pose_pick.copy()
#     wrist_pose[:2, 3] -= 0.01

#     squeeze_grasp_pose = target_traj[pick, 6:]
#     grasp_pose = robot_traj[pick, 6:]
                


# Viewer setting
obj_name = "bottle"
save_video = False
save_state = True
view_physics = True
view_replay = True
headless = False


robot = RobotWrapper(
    os.path.join(rsc_path, "xarm6", "xarm6_allegro_wrist_mounted_rotate.urdf")
)
link_index = robot.get_link_index("palm_link")

obj_mesh = o3d.io.read_triangle_mesh(os.path.join(rsc_path, "bottle", "bottle.obj"))

def get_traj(tx, ty, theta, obj_pose, wrist_pose_demo, hand_pose):
    rot_mat = R.from_euler('z', theta, degrees=True).as_matrix()
    rot_T = np.eye(4)
    rot_T[:3, :3] = rot_mat

    wrist_pose = wrist_pose_demo.copy()
    wrist_pose = np.einsum('ij, kjl -> kil', rot_T, wrist_pose)
    wrist_pose[:, :2, 3] += np.array([tx, ty]) / 100

    obj_pose[:2, 3] = np.array([tx, ty]) / 100


    T = wrist_pose.shape[0]
    q_pose_list = []
    last_qpos = np.zeros(22)

    traj_success = True

    for i in range(T):
        last_qpos, success = robot.solve_ik(
            wrist_pose[i],
            "palm_link",
            q_init=last_qpos
        )
        if not success:
            traj_success = False
            break

        q_pose_list.append(last_qpos)
        q_pose_list[-1][6:] = hand_pose[i]
    
    if not traj_success:
        return None, None
    
    q_pose_list = np.array(q_pose_list)
    return q_pose_list, obj_pose

for demo_name in demo_path_list:
    # obj_pose_demo, wrist_pose_demo, hand_pose = load_demo(demo_name)
    
    traj_list_demo = pickle.load(open(os.path.join("grasp", "policy", f"{demo_name}.pickle"), "rb"))
    traj_list_demo = traj_list_demo[:1]
    print(traj_list_demo[0]['theta'])
    sim = simulator(
        obj_name,
        view_physics,
        view_replay,
        headless,
        save_video,
        save_state,
        fixed=True,
        num_envs=len(traj_list_demo)
    )
    
    sim.set_savepath(None, f"grasp/state/{demo_name}.pickle")
    
    step = 0
    while True:
        q_pose_list = []
        viz_q_pose_list = []
        obj_pose_list = []

        finished = True

        for i in range(len(traj_list_demo)):
            traj_info = traj_list_demo[i]
            if step >= traj_info['q_pose'].shape[0]:
                q_pose_list.append(traj_info['q_pose'][-1])
                viz_q_pose_list.append(traj_info['q_pose'][-1].copy())
                continue

            finished = False
            q_pose_list.append(traj_info['q_pose'][step])
            viz_q_pose_list.append(traj_info['q_pose'][step].copy())
            obj_pose_list.append(traj_info['obj_pose'].copy())

        if finished:
            break

        sim.step(q_pose_list, viz_q_pose_list, obj_pose_list)
        step += 1

    sim.terminate()
