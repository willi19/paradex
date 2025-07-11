import os
from paradex.utils.file_io import rsc_path
import numpy as np
from paradex.robot import RobotWrapper
import pickle
import open3d as o3d
from paradex.utils.metric import get_pickplace_timing, compute_mesh_to_ground_distance

obj_mesh = o3d.io.read_triangle_mesh(os.path.join(rsc_path, "object", "bottle", "bottle.obj"))

def load_demo(demo_path, demo_name):
    prev_urdf_file = os.path.join(rsc_path, "robot", "xarm", "xarm6_allegro_wrist_mounted_rotate_prev.urdf")
    robot = RobotWrapper(prev_urdf_file)

    palm_id = robot.get_link_index("palm_link")

    obj_T = pickle.load(open(os.path.join(demo_path, demo_name, "obj_traj.pickle"), "rb"))['bottle']
    robot_traj = np.load(os.path.join(demo_path, demo_name, "robot_qpos.npy"))
    target_traj = np.load(os.path.join(demo_path, demo_name, "target_qpos.npy"))

    T = obj_T.shape[0]

    height_list = []

    for step in range(T):
        h = compute_mesh_to_ground_distance(obj_T[step], obj_mesh)    
        height_list.append(h)

    t_pick_end, t_place_start = get_pickplace_timing(height_list)
    t_pick_start, t_place_end = -1, -1
    
    dist = 0.1
    for step in range(T):
        q_pose = robot_traj[step]
        robot.compute_forward_kinematics(q_pose)
        wrist_pose = robot.get_link_pose(palm_id)
        obj_wrist_pose = np.linalg.inv(obj_T[step]) @ wrist_pose
        obj_pos = obj_wrist_pose[:2, 3]

        d = np.linalg.norm(obj_pos[:2])
        if d < dist and t_pick_start == -1:
            t_pick_start = step
        if step > t_place_start and d > dist and t_place_end == -1:
            t_place_end = step
    
    pick_6D = obj_T[0].copy()
    place_6D = obj_T[-1].copy()
    
    pick_6D[:3,:3] = np.eye(3)
    place_6D[:3,:3] = np.eye(3)
    
    pick = np.zeros((t_pick_end-t_pick_start+1, 4, 4))
    place = np.zeros((t_place_end-t_place_start+1, 4, 4))  
    
    pick_hand_action = np.zeros((t_pick_end-t_pick_start+1, 16))
    place_hand_action = np.zeros((t_place_end-t_place_start+1, 16))
     
    for step in range(t_pick_start, t_pick_end+1):
        q_pose = robot_traj[step]
        robot.compute_forward_kinematics(q_pose)
        
        wrist_pose = robot.get_link_pose(palm_id)
        obj_wrist_pose = np.linalg.inv(pick_6D) @ wrist_pose
        pick[step-t_pick_start] = obj_wrist_pose
        pick_hand_action[step-t_pick_start] = target_traj[step][6:]
        
    for step in range(t_place_start, t_place_end+1):
        q_pose = robot_traj[step]
        robot.compute_forward_kinematics(q_pose)
        
        wrist_pose = robot.get_link_pose(palm_id)
        obj_wrist_pose = np.linalg.inv(place_6D) @ wrist_pose
        place[step-t_place_start] = obj_wrist_pose
        place_hand_action[step-t_place_start] = target_traj[step][6:]
        
    return pick, place, pick_hand_action, place_hand_action

demo_path = "data_Icra/teleoperation/bottle"
demo_path_list = os.listdir(demo_path)
demo_path_list.sort()

for demo_name in demo_path_list:
    os.makedirs(f"lookup/bottle/{demo_name}", exist_ok=True)
    pick, place, pick_hand, place_hand = load_demo(demo_path, demo_name)
    np.save(f"lookup/bottle/{demo_name}/pick.npy", pick)
    np.save(f"lookup/bottle/{demo_name}/place.npy", place)
    np.save(f"lookup/bottle/{demo_name}/pick_hand.npy", pick_hand)
    np.save(f"lookup/bottle/{demo_name}/place_hand.npy", place_hand)