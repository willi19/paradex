import os
from paradex.utils.file_io import rsc_path, shared_dir
import numpy as np
from paradex.robot import RobotWrapper
import pickle

def load_demo(demo_path, demo_name):
    arm_qpos = np.load(os.path.join(demo_path, str(demo_name), "arm", "position.npy"))
    arm_time = np.load(os.path.join(demo_path, str(demo_name), "arm", "time.npy"))
    
    hand_qpos = np.load(os.path.join(demo_path, str(demo_name), "inspire", "position.npy"))
    hand_time = np.load(os.path.join(demo_path, str(demo_name), "inspire", "time.npy"))
    
    urdf_file = os.path.join(rsc_path, "robot", "xarm.urdf")
    robot = RobotWrapper(urdf_file)

    link6_id = robot.get_link_index("link6")

    pick_T = np.load(os.path.join(demo_path, str(demo_name), "start", "obj_6D.npy"))
    pick_T[:3,:3] = np.eye(3)
    
    place_T = np.load(os.path.join(demo_path, str(demo_name), "end", "obj_6D.npy"))
    place_T[:3,:3] = np.eye(3)
    
    t_pick_start = 580
    t_pick_end = 750
    
    t_place_start = 920
    t_place_end = 1260
    
    pick = arm_qpos[t_pick_start:t_pick_end+1]
    place = arm_qpos[t_place_start:t_place_end+1]
    
    t_pick_start_time = arm_time[t_pick_start]
    t_pick_end_time = arm_time[t_pick_end]
    
    t_place_start_time = arm_time[t_place_start]
    t_place_end_time = arm_time[t_place_end]
    
    pick_hand_action = []
    place_hand_action = []
    
    for step in range(hand_qpos.shape[0]):
        hand_t = hand_time[step]
        if t_pick_start_time <= hand_t and hand_t <= t_pick_end_time:
            pick_hand_action.append(hand_qpos[step])    
        
        if t_place_start_time <= hand_t and hand_t <= t_place_end_time:
            place_hand_action.append(hand_qpos[step])
    
    pick_hand_action = np.array(pick_hand_action)
    place_hand_action = np.array(place_hand_action)

    final_pick_range = 150
    final_place_range = 150
    
    pick_indices = np.sort(np.random.choice(len(pick_hand_action), final_pick_range, replace=False))
    place_indices = np.sort(np.random.choice(len(place_hand_action), final_place_range, replace=False))

    pick_hand_action = pick_hand_action[pick_indices]
    place_hand_action = place_hand_action[place_indices]

    pick_indices = np.sort(np.random.choice(len(pick), final_pick_range, replace=False))
    place_indices = np.sort(np.random.choice(len(place), final_place_range, replace=False))

    pick_qpos = pick[pick_indices]
    place_qpos = place[place_indices]
    
    pick = []
    place = []
    
    for i in range(final_pick_range):
        q_pose = pick_qpos[i]
        robot.compute_forward_kinematics(q_pose)
        wrist_pose = robot.get_link_pose(link6_id)
        
        obj_wrist_pose = np.linalg.inv(pick_T) @ wrist_pose
        pick.append(obj_wrist_pose.copy())
    
    pick = np.array(pick)
    
    for i in range(final_place_range):
        q_pose = place_qpos[i]
        robot.compute_forward_kinematics(q_pose)
        wrist_pose = robot.get_link_pose(link6_id)
        
        obj_wrist_pose = np.linalg.inv(place_T) @ wrist_pose
        place.append(obj_wrist_pose.copy())
    
    place = np.array(place)
    # t_pick_end, t_place_start = 
    # t_pick_start, t_place_end = -1, -1
    
    # dist = 0.1
    # for step in range(T):
    #     q_pose = robot_traj[step]
    #     robot.compute_forward_kinematics(q_pose)
    #     wrist_pose = robot.get_link_pose(palm_id)
    #     obj_wrist_pose = np.linalg.inv(obj_T[step]) @ wrist_pose
    #     obj_pos = obj_wrist_pose[:2, 3]

    #     d = np.linalg.norm(obj_pos[:2])
    #     if d < dist and t_pick_start == -1:
    #         t_pick_start = step
    #     if step > t_place_start and d > dist and t_place_end == -1:
    #         t_place_end = step
    
    # pick_6D = obj_T[0].copy()
    # place_6D = obj_T[-1].copy()
    
    # pick_6D[:3,:3] = np.eye(3)
    # place_6D[:3,:3] = np.eye(3)
    
    # pick = np.zeros((t_pick_end-t_pick_start+1, 4, 4))
    # place = np.zeros((t_place_end-t_place_start+1, 4, 4))  
    
    # pick_hand_action = np.zeros((t_pick_end-t_pick_start+1, 16))
    # place_hand_action = np.zeros((t_place_end-t_place_start+1, 16))
     
    # for step in range(t_pick_start, t_pick_end+1):
    #     q_pose = robot_traj[step]
    #     robot.compute_forward_kinematics(q_pose)
        
    #     wrist_pose = robot.get_link_pose(palm_id)
    #     obj_wrist_pose = np.linalg.inv(pick_6D) @ wrist_pose
    #     pick[step-t_pick_start] = obj_wrist_pose
    #     pick_hand_action[step-t_pick_start] = target_traj[step][6:]
        
    # for step in range(t_place_start, t_place_end+1):
    #     q_pose = robot_traj[step]
    #     robot.compute_forward_kinematics(q_pose)
        
    #     wrist_pose = robot.get_link_pose(palm_id)
    #     obj_wrist_pose = np.linalg.inv(place_6D) @ wrist_pose
    #     place[step-t_place_start] = obj_wrist_pose
    #     place_hand_action[step-t_place_start] = target_traj[step][6:]
        
    return pick, place, pick_hand_action, place_hand_action

demo_path = os.path.join(shared_dir, "lookup", "pringles")
demo_path_list = os.listdir(demo_path)
demo_path_list.sort()

for demo_name in demo_path_list:
    pick, place, pick_hand, place_hand = load_demo(demo_path, demo_name)
    os.makedirs(f"lookup/pringles/{demo_name}",exist_ok=True)
    np.save(f"lookup/pringles/{demo_name}/pick.npy", pick)
    np.save(f"lookup/pringles/{demo_name}/place.npy", place)
    np.save(f"lookup/pringles/{demo_name}/pick_hand.npy", pick_hand)
    np.save(f"lookup/pringles/{demo_name}/place_hand.npy", place_hand)