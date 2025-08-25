import os
from paradex.utils.file_io import rsc_path, shared_dir
import numpy as np
from paradex.robot import RobotWrapper
import open3d as o3d
import argparse
from paradex.inference.metric import get_pickplace_timing, compute_mesh_to_ground_distance
from paradex.geometry.coordinate import DEVICE2WRIST

hand_name = "allegro"

def load_demo(demo_path, obj_mesh):
    last_link_pose = np.load(os.path.join(demo_path, "xarm", "action.npy"))
    hand_qpos = np.load(os.path.join(demo_path, hand_name, "action.npy"))
    obj_T = np.load(os.path.join(demo_path, "obj_T.npy"))
    C2R = np.load(os.path.join(demo_path, "C2R.npy"))
    
    for i in range(obj_T.shape[0]):
        obj_T[i] = np.linalg.inv(C2R) @ obj_T[i]
        
    T = min(obj_T.shape[0], hand_qpos.shape[0])

    height_list = []
    for step in range(T):
        h = compute_mesh_to_ground_distance(obj_T[step], obj_mesh)    
        height_list.append(h)
    
    t_pick_end, t_place_start = get_pickplace_timing(height_list)
    t_pick_start, t_place_end = -1, -1
    
    dist = 0.27
    for step in range(T):
        if np.linalg.norm(obj_T[step]) < 0.1:
            continue
        
        obj_wrist_pose = np.linalg.inv(obj_T[step]) @ last_link_pose[step] # @ np.linalg.inv(DEVICE2WRIST['xarm']) @ DEVICE2WRIST['inspire']
        obj_pos = obj_wrist_pose[:2, 3]

        d = np.linalg.norm(obj_pos[:2])
        print(d, step, obj_wrist_pose[:3,3], height_list[step], obj_T[step][:3,3])
        if d < dist and t_pick_start == -1:
            t_pick_start = step
        if step > t_place_start and d > dist and t_place_end == -1:
            t_place_end = step
    
    # print(T, t_pick_start, t_pick_end, t_place_start, t_place_end, demo_path)

    pick_6D = obj_T[0].copy()
    place_6D = obj_T[t_place_end].copy()
    
    print(t_pick_start, t_pick_end)
    print(t_place_start, t_place_end)
    
    pick_6D[:3,:3] = np.eye(3)
    place_6D[:3,:3] = np.eye(3)
    # print(place_6D, pick_6D, obj_T.shape)
    pick = np.zeros((t_pick_end-t_pick_start+1, 4, 4))
    place = np.zeros((t_place_end-t_place_start+1, 4, 4))  
    
    pick_hand_action = hand_qpos[t_pick_start:t_pick_end+1]
    place_hand_action = hand_qpos[t_place_start:t_place_end+1]
     
    for step in range(t_pick_start, t_pick_end+1):
        pick[step-t_pick_start] = np.linalg.inv(pick_6D) @ last_link_pose[step]
        
    for step in range(t_place_start, t_place_end+1):
        place[step-t_place_start] = np.linalg.inv(place_6D) @ last_link_pose[step]
        
    return pick, place, pick_hand_action, place_hand_action

parser = argparse.ArgumentParser()
parser.add_argument('--obj_name', nargs="+", type=str, default=None)
parser.add_argument('--grasp_type', nargs="+", type=str, default=None)
args = parser.parse_args()

process_list = []

if args.obj_name == None:
    name_list = os.listdir(os.path.join(shared_dir, 'capture_', "lookup"))
    name_list.sort()

else:
    name_list = args.obj_name
    
for name in name_list:
    grasp_list = os.listdir(os.path.join(shared_dir, "capture_", "lookup", name))
    if args.grasp_type is not None:
        for grasp_name in args.grasp_type:
            if grasp_name in grasp_list:
                process_list.append((name, grasp_name))
    if args.grasp_type is None:
        for grasp_name in grasp_list:
            process_list.append((name, grasp_name))

for name, grasp_type in process_list:
        root_dir = os.path.join(shared_dir, "capture_", "lookup", name, grasp_type)
        index_list = os.listdir(root_dir)
        index_list.sort(key=lambda x:int(x))
        
        obj_mesh = o3d.io.read_triangle_mesh(os.path.join(rsc_path, "object", name, f"{name}.obj"))
        for index in index_list:                
            demo_path = os.path.join(root_dir, index)
            
            pick, place, pick_hand, place_hand = load_demo(demo_path, obj_mesh)
            os.makedirs(f"data/lookup/{name}/{grasp_type}/{index}",exist_ok=True)
            np.save(f"data/lookup/{name}/{grasp_type}/{index}/pick.npy", pick)
            np.save(f"data/lookup/{name}/{grasp_type}/{index}/place.npy", place)
            np.save(f"data/lookup/{name}/{grasp_type}/{index}/pick_hand.npy", pick_hand)
            np.save(f"data/lookup/{name}/{grasp_type}/{index}/place_hand.npy", place_hand)