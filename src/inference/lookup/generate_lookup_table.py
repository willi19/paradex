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
    
    split_t = -1
    max_h = -1
    
    for step in range(T):
        if np.linalg.norm(obj_T[step]) < 0.1:
            continue
        place_6D = obj_T[step].copy()
        if obj_T[step, 2, 3] > max_h:
            max_h = obj_T[step, 2, 3]
            split_t = step
            
    pick_6D = obj_T[0].copy()
    if "lay" in demo_path:
        z = np.array([pick_6D[0,2], pick_6D[1, 2], 0])
        z /= np.linalg.norm(z)
                    
        pick_6D[:3, 2] = z
        pick_6D[:3, 0] = np.array([0,0,1])
        pick_6D[:3, 1] = np.array([z[1],-z[0],0])
    
    else:
        pick_6D[:3, :3] = np.eye(3)
        
    place_6D[:3,:3] = np.eye(3)
            
    pick_hand_action = hand_qpos[:split_t]
    place_hand_action = hand_qpos[split_t:]
    pick = last_link_pose[:split_t]
    place = last_link_pose[split_t:]
    
    for i in range(len(pick_hand_action)):
        pick[i] = np.linalg.inv(pick_6D) @ pick[i]
        
    for i in range(len(place_hand_action)):
        place[i] = np.linalg.inv(place_6D) @ place[i]
        
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