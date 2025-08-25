import os
import numpy as np
import open3d as o3d
import argparse

from paradex.utils.file_io import shared_dir

def load_demo(demo_path):
    last_link_pose = np.load(os.path.join(demo_path, "xarm", "action.npy"))
    hand_qpos = np.load(os.path.join(demo_path, "hand", "action.npy"))
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
    if pick_6D[2, 2] < 0.7:
        z = np.array([pick_6D[0,2], pick_6D[1, 2], 0])
        if z[0] == -1:
            z *= -1
        z /= np.linalg.norm(z)
                    
        pick_6D[:3, 2] = z
        pick_6D[:3, 0] = np.array([0,0,1])
        pick_6D[:3, 1] = np.array([z[1],-z[0],0])
    
    else:
        pick_6D[:3, :3] = np.eye(3)

    pick_6D_diff = np.linalg.inv(obj_T[0]) @ pick_6D
    place_6D_diff = np.eye(4)
    place_6D_diff[:3,:3] = np.linalg.inv(place_6D[:3,:3])

    print(pick_6D_diff, place_6D_diff)
    place_6D[:3,:3] = np.eye(3)
    
    place_T = min(last_link_pose.shape[0], obj_T.shape[0])
    pick_hand_action = hand_qpos[:split_t]
    place_hand_action = hand_qpos[split_t:place_T]
    pick = last_link_pose[:split_t]
    place = last_link_pose[split_t:place_T]

    pick_objT = obj_T[:split_t]
    place_objT = obj_T[split_t:place_T]
    
    for i in range(len(pick_hand_action)):
        pick[i] = np.linalg.inv(pick_6D) @ pick[i]
        pick_objT[i] = pick_objT[i] @ pick_6D_diff
        
    for i in range(len(place_hand_action)):
        place[i] = np.linalg.inv(place_6D) @ place[i]
        place_objT[i] = place_objT[i] @ place_6D_diff
        
    return pick, place, pick_hand_action, place_hand_action, pick_objT, place_objT

parser = argparse.ArgumentParser()
parser.add_argument('--obj_name', nargs="+", type=str, default=None)
parser.add_argument('--hand_type', nargs="+", type=str, default=None)
args = parser.parse_args()

process_list = []

if args.obj_name == None:
    name_list = os.listdir(os.path.join(shared_dir, 'capture', "lookup"))
    name_list.sort()

else:
    name_list = args.obj_name
    
for name in name_list:
    hand_list = os.listdir(os.path.join(shared_dir, "capture", "lookup", name))
    if args.hand_type is not None:
        for hand_name in args.hand_type:
            if hand_name in hand_list:
                process_list.append((name, hand_name))
    if args.hand_type is None:
        for hand_name in hand_list:
            process_list.append((name, hand_name))

for name, hand_type in process_list:
    root_dir = os.path.join(shared_dir, "capture", "lookup", name, hand_name)
    index_list = os.listdir(root_dir)
    index_list.sort(key=lambda x:int(x))
    
    for index in index_list:                
        demo_path = os.path.join(root_dir, index)
        
        pick, place, pick_hand, place_hand, pick_obj_T, place_obj_T = load_demo(demo_path)
        
        np.save(f"{shared_dir}/capture/lookup/{name}/{hand_type}/{index}/pick.npy", pick)
        np.save(f"{shared_dir}/capture/lookup/{name}/{hand_type}/{index}/place.npy", place)
        np.save(f"{shared_dir}/capture/lookup/{name}/{hand_type}/{index}/pick_hand.npy", pick_hand)
        np.save(f"{shared_dir}/capture/lookup/{name}/{hand_type}/{index}/place_hand.npy", place_hand)
        np.save(f"{shared_dir}/capture/lookup/{name}/{hand_type}/{index}/pick_obj_T.npy", pick_obj_T)
        np.save(f"{shared_dir}/capture/lookup/{name}/{hand_type}/{index}/place_obj_T.npy", place_obj_T)