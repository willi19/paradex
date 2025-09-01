import os
import numpy as np
import open3d as o3d
import argparse

from paradex.utils.file_io import rsc_path, shared_dir, load_c2r

def normalize(obj_6D):
    ret = obj_6D.copy()
    
    if obj_6D[2, 2] < 0.7:
        z = np.array([obj_6D[0,2], obj_6D[1, 2], 0])
        if z[0] == -1:
            z *= -1
        z /= np.linalg.norm(z)
        ret[:3, 2] = z
        ret[:3, 0] = np.array([0,0,1])
        ret[:3, 1] = np.array([z[1],-z[0],0])
    else:
        ret[:3, :3] = np.eye(3)
        
    return ret

def load_demo(demo_path):
    last_link_pose = np.load(os.path.join(demo_path, "arm", "action.npy"))
    hand_qpos = np.load(os.path.join(demo_path, "hand", "action.npy"))
    
    obj_T_dict = np.load(os.path.join(demo_path, "obj_T.npy"), allow_pickle=True).item()
    obj_T = list(obj_T_dict.values())[0]
    import pdb; pdb.set_trace()
    T = min(obj_T.shape[0], hand_qpos.shape[0])
    split_t = -1
    max_h = -1
    pick_6D = np.zeros((4,4))
    for step in range(T):
        if np.linalg.norm(obj_T[step]) < 0.1:
            continue
        if np.linalg.norm(pick_6D) < 0.1:
            pick_6D = obj_T[step].copy()
        place_6D_orig = obj_T[step].copy()
        if obj_T[step, 2, 3] > max_h:
            max_h = obj_T[step, 2, 3]
            split_t = step
    # pick_6D = normalize(obj_T[0].copy())
    place_6D = normalize(place_6D_orig.copy())
        
    if np.linalg.norm(obj_T[0]) < 0.1:
        print(demo_path)
        
    pick_6D_diff = np.linalg.inv(obj_T[0]) @ pick_6D
    place_6D_diff = np.linalg.inv(place_6D_orig) @ place_6D
    
    place_T = min(last_link_pose.shape[0], obj_T.shape[0])
    
    pick_hand_action = hand_qpos[:split_t]
    place_hand_action = hand_qpos[split_t:place_T]
    
    pick = last_link_pose[:split_t]
    place = last_link_pose[split_t:place_T]
    
    pick_objT = obj_T[:split_t]
    place_objT = obj_T[split_t:place_T]
    
    for i in range(len(pick_hand_action)):
        pick[i] = np.linalg.inv(pick_6D) @ pick[i]
        if np.linalg.norm(pick_objT[i]) < 0.1:
            pick_objT[i] = pick_objT[i-1].copy()
            
        pick_objT[i] = pick_objT[i] @ pick_6D_diff
    
    for i in range(len(place_hand_action)):
        place[i] = np.linalg.inv(place_6D) @ place[i]
        if np.linalg.norm(place_objT[i]) < 0.1:
            place_objT[i] = place_objT[i-1].copy()
        place_objT[i] = place_objT[i] @ place_6D_diff
    
    return pick, place, pick_hand_action, place_hand_action, pick_objT, place_objT

def plot_table():
    pass

parser = argparse.ArgumentParser()
parser.add_argument('--obj', nargs="+", type=str, default=None)
args = parser.parse_args()

process_list = []

if args.obj == None:
    name_list = os.listdir(os.path.join(shared_dir, 'capture', "lookup"))
    name_list.sort()

else:
    name_list = args.obj
    
for name in name_list:
    vis_list = []
    index_list = os.listdir(os.path.join(shared_dir, 'capture', "lookup", name))
    for index in index_list:
        demo_path = os.path.join(shared_dir, 'capture', "lookup", name, index)
        if os.path.exists(os.path.join(demo_path, "pick.npy")) and os.path.exists(os.path.join(demo_path, "place.npy")):
            continue
    
        pick, place, pick_hand, place_hand, pick_obj_T, place_obj_T = load_demo(demo_path)
        
        np.save(f"{demo_path}/pick.npy", pick)
        np.save(f"{demo_path}/place.npy", place)
        np.save(f"{demo_path}/pick_hand.npy", pick_hand)
        np.save(f"{demo_path}/place_hand.npy", place_hand)
        np.save(f"{demo_path}/pick_obj_T.npy", pick_obj_T)
        np.save(f"{demo_path}/place_obj_T.npy", place_obj_T)