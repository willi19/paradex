import numpy as np
from scipy.spatial.transform import Rotation as R
from scipy.spatial.transform import Rotation as R, Slerp
import os
import random

from paradex.utils.file_io import shared_dir

lookup_table_path = os.path.join(shared_dir, "capture", "lookup")

def get_pringles_index(hand, pick6D, place6D):
    # pick_lay = False
    # if pick6D[2, 2] < 0.7:# or place6D[2, 2] < 0.7:
    #     pick_lay = True
    
    # place_lay = False
    # if place6D[2, 2] < 0.7:# or place6D[2, 2] < 0.7:
    #     place_lay = True
    
    # table_path = os.path.join(lookup_table_path, "pringles")
    # index_list = os.listdir(table_path)
    
    # candidate = []
    # for index in index_list:
    #     pick_T = np.load(os.path.join(table_path, index, "pick_obj_T.npy"))[0]
    #     place_T = np.load(os.path.join(table_path, index, "place_obj_T.npy"))[-1]
        
    #     table_pick_lay = (pick_T[2, 2]< 0.7)
    #     table_place_lay = (place_T[2, 2] < 0.7)
        
    #     score = 1
    #     if pick_lay != table_pick_lay or place_lay != table_place_lay:
    #         score = 0
    #     candidate.append((index, score))
    
    # indices, weights = zip(*candidate)
    # selected_index = random.choices(indices, weights=weights, k=1)[0]
    # return selected_index
    return "1"

def get_traj(obj, hand, start6D, pick6D, place6D):
    if obj == "pringles":
        index = get_pringles_index(hand, pick6D, place6D)
        index_path = os.path.join(lookup_table_path, obj, index)
        
        pick_traj = np.load(f"{index_path}/pick.npy")
        place_traj = np.load(f"{index_path}/place.npy")
        
        pick_hand_traj = np.load(f"{index_path}/pick_hand.npy")
        place_hand_traj = np.load(f"{index_path}/place_hand.npy")
        start_hand = np.zeros((pick_hand_traj.shape[1]))
        
        pick_traj = pick6D @ pick_traj
        place_traj = place6D @ place_traj 
        
        
        approach_traj, approach_hand_traj = get_linear_path(start6D, pick_traj[0], start_hand, pick_hand_traj[0],100)
        return_traj, return_hand_traj = get_linear_path(place_traj[-1], start6D, pick_hand_traj[-1], start_hand, 100)
        move_traj, move_hand = get_linear_path(pick_traj[-1], place_traj[0], pick_hand_traj[-1], place_hand_traj[0], 100)
        
        
        traj = np.concatenate([approach_traj, pick_traj, move_traj, place_traj, return_traj])
        hand_traj = np.concatenate([approach_hand_traj, pick_hand_traj, move_hand, place_hand_traj, return_hand_traj])
        
    else:
        raise NotImplementedError
    
    return index, traj, hand_traj

def get_linear_path(start_6D, end_6D, start_hand, end_hand, length):
    move_traj = np.zeros((length, 4, 4))
    move_hand = np.zeros((length, start_hand.shape[0]))
    
    start_pos = start_6D[:3,3]
    end_pos = end_6D[:3,3]
    
    start_rot = R.from_matrix(start_6D[:3, :3])
    end_rot = R.from_matrix(end_6D[:3, :3])
    key_times = [0, 1]
    key_rots = R.concatenate([start_rot, end_rot])  # Rotation 배열 생성

    slerp = Slerp(key_times, key_rots)
    
    for i in range(length):
        alpha = (i + 1) / length
        
        pos = (1 - alpha) * start_pos + alpha * end_pos
        rot = slerp(alpha).as_matrix()
        
        T = np.eye(4)
        T[:3, :3] = rot
        T[:3, 3] = pos
        
        move_traj[i] = T
        move_hand[i] = (start_hand * (1-alpha) + end_hand * alpha)
    
    return move_traj, move_hand