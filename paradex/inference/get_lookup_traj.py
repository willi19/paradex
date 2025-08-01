import numpy as np
from scipy.spatial.transform import Rotation as R
from scipy.spatial.transform import Rotation as R, Slerp

def get_traj(pick_traj, pick_6D, place_traj, place_6D, pick_hand, place_hand):
    lift_T = 100
    lower_T = 100
    move_T = 150
    pick_T = pick_traj.shape[0]
    place_T = place_traj.shape[0]
    total_T = pick_T + lift_T + move_T + lower_T + place_T
    traj = np.zeros((total_T, 4, 4))
    
    hand_dof = pick_hand.shape[-1]
    hand_traj = np.zeros((total_T, hand_dof))
    
    for i in range(pick_T):
        traj[i] = pick_6D @ pick_traj[i]
        hand_traj[i] = pick_hand[i]
        
    for i in range(lift_T):
        traj[i+pick_T] = traj[pick_T-1].copy()
        traj[i+pick_T][2,3] += 0.001 * (i+1)
        hand_traj[i+pick_T] = pick_hand[-1]
    
    for i in range(place_T):
        traj[i+total_T-place_T] = place_6D @ place_traj[i]
        hand_traj[i+total_T-place_T] = place_hand[i]
        
    for i in range(lower_T):
        traj[total_T-place_T-1-i] = traj[total_T-place_T].copy()
        traj[total_T-place_T-1-i][2,3] += 0.001 * (i+1)    
        hand_traj[total_T-place_T-1-i] = place_hand[0]
        
    
    start_pose = traj[pick_T + lift_T - 1]
    end_pose = traj[total_T - place_T - lower_T]

    start_pos = start_pose[:3, 3]
    end_pos = end_pose[:3, 3]
    
    start_rot = R.from_matrix(start_pose[:3, :3])
    end_rot = R.from_matrix(end_pose[:3, :3])
    key_times = [0, 1]
    key_rots = R.concatenate([start_rot, end_rot])  # Rotation 배열 생성

    slerp = Slerp(key_times, key_rots)

    for i in range(move_T):
        alpha = (i + 1) / move_T
        pos = (1 - alpha) * start_pos + alpha * end_pos
        rot = slerp(alpha).as_matrix()
        
        T = np.eye(4)
        T[:3, :3] = rot
        T[:3, 3] = pos
        traj[pick_T + lift_T + i] = T
        hand_traj[pick_T + lift_T + i] = (pick_hand[-1] * (1-alpha) + place_hand[0] * alpha)
    
    return traj, hand_traj