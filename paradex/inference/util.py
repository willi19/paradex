import chime
import time
import numpy as np
from scipy.spatial.transform import Rotation as R, Slerp

chime.theme('pokemon')

def home_robot(arm, pose):
    arm.home_robot(pose.copy())  
    
    while not arm.is_ready():
        time.sleep(0.1)

    chime.info()
    
def get_linear_path(start_6D, end_6D, start_hand=None, end_hand=None, length=100):
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