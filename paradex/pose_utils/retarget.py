from paradex.pose_utils.dex_retargeting.dex_retargeting.constants import (
    RobotName, HandType, RetargetingType, get_default_config_path
)
from paradex.pose_utils.dex_retargeting.dex_retargeting.retargeting_config import RetargetingConfig

import numpy as np

def build_allegro_retargeter_right():
    cfg_path = get_default_config_path(
        RobotName.allegro, RetargetingType.position, HandType.right
    )
    return RetargetingConfig.load_from_file(cfg_path).build()

def position_retarget(hand_keypoint_dict):
    retargeter = build_allegro_retargeter_right()
    indices = retargeter.optimizer.target_link_human_indices
    num_frames = len(hand_keypoint_dict.keys())
    
    for i in range(5):
        retargeter.retarget(hand_keypoint_dict[0][indices, :])
    
    q_dict = {}
    for frame in range(0, num_frames):
        q = retargeter.retarget(hand_keypoint_dict[frame][indices, :])
        q_dict[frame] = q
        
    return q_dict

def qpose_dict_to_traj(q_pose_dict):
    sorted_keys = sorted(q_pose_dict.keys())
    q_list = []
    for key in sorted_keys:
        q_list.append(q_pose_dict[key])
    q_array = np.array(q_list)
    
    return q_array[:,:6], q_array[:,6:]