import numpy as np

mimic_joint = {
    "right_ring_2_joint":{
        "joint":"right_ring_1_joint",
        "ratio":1.0
    },
    "right_little_2_joint":{
        "joint":"right_little_1_joint",
        "ratio":1.0
    },
    "right_middle_2_joint":{
        "joint":"right_middle_1_joint",
        "ratio":1.0
    },
    "right_index_2_joint":{
        "joint":"right_index_1_joint",
        "ratio":1.0
    },
    "right_thumb_4_joint":{
        "joint":"right_thumb_2_joint",
        "ratio":1.2
    },
    "right_thumb_3_joint":{
        "joint":"right_thumb_2_joint",
        "ratio":1.0
    }
}

limit = {
    "right_little_1_joint":1.6,
    "right_ring_1_joint":1.6,
    "right_middle_1_joint":1.6,
    "right_index_1_joint":1.6,
    "right_thumb_2_joint":0.55,
    "right_thumb_1_joint":1.15
}

def parse_inspire(inspire_traj, joint_order = ['right_thumb_1_joint', 'right_thumb_2_joint', 'right_index_1_joint', 'right_middle_1_joint', 'right_ring_1_joint', 'right_little_1_joint']):
    # 0 thumb 5 pinky for human action
    T = inspire_traj.shape[0]
    ret = np.zeros((T, len(joint_order)))

    orig_joint_order = ['right_little_1_joint', 'right_ring_1_joint', 'right_middle_1_joint', 'right_index_1_joint', 'right_thumb_2_joint', 'right_thumb_1_joint']
    # Joint order
    # 'right_index_1_joint', 'right_index_2_joint', 'right_little_1_joint', 'right_little_2_joint', 'right_middle_1_joint', 'right_middle_2_joint', 'right_ring_1_joint', 'right_ring_2_joint', 'right_thumb_1_joint', 'right_thumb_2_joint', 'right_thumb_3_joint', 'right_thumb_4_joint'

    for i, joint_name in enumerate(orig_joint_order):
        index = joint_order.index(joint_name)
        ret[:, index] = limit[joint_name] * (1 - inspire_traj[:, i] / 1000)
    
    return ret

def parse_inspire_mimic(inspire_traj, joint_order = ['right_index_1_joint', 'right_index_2_joint', 'right_little_1_joint', 'right_little_2_joint', 'right_middle_1_joint', 'right_middle_2_joint', 'right_ring_1_joint', 'right_ring_2_joint', 'right_thumb_1_joint', 'right_thumb_2_joint', 'right_thumb_3_joint', 'right_thumb_4_joint']):
    # 0 pinky 5 thumb for robot action
    T = inspire_traj.shape[0]
    ret = np.zeros((T, len(joint_order)))

    orig_joint_order = ['right_little_1_joint', 'right_ring_1_joint', 'right_middle_1_joint', 'right_index_1_joint', 'right_thumb_2_joint', 'right_thumb_1_joint']
    # Joint order
    # 'right_index_1_joint', 'right_index_2_joint', 'right_little_1_joint', 'right_little_2_joint', 'right_middle_1_joint', 'right_middle_2_joint', 'right_ring_1_joint', 'right_ring_2_joint', 'right_thumb_1_joint', 'right_thumb_2_joint', 'right_thumb_3_joint', 'right_thumb_4_joint'

    for i, joint_name in enumerate(orig_joint_order):
        index = joint_order.index(joint_name)
        ret[:, index] = limit[joint_name] * (1 - inspire_traj[:, i] / 1000)
    
    for joint_name, mimic in mimic_joint.items():
        index = joint_order.index(joint_name)
        mj = mimic["joint"]
        ratio = mimic["ratio"]
        
        mimic_joint_id = joint_order.index(mj)
        ret[:, index] = ret[:, mimic_joint_id] * ratio
    return ret