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

def inspire_action_to_qpos(action: np.ndarray) -> np.ndarray:
    # Map Inspire raw action (little, ring, middle, index, thumb_2, thumb_1) to joint angles.
    limits = {
        "pinky_proximal_joint": 1.47,
        "ring_proximal_joint": 1.47,
        "middle_proximal_joint": 1.47,
        "index_proximal_joint": 1.47,
        "thumb_proximal_pitch_joint": 0.60, # 0.55
        "thumb_proximal_yaw_joint": 1.40, # 1.15
    }
    
    offsets = {
        "pinky_proximal_joint": 0.0,
        "ring_proximal_joint": 0.0,
        "middle_proximal_joint": 0.0,
        "index_proximal_joint": 0.0,
        "thumb_proximal_pitch_joint": 0.0,
        "thumb_proximal_yaw_joint": 0.0,
    }
    qpos = np.zeros_like(action, dtype=float)
    qpos[:, 0] = limits["thumb_proximal_yaw_joint"] * (1.0 - action[:, 5] / 1000.0)
    qpos[:, 1] = limits["thumb_proximal_pitch_joint"] * (1.0 - action[:, 4] / 1000.0)
    # qpos[:, 2] = limits["index_proximal_joint"] * (1.0 - action[:, 3] / 1000.0)
    # qpos[:, 3] = limits["middle_proximal_joint"] * (1.0 - action[:, 2] / 1000.0)
    # qpos[:, 4] = limits["ring_proximal_joint"] * (1.0 - action[:, 1] / 1000.0)
    # qpos[:, 5] = limits["pinky_proximal_joint"] * (1.0 - action[:, 0] / 1000.0)
    
    qpos[:, 2] = (-4e-8 * action[:, 3]**3 + 3e-5 * action[:, 3]**2 - 0.0704 * action[:, 3] + 83.572) * 3.14159 / 180.0
    qpos[:, 3] = (-4e-8 * action[:, 2]**3 + 3e-5 * action[:, 2]**2 - 0.0704 * action[:, 2] + 83.572) * 3.14159 / 180.0
    qpos[:, 4] = (-4e-8 * action[:, 1]**3 + 3e-5 * action[:, 1]**2 - 0.0704 * action[:, 1] + 83.572) * 3.14159 / 180.0
    
    qpos[:, 5] = (-4e-8 * action[:, 0]**3 + 3e-5 * action[:, 0]**2 - 0.0704 * action[:, 0] + 83.572) * 3.14159 / 180.0
    return qpos



def inspire_action_to_qpos_dof12(action: np.ndarray) -> np.ndarray:
    # Map Inspire raw action (little, ring, middle, index, thumb_2, thumb_1) to joint angles.
    limits = {
        "pinky_proximal_joint": 1.47,
        "ring_proximal_joint": 1.47,
        "middle_proximal_joint": 1.47,
        "index_proximal_joint": 1.47,
        "thumb_proximal_pitch_joint": 0.60, # 0.55
        "thumb_proximal_yaw_joint": 1.40, # 1.15
        "index_2_joint": 1.57,
        "middle_2_joint": 1.57,
        "ring_2_joint": 1.57,
        "pinky_2_joint": 1.57
    }
    
    offsets = {
        "pinky_proximal_joint": 0.0,
        "ring_proximal_joint": 0.0,
        "middle_proximal_joint": 0.0,
        "index_proximal_joint": 0.0,
        "thumb_proximal_pitch_joint": 0.0,
        "thumb_proximal_yaw_joint": 0.0,
    }
    qpos = np.zeros((action.shape[0], 12), dtype=float)
    print(action.shape, qpos.shape)
    # thumb_1_joint
    mask = action[:, 5] <= 100
    
    qpos[:, 0] = (7e-9 * action[:, 5]**3 - 1e-5 * action[:, 5]**2 - 0.073 * action[:, 5] + 75.866) * 3.14159 / 180.0
    qpos[mask] = 68.5 * 3.14159 / 180.0

    # thumb_2_joint
    qpos[:, 1] = (2e-8 * action[:, 4]**3 - 5e-5 * action[:, 4]**2 - 0.005 * action[:, 4] + 31.407) * 3.14159 / 180.0
    # index_1_joint
    qpos[:, 4] = (-4e-8 * action[:, 3]**3 + 3e-5 * action[:, 3]**2 - 0.0704 * action[:, 3] + 83.572) * 3.14159 / 180.0
    # middle_1_joint
    qpos[:, 6] = (-4e-8 * action[:, 2]**3 + 3e-5 * action[:, 2]**2 - 0.0704 * action[:, 2] + 83.572) * 3.14159 / 180.0
    # ring_1_joint
    qpos[:, 8] = (-4e-8 * action[:, 1]**3 + 3e-5 * action[:, 1]**2 - 0.0704 * action[:, 1] + 83.572) * 3.14159 / 180.0
    # little_1_joint
    qpos[:, 10] = (-4e-8 * action[:, 0]**3 + 3e-5 * action[:, 0]**2 - 0.0704 * action[:, 0] + 83.572) * 3.14159 / 180.0
    

    # thumb_3_joint
    qpos[:, 2] = (3e-11 * action[:, 4]**4  - 4e-8 * action[:, 4]**3 + 9e-6 * action[:, 4]**2 - 0.025 * action[:, 4] + 28.197) * 3.14159 / 180.0
    # thumb_4_joint
    qpos[:, 3] = (8e-9 * action[:, 4]**3 - 5e-6 * action[:, 4]**2 - 0.0267 * action[:, 4] + 24.189) * 3.14159 / 180.0
    # index_2_joint
    qpos[:, 5] = limits["index_2_joint"] * (1.0 - action[:, 3] / 1000.0)
    # middle_2_joint
    qpos[:, 7] = limits["middle_2_joint"] * (1.0 - action[:, 2] / 1000.0)
    # ring_2_joint
    qpos[:, 9] = limits["ring_2_joint"] * (1.0 - action[:, 1] / 1000.0)
    # little_2_joint
    qpos[:, 11] = limits["pinky_2_joint"] * (1.0 - action[:, 0] / 1000.0)


    return qpos