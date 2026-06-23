import numpy as np
from scipy.spatial.transform import Rotation as R

from typing import Dict
import time
import copy
def allegro(hand_pose_frame):
    hand_joint_angle = np.zeros((20,3))
    allegro_angles = np.zeros(16)
    # for finger_id in range(4):
    #     for joint_id in range(4):
    #         if joint_id == 0:
    #             rot_mat = np.linalg.inv(hand_pose_frame[0,:3,:3]) @ hand_pose_frame[finger_id * 4 + joint_id + 1, :3,:3]
    #         else:
    #             rot_mat = np.linalg.inv(hand_pose_frame[hand_index.hand_index_parent[finger_id * 4 + joint_id+1], :3,:3]) @ hand_pose_frame[finger_id * 4 + joint_id + 1, :3,:3]
    #         hand_joint_angle[finger_id * 4 + joint_id + 1] = Rotation.from_matrix(rot_mat).as_euler("zyx")
    
    # zyx euler angle in hand frame = zxy axis angle in robot frame
    
    # Ring
    joint_name_list = ["metacarpal", "proximal", "intermediate","distal"]
    for i, finger_name in enumerate(["index", "middle", "ring"]):
        metacarpal = finger_name + "_metacarpal"
        distal = finger_name + "_distal"
        
        tip_position = (np.linalg.inv(hand_pose_frame["wrist"]) @ hand_pose_frame[distal])[:3, 3]
        finger_base_position = (np.linalg.inv(hand_pose_frame["wrist"]) @ hand_pose_frame[metacarpal])[:3, 3]
        
        tip_position = tip_position - finger_base_position
        tip_direction  = tip_position / np.linalg.norm(tip_position)

        if tip_direction[1] > 0.9:
            allegro_angles[4*i] = 0
        else:
            allegro_angles[4*i] = np.arctan(tip_direction[0] / tip_direction[2]) * (0.9-tip_direction[1])
        
        for j in range(3):
            parent_name = finger_name + "_" + joint_name_list[j]
            joint_name = finger_name + "_" + joint_name_list[j+1]
            rot_mat = np.linalg.inv(hand_pose_frame[parent_name][:3,:3]) @ hand_pose_frame[joint_name][:3,:3]
            v = rot_mat[1, 1] if rot_mat[2, 1] >= 0 else 1
            v = max(-1, min(1, v))
            allegro_angles[4*i+j+1] = np.arccos(v)
        allegro_angles[4*i+1] = (allegro_angles[4*i+1]-0.35) * 1.5


    # Thumb
    thumb_meta = np.dot(hand_pose_frame["wrist"][:3,:3].T, hand_pose_frame["thumb_metacarpal"][:3,:3])
    thumb_meta_angle = R.from_matrix(thumb_meta).as_euler("xyz")
    allegro_angles[12] = thumb_meta_angle[0] 
    allegro_angles[13] = -thumb_meta_angle[2]-1.57

    for i, (parent_name, joint_name) in enumerate([("thumb_metacarpal", "thumb_proximal"),("thumb_proximal", "thumb_distal")]):
        rot_mat = np.linalg.inv(hand_pose_frame[parent_name][:3,:3]) @ hand_pose_frame[joint_name][:3,:3]
        allegro_angles[14+i] = rot_mat[2, 1] * 1.2

    return allegro_angles

def inspire(hand_pose_frame):
    inspire_angles = np.zeros(6)

    for i, finger_name in enumerate(["thumb", "index", "middle", "ring", "pinky"]):
        metacarpal = finger_name + "_metacarpal"
        distal = finger_name + "_distal"
        
        tip_pos = (np.linalg.inv(hand_pose_frame["wrist"]) @ hand_pose_frame[distal])
        
        if finger_name != "thumb":
            angle = np.arctan2(tip_pos[2, 1], tip_pos[1, 1])
            if angle < -np.pi / 2:
                angle = 2 * np.pi + angle
            
            inspire_angles[4-i] = (1-max(0, min(1, angle / np.pi))) * 1000

        else:
            tip_position = tip_pos[:3, 3]
            finger_base_position = (np.linalg.inv(hand_pose_frame["wrist"]) @ hand_pose_frame[metacarpal])[:3, 3]
            tip_direction = tip_position - finger_base_position
            tip_direction  = tip_direction / np.linalg.norm(tip_direction)
            tip_direction[1] *= -1
            tip_direction[2] *= -1
            if tip_direction[0] > 0:
                inspire_angles[5] = 1000 - np.arctan(-tip_direction[2] / abs(tip_direction[0])) / np.pi * 2000
                inspire_angles[4] = np.arccos(-tip_direction[1]) * 2000 - 1000 # no divide by pi for better range
            else:
                inspire_angles[5] = 0
                inspire_angles[4] = np.arcsin(-tip_direction[2]) / np.pi * 2000  * 3.5 - 1000
    # print(inspire_angles)
    return inspire_angles

def inspire_f1_deprecated(hand_pose_frame):
    """
    Same kinematic mapping as inspire(), but scale each DOF to Inspire F1 raw range.
    """
    # inspire_angles = inspire(hand_pose_frame)
    # f1_ranges = np.array([
    #     [900, 1740],
    #     [900, 1740],
    #     [900, 1740],
    #     [900, 1740],
    #     [1100, 1350],
    #     [600, 1800],
    # ], dtype=np.float64)

    # # Clamp to [0, 1000] then scale to per-DOF range
    # inspire_angles = np.clip(inspire_angles, 0.0, 1000.0)
    # mins = f1_ranges[:, 0]
    # maxs = f1_ranges[:, 1]
    # scaled = mins + (inspire_angles / 1000.0) * (maxs - mins)
    # inspire_angles[:4] = [1000.0] * 4

    
    
    
    inspire_angles = np.zeros(6)

    for i, finger_name in enumerate(["thumb", "index", "middle", "ring", "pinky"]):
        metacarpal = finger_name + "_metacarpal"
        distal = finger_name + "_distal"
        
        tip_pos = (np.linalg.inv(hand_pose_frame["wrist"]) @ hand_pose_frame[distal])
        
        if finger_name != "thumb":
            angle = np.arctan2(tip_pos[2, 1], tip_pos[1, 1])
            if angle < -np.pi / 2:
                angle = 2 * np.pi + angle
            
            inspire_angles[4-i] = (1-max(0, min(1, angle / np.pi))) * 500 + 500

        else:
            tip_position = tip_pos[:3, 3]
            finger_base_position = (np.linalg.inv(hand_pose_frame["wrist"]) @ hand_pose_frame[metacarpal])[:3, 3]
            tip_direction = tip_position - finger_base_position
            tip_direction  = tip_direction / np.linalg.norm(tip_direction)
            tip_direction[1] *= -1
            tip_direction[2] *= -1
            
            print(tip_direction)
            if tip_direction[0] > 0:
                inspire_angles[4] = 1000 - np.arctan(-tip_direction[2] / abs(tip_direction[0])) / np.pi * 250
                inspire_angles[5] = -np.arccos(tip_direction[0]) * 800 + 1500 # no divide by pi for better range
            else:
                inspire_angles[4] = 1000 - np.arctan(-tip_direction[2] / abs(tip_direction[0])) / np.pi * 250
                # inspire_angles[4] = 1000 - np.arctan(-tip_direction[2] / abs(tip_direction[0])) / np.pi * 2000
                print("asfd")
                inspire_angles[5] = 300
    print(inspire_angles)

                
        # inspire_angles[:4] = 500.0 + inspire_angles[:4] * 0.5
        
        
    return inspire_angles
        


def inspire_f1(hand_pose_frame: Dict[str, np.ndarray], is_right: bool = True):
    required = [
        "wrist",
        "thumb_metacarpal",
        "thumb_distal",
        "index_distal",
        "middle_distal",
        "ring_distal",
        "pinky_distal",
    ]
    if any(k not in hand_pose_frame for k in required):
        return None

    inspire_angles = np.zeros(6, dtype=np.float64)
    wrist_inv = np.linalg.inv(hand_pose_frame["wrist"])

    for i, finger_name in enumerate(["thumb", "index", "middle", "ring", "pinky"]):
        metacarpal = f"{finger_name}_metacarpal"
        distal = f"{finger_name}_distal"

        tip_pos = wrist_inv @ hand_pose_frame[distal]

        if finger_name != "thumb":
            angle = np.arctan2(tip_pos[2, 1], tip_pos[1, 1])
            if angle < -np.pi / 2:
                angle = 2 * np.pi + angle
            inspire_angles[4 - i] = (1 - max(0.0, min(1.0, angle / np.pi))) * 840.0 + 900.0
        else:
            tip_position = tip_pos[:3, 3]
            finger_base_position = (wrist_inv @ hand_pose_frame[metacarpal])[:3, 3]
            tip_direction = tip_position - finger_base_position
            norm = np.linalg.norm(tip_direction)
            if norm < 1e-8:
                return None
            tip_direction = tip_direction / norm

            if is_right:
                inspire_angles[5] = 660 * tip_direction[0] + 700
                if tip_direction[0] < 0:
                    inspire_angles[4] = 165 * np.arctan(tip_direction[2] / abs(tip_direction[0])) + 968
                else:
                    inspire_angles[4] = 1350
            else:
                inspire_angles[5] = -500 * tip_direction[0] + 850
                if tip_direction[0] > 0:
                    inspire_angles[4] = -125 * np.arctan(tip_direction[2] / abs(tip_direction[0])) + 1025
                else:
                    inspire_angles[4] = 1350

    inspire_angles = np.clip(np.rint(inspire_angles), 0, 1740).astype(np.int32)
    inspire_angles[5] = np.clip(inspire_angles[5], 600, 1800)
    inspire_angles[4] = np.clip(inspire_angles[4], 1100, 1350)

    return inspire_angles



def kistar(hand_pose_frame):
    required = [
        "wrist",
        "thumb_metacarpal",
        "thumb_proximal",
        "thumb_distal",
        "index_metacarpal",
        "index_proximal",
        "index_intermediate",
        "index_distal",
        "middle_metacarpal",
        "middle_proximal",
        "middle_intermediate",
        "middle_distal",
        "ring_metacarpal",
        "ring_proximal",
        "ring_intermediate",
        "ring_distal",
    ]
    if any(k not in hand_pose_frame for k in required):
        return None

    def _angle_to_raw(v, max_angle=1.8):
        v = float(np.clip(v, 0.0, max_angle))
        return int(np.clip(np.rint(v / max_angle * 4096.0), 0, 4096))

    # intermediate representation in radians (allegro-like extraction)
    allegro_angles = np.zeros(16, dtype=np.float64)
    joint_name_list = ["metacarpal", "proximal", "intermediate", "distal"]
    wrist_inv = np.linalg.inv(hand_pose_frame["wrist"])

    for i, finger_name in enumerate(["index", "middle", "ring"]):
        metacarpal = f"{finger_name}_metacarpal"
        distal = f"{finger_name}_distal"

        tip_position = (wrist_inv @ hand_pose_frame[distal])[:3, 3]
        finger_base_position = (wrist_inv @ hand_pose_frame[metacarpal])[:3, 3]
        tip_direction = tip_position - finger_base_position
        norm = np.linalg.norm(tip_direction)
        if norm < 1e-8:
            return None
        tip_direction = tip_direction / norm

        if tip_direction[1] > 0.9:
            allegro_angles[4 * i] = 0.0
        else:
            allegro_angles[4 * i] = np.arctan2(tip_direction[0], tip_direction[2]) * (0.9 - tip_direction[1])

        for j in range(3):
            parent_name = f"{finger_name}_{joint_name_list[j]}"
            joint_name = f"{finger_name}_{joint_name_list[j + 1]}"
            rot_mat = np.linalg.inv(hand_pose_frame[parent_name][:3, :3]) @ hand_pose_frame[joint_name][:3, :3]
            v = rot_mat[1, 1] if rot_mat[2, 1] >= 0 else 1
            v = max(-1.0, min(1.0, v))
            allegro_angles[4 * i + j + 1] = np.arccos(v)
        allegro_angles[4 * i + 1] = (allegro_angles[4 * i + 1] - 0.35) * 1.5

    thumb_meta = hand_pose_frame["wrist"][:3, :3].T @ hand_pose_frame["thumb_metacarpal"][:3, :3]
    thumb_meta_angle = R.from_matrix(thumb_meta).as_euler("xyz")
    allegro_angles[12] = thumb_meta_angle[0]
    allegro_angles[13] = -thumb_meta_angle[2] - 1.57
    for i, (parent_name, joint_name) in enumerate(
        [("thumb_metacarpal", "thumb_proximal"), ("thumb_proximal", "thumb_distal")]
    ):
        rot_mat = np.linalg.inv(hand_pose_frame[parent_name][:3, :3]) @ hand_pose_frame[joint_name][:3, :3]
        allegro_angles[14 + i] = rot_mat[2, 1] * 1.2

    # KISTAR raw command (0=open, 4096=closed), 16-dof with fixed indices
    kistar_raw = np.zeros(16, dtype=np.int32)

    # thumb: active [0,2,3], fixed [1]
    kistar_raw[0] = _angle_to_raw(allegro_angles[13], max_angle=0.35)    
    kistar_raw[1] = -_angle_to_raw(allegro_angles[13], max_angle=0.35)

    kistar_raw[2] = _angle_to_raw(allegro_angles[14], max_angle=1.5)
    kistar_raw[3] = _angle_to_raw(allegro_angles[15], max_angle=1.5)

    # index: active [5,6,7], fixed [4]
    kistar_raw[5] = _angle_to_raw(allegro_angles[1], max_angle=1.8)
    kistar_raw[6] = _angle_to_raw(allegro_angles[2], max_angle=1.8)
    kistar_raw[7] = _angle_to_raw(allegro_angles[3], max_angle=1.8)

    # middle: active [9,10,11], fixed [8]
    kistar_raw[9] = _angle_to_raw(allegro_angles[5], max_angle=1.8)
    kistar_raw[10] = _angle_to_raw(allegro_angles[6], max_angle=1.8)
    kistar_raw[11] = _angle_to_raw(allegro_angles[7], max_angle=1.8)

    # ring: active [13,14,15], fixed [12]
    kistar_raw[13] = _angle_to_raw(allegro_angles[9], max_angle=1.8)
    kistar_raw[14] = _angle_to_raw(allegro_angles[10], max_angle=1.8)
    kistar_raw[15] = _angle_to_raw(allegro_angles[11], max_angle=1.8)

    # fixed joints: 1,4,8,12 are already 0
    return kistar_raw


def allegro_v5(hand_pose_frame):
    hand_joint_angle = np.zeros((20,3))
    allegro_angles = np.zeros(16)
    # for finger_id in range(4):
    #     for joint_id in range(4):
    #         if joint_id == 0:
    #             rot_mat = np.linalg.inv(hand_pose_frame[0,:3,:3]) @ hand_pose_frame[finger_id * 4 + joint_id + 1, :3,:3]
    #         else:
    #             rot_mat = np.linalg.inv(hand_pose_frame[hand_index.hand_index_parent[finger_id * 4 + joint_id+1], :3,:3]) @ hand_pose_frame[finger_id * 4 + joint_id + 1, :3,:3]
    #         hand_joint_angle[finger_id * 4 + joint_id + 1] = Rotation.from_matrix(rot_mat).as_euler("zyx")
    
    # zyx euler angle in hand frame = zxy axis angle in robot frame
    
    # Ring
    joint_name_list = ["metacarpal", "proximal", "intermediate","distal"]
    for i, finger_name in enumerate(["index", "middle", "ring"]):
        metacarpal = finger_name + "_metacarpal"
        distal = finger_name + "_distal"
        
        tip_position = (np.linalg.inv(hand_pose_frame["wrist"]) @ hand_pose_frame[distal])[:3, 3]
        finger_base_position = (np.linalg.inv(hand_pose_frame["wrist"]) @ hand_pose_frame[metacarpal])[:3, 3]
        
        tip_position = tip_position - finger_base_position
        tip_direction  = tip_position / np.linalg.norm(tip_position)

        if tip_direction[1] > 0.9:
            allegro_angles[4*i] = 0
        else:
            allegro_angles[4*i] = np.arctan(tip_direction[0] / tip_direction[2]) * (0.9-tip_direction[1])
        
        for j in range(3):
            parent_name = finger_name + "_" + joint_name_list[j]
            joint_name = finger_name + "_" + joint_name_list[j+1]
            rot_mat = np.linalg.inv(hand_pose_frame[parent_name][:3,:3]) @ hand_pose_frame[joint_name][:3,:3]
            v = rot_mat[1, 1] if rot_mat[2, 1] >= 0 else 1
            v = max(-1, min(1, v))
            allegro_angles[4*i+j+1] = np.arccos(v)
        allegro_angles[4*i+1] = (allegro_angles[4*i+1]-0.35) * 1.5


    # Thumb
    thumb_meta = np.dot(hand_pose_frame["wrist"][:3,:3].T, hand_pose_frame["thumb_metacarpal"][:3,:3])
    thumb_meta_angle = R.from_matrix(thumb_meta).as_euler("xyz")
    allegro_angles[12] = thumb_meta_angle[0] 
    allegro_angles[13] = -thumb_meta_angle[2]-1.57

    for i, (parent_name, joint_name) in enumerate([("thumb_metacarpal", "thumb_proximal"),("thumb_proximal", "thumb_distal")]):
        rot_mat = np.linalg.inv(hand_pose_frame[parent_name][:3,:3]) @ hand_pose_frame[joint_name][:3,:3]
        allegro_angles[14+i] = rot_mat[2, 1] * 1.2

    return allegro_angles
