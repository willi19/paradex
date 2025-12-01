import numpy as np
from scipy.spatial.transform import Rotation as R
import time

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
        