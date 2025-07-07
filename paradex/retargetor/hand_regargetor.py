import numpy as np

def allegro(hand_pose_frame):
    allegro_angles = np.zeros(16)
    # hand_joint_angle = np.zeros((20,3))
    
    # # for finger_id in range(4):
    # #     for joint_id in range(4):
    # #         if joint_id == 0:
    # #             rot_mat = np.linalg.inv(hand_pose_frame[0,:3,:3]) @ hand_pose_frame[finger_id * 4 + joint_id + 1, :3,:3]
    # #         else:
    # #             rot_mat = np.linalg.inv(hand_pose_frame[hand_index.hand_index_parent[finger_id * 4 + joint_id+1], :3,:3]) @ hand_pose_frame[finger_id * 4 + joint_id + 1, :3,:3]
    # #         hand_joint_angle[finger_id * 4 + joint_id + 1] = Rotation.from_matrix(rot_mat).as_euler("zyx")
    
    # # zyx euler angle in hand frame = zxy axis angle in robot frame
    
    # # Ring
    # for i in range(3):
    #     tip_position = (np.linalg.inv(hand_pose_frame[0]) @ hand_pose_frame[7+4*i])[:3, 3]
    #     finger_base_position = (np.linalg.inv(hand_pose_frame[0]) @ hand_pose_frame[4+4*i])[:3, 3]
        
    #     tip_position = tip_position - finger_base_position
    #     tip_direction  = tip_position / np.linalg.norm(tip_position)

    #     if tip_direction[1] < -0.7:
    #         allegro_angles[4*i] = 0
    #     else:
    #         allegro_angles[4*i] = np.arctan(tip_direction[0] / tip_direction[2]) * 0.5

    #     for j in range(3):
    #         rot_mat = np.linalg.inv(hand_pose_frame[hand_index.hand_index_parent[4*i+j+5], :3,:3]) @ hand_pose_frame[5+4*i+j, :3,:3]
    #         v = rot_mat[1, 1]
    #         v = max(-1, min(1, v))
    #         allegro_angles[4*i+j+1] = np.arccos(v)
    #     allegro_angles[4*i+1] *= 1.2
    #         # allegro_angles[4*i+j+1] = np.arccos(rot_mat[1, 1])
            

    # # Thumb
    # thumb_meta = np.dot(hand_pose_frame[0,:3,:3].T, hand_pose_frame[1,:3,:3])
    # thumb_meta_angle = Rotation.from_matrix(thumb_meta).as_euler("xyz")
    # allegro_angles[12] = thumb_meta_angle[0]  
    # allegro_angles[13] = thumb_meta_angle[1] - 0.5

    # for i in range(2):
    #     rot_mat = np.linalg.inv(hand_pose_frame[hand_index.hand_index_parent[i+2], :3,:3]) @ hand_pose_frame[i+2, :3,:3]
    #     allegro_angles[14+i] = rot_mat[2, 1] * 1.2

    return allegro_angles

def inspire(self, hand_pose_frame):
    inspire_angles = np.zeros(6)

    # # print(hand_pose_frame[0,:3,:3])
    # for i in range(5):
    #     print(hand_pose_frame[0])
    #     tip_position = (np.linalg.inv(hand_pose_frame[0]) @ hand_pose_frame[3+4*i])[:3, 3]
    #     finger_base_position = (np.linalg.inv(hand_pose_frame[0]) @ hand_pose_frame[1+4*i])[:3, 3]

    #     tip_position = tip_position - finger_base_position
    #     tip_direction  = tip_position / np.linalg.norm(tip_position)
        
        
    #     if i != 0:
    #         if tip_direction[1] > 0:
    #             y_dir = min(1, ((max(-1.0, tip_direction[1]-0.03)+1) * 1.57) - 1)
    #             inspire_angles[4-i] = np.arccos(y_dir) / np.pi * 1000
    #         else:
    #             y_dir = max(-1, ((max(-1.0, tip_direction[1]-0.03)+1) * 1.57) - 1)
    #             inspire_angles[4-i] = np.arccos(y_dir) / np.pi * 1000

    #     else:
    #         if tip_direction[0] > 0:
    #             inspire_angles[5] = 1000 - np.arctan(-tip_direction[2] / abs(tip_direction[0])) / np.pi * 2000
    #             inspire_angles[4] = np.arccos(-tip_direction[1]) * 2000 - 1000 # no divide by pi for better range
    #         else:
    #             inspire_angles[5] = 0
    #             inspire_angles[4] = np.arcsin(-tip_direction[2]) / np.pi * 2000  * 3.5 - 1000

    #         print(tip_position, tip_direction, inspire_angles[5])
            
    #         # print(inspire_angles[4], inspire_angles[5])

    return inspire_angles
        