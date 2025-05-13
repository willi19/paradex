import numpy as np
from scipy.spatial.transform import Rotation
from paradex.io.xsens import hand_index

import transforms3d as t3d
import os
from paradex.robot.robot_wrapper import RobotWrapper
from paradex.utils.file_io import rsc_path

LINK62PALM = np.array(
    [
        [0, -1, 0, 0],
        [0, 0, 1, 0],
        [-1, 0, 0, 0],
        [0, 0, 0, 1],
    ]
)

XSENS2ISAAC = np.array(
    [
        [0, -1, 0, 0],
        [0, 0, -1, 0],
        [1, 0, 0, 0],
        [0, 0, 0, 1],
    ]
)

def homo2cart(h):
    if h.shape == (4, 4):
        t = h[:3, 3]
        R = h[:3, :3]

        axis, angle = t3d.axangles.mat2axangle(R)
        axis_angle = axis * angle
    else:
        raise ValueError("Invalid input shape.")
    return np.concatenate([t, axis_angle])

    
robot = RobotWrapper(
    os.path.join(rsc_path, "allegro", "allegro.urdf")
)
link_list = ["palm_link", "thumb_base", "thumb_proximal", "thumb_medial", "thumb_distal", "thumb_tip", 
             "index_base", "index_proximal", "index_medial", "index_distal", "index_tip", 
             "middle_base", "middle_proximal", "middle_medial", "middle_distal", "middle_tip", 
             "ring_base", "ring_proximal", "ring_medial", "ring_distal", "ring_tip"]

def safety_bound(target_action):
    angle = np.linalg.norm(target_action[3:6])
    axis = target_action[3:6] / angle
    
    R = Rotation.from_rotvec(angle * axis).as_matrix()

    euler = Rotation.from_matrix(R).as_euler("XYZ")
    t = target_action[:3]

    pin_action = np.concatenate([t, euler, np.zeros(16)])
    robot.compute_forward_kinematics(pin_action)
    min_z = 10
    for link in link_list:
        link_index = robot.get_link_index(link)
        link_pose = robot.get_link_pose(link_index)
        min_z = min(min_z, link_pose[2,3])
    
    # print(max(0, 0.05 - min_z), min_z, link_min_name)
    target_action[2] += max(0, -0.05 - min_z)
    return target_action

class retargetor(): # Input is only from Xsens
    def __init__(self, arm_name=None, hand_name=None, home_arm_pose=None):
        self.arm_name = arm_name
        self.hand_name = hand_name
        if arm_name is not None and arm_name not in ["xarm", "franka"]:
            raise ValueError("Invalid arm name")
        if hand_name is not None and hand_name not in ["inspire", "allegro"]:
            raise ValueError("Invalid hand name")
        
        self.home_arm_pose = home_arm_pose.copy() #Wrist 4x4

        self.init_wrist_pose = None
        self.init_robot_pose = home_arm_pose.copy()

        self.last_arm_pose = home_arm_pose.copy()

    def get_action(self, data):
        if self.init_wrist_pose is None:
            self.init_wrist_pose = data["hand_pose"][0].copy()
        
        if self.arm_name == "xarm":
            try:
                delta_wrists_R = LINK62PALM[:3,:3] @ XSENS2ISAAC[:3,:3].T @ np.linalg.inv(self.init_wrist_pose[:3,:3]) @ data["hand_pose"][0][:3,:3] @ XSENS2ISAAC[:3,:3] @ LINK62PALM[:3,:3].T
            except:
                delta_wrists_R = np.eye(3)

            delta_wrists_t = data["hand_pose"][0][:3,3] - self.init_wrist_pose[:3,3]

            robot_wrist_pose = np.zeros((4,4))
            robot_wrist_pose[:3,:3] = self.init_robot_pose[:3,:3] @ delta_wrists_R
                    
            robot_wrist_pose[:3,3] = delta_wrists_t + self.init_robot_pose[:3,3]
            robot_wrist_pose[3,3] = 1

            self.last_arm_pose = robot_wrist_pose.copy()

            arm_action = homo2cart(robot_wrist_pose)
            arm_action = safety_bound(arm_action)
        else:
            arm_action = None
        
        if self.hand_name == "inspire":
            hand_action = self.inspire(data["hand_pose"])
        elif self.hand_name == "allegro":
            hand_action = self.allegro(data["hand_pose"])
        
        return arm_action, hand_action

    def reset(self):
        if self.arm_name is not None:
            self.init_robot_pose = self.home_arm_pose.copy()
            self.last_arm_pose = self.home_arm_pose.copy()
            self.init_wrist_pose = None

    def pause(self):
        if self.arm_name is not None:
            self.init_wrist_pose = None
            self.init_robot_pose = self.last_arm_pose.copy()

    def allegro(self, hand_pose_frame):
        allegro_angles = np.zeros(16)
        hand_joint_angle = np.zeros((20,3))
        
        # for finger_id in range(4):
        #     for joint_id in range(4):
        #         if joint_id == 0:
        #             rot_mat = np.linalg.inv(hand_pose_frame[0,:3,:3]) @ hand_pose_frame[finger_id * 4 + joint_id + 1, :3,:3]
        #         else:
        #             rot_mat = np.linalg.inv(hand_pose_frame[hand_index.hand_index_parent[finger_id * 4 + joint_id+1], :3,:3]) @ hand_pose_frame[finger_id * 4 + joint_id + 1, :3,:3]
        #         hand_joint_angle[finger_id * 4 + joint_id + 1] = Rotation.from_matrix(rot_mat).as_euler("zyx")
        
        # zyx euler angle in hand frame = zxy axis angle in robot frame
        
        # Ring
        for i in range(3):
            tip_position = (np.linalg.inv(hand_pose_frame[0]) @ hand_pose_frame[7+4*i])[:3, 3]
            finger_base_position = (np.linalg.inv(hand_pose_frame[0]) @ hand_pose_frame[4+4*i])[:3, 3]
            
            tip_position = tip_position - finger_base_position
            tip_direction  = tip_position / np.linalg.norm(tip_position)

            if tip_direction[1] < -0.7:
                allegro_angles[4*i] = 0
            else:
                allegro_angles[4*i] = np.arctan(tip_direction[0] / tip_direction[2]) * 0.5

            for j in range(3):
                rot_mat = np.linalg.inv(hand_pose_frame[hand_index.hand_index_parent[4*i+j+5], :3,:3]) @ hand_pose_frame[5+4*i+j, :3,:3]
                v = rot_mat[1, 1]
                v = max(-1, min(1, v))
                allegro_angles[4*i+j+1] = np.arccos(v)
            allegro_angles[4*i+1] *= 1.2
                # allegro_angles[4*i+j+1] = np.arccos(rot_mat[1, 1])
                

        # Thumb
        thumb_meta = np.dot(hand_pose_frame[0,:3,:3].T, hand_pose_frame[1,:3,:3])
        thumb_meta_angle = Rotation.from_matrix(thumb_meta).as_euler("xyz")
        allegro_angles[12] = thumb_meta_angle[0]  
        allegro_angles[13] = thumb_meta_angle[1] - 0.5

        for i in range(2):
            rot_mat = np.linalg.inv(hand_pose_frame[hand_index.hand_index_parent[i+2], :3,:3]) @ hand_pose_frame[i+2, :3,:3]
            allegro_angles[14+i] = rot_mat[2, 1] * 1.2

        return allegro_angles
    
    def inspire(self, hand_pose_frame):
        inspire_angles = np.zeros(6)

        # print(hand_pose_frame[0,:3,:3])
        for i in range(5):
            print(hand_pose_frame[0])
            tip_position = (np.linalg.inv(hand_pose_frame[0]) @ hand_pose_frame[3+4*i])[:3, 3]
            finger_base_position = (np.linalg.inv(hand_pose_frame[0]) @ hand_pose_frame[1+4*i])[:3, 3]

            tip_position = tip_position - finger_base_position
            tip_direction  = tip_position / np.linalg.norm(tip_position)
            
            
            if i != 0:
                if tip_direction[1] > 0:
                    y_dir = min(1, ((max(-1.0, tip_direction[1]-0.03)+1) * 1.57) - 1)
                    inspire_angles[4-i] = np.arccos(y_dir) / np.pi * 1000
                else:
                    y_dir = max(-1, ((max(-1.0, tip_direction[1]-0.03)+1) * 1.57) - 1)
                    inspire_angles[4-i] = np.arccos(y_dir) / np.pi * 1000

            else:
                if tip_direction[0] > 0:
                    inspire_angles[5] = 1000 - np.arctan(-tip_direction[2] / abs(tip_direction[0])) / np.pi * 2000
                    inspire_angles[4] = np.arccos(-tip_direction[1]) * 2000 - 1000 # no divide by pi for better range
                else:
                    inspire_angles[5] = 0
                    inspire_angles[4] = np.arcsin(-tip_direction[2]) / np.pi * 2000  * 3.5 - 1000

                print(tip_position, tip_direction, inspire_angles[5])
                
                # print(inspire_angles[4], inspire_angles[5])

        return inspire_angles