import numpy as np
from paradex.transforms.coordinate import DEVICE2WRIST,  DEVICE2GLOBAL
from paradex.retargetor.hand_regargetor import inspire, allegro

class Retargetor(): # Input is only from Xsens
    def __init__(self, arm_name=None, hand_name=None, hand_side="Right"):
        self.arm_name = arm_name
        self.hand_name = hand_name
        
        if arm_name not in [None, "xarm", "franka"]:
            raise ValueError("Invalid arm name")
        if hand_name not in [None, "inspire", "allegro"]:
            raise ValueError("Invalid hand name")
        
        if hand_side not in ["Right", "Left", "Bimanual"]:
            raise ValueError("Invalid hand side")
        
        if hand_side == "Bimanual" and arm_name is not None:
            raise ValueError("Bimanual teleop not supported for arm control")

        self.hand_side = hand_side

        self.hand_retargetor = None
        if self.hand_name == "inspire":
            self.hand_retargetor = inspire
        elif self.hand_name == "allegro":
            self.hand_retargetor = allegro
        else:
            self.hand_retargetor = None

        if self.arm_name is not None:
            self.device2wrist = DEVICE2WRIST[self.arm_name + "_" + hand_side].copy()
            self.device2global = DEVICE2GLOBAL[self.arm_name].copy()
        else:
            self.device2wrist = DEVICE2WRIST[self.hand_name].copy()
            self.device2global = np.eye(4)

    def get_action(self, data):

        


        if self.hand_side == "Bimanual":
            if self.hand_name is not None:
                hand_action_left = self.hand_retargetor(data["Left"])
                hand_action_right = self.hand_retargetor(data["Right"])
            else:
                hand_action = None
                
            return None, hand_action_left, hand_action_right
        else: 
            if self.init_human_pose is None:
                self.init_human_pose = data[self.hand_side]["wrist"].copy()
            # print(self.device2wrist.T.shape)
            delta_wrists_R = self.device2wrist[:3,:3].T @ np.linalg.inv(self.init_human_pose[:3,:3]) @ data[self.hand_side]["wrist"][:3,:3] @ self.device2wrist[:3,:3]
            delta_wrists_t = data[self.hand_side]["wrist"][:3,3] - self.init_human_pose[:3,3]

            robot_wrist_pose = np.zeros((4,4))
            robot_wrist_pose[:3,:3] = self.init_robot_pose[:3,:3] @ delta_wrists_R
                    
            robot_wrist_pose[:3,3] = delta_wrists_t + self.init_robot_pose[:3,3]
            robot_wrist_pose[3,3] = 1
            
            self.last_arm_pose = robot_wrist_pose.copy()
            arm_action = robot_wrist_pose.copy()
                
            if self.hand_name is not None:
                hand_action = self.hand_retargetor(data[self.hand_side])
            else:
                hand_action = None
                
            return arm_action, hand_action


    def start(self, home_pose):
        self.init_robot_pose = home_pose.copy()
        self.last_arm_pose = home_pose.copy()
        self.init_human_pose = None

    def stop(self):
        self.init_human_pose = None
        self.init_robot_pose = self.last_arm_pose.copy()