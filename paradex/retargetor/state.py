import numpy as np

class HandStateExtractor:
    def __init__(self):
        self.parent = {
            "proximal":"metacarpal", 
            "intermediate":"proximal", 
            "distal":"intermediate"
        }

    def get_state(self, pose_data):
        straight = self.check_straight(pose_data)
        if straight[0] and straight[1] and not straight[2] and not straight[3]: # V pose
            return 2
        
        if straight[0] and straight[1] and straight[2] and straight[3]: # Fist pose
            return 1
        
        if straight[0] and not straight[1] and not straight[2] and not straight[3]: # Open pose
            return 3
        
        return 0
    
    def check_straight(self, pose_data):
        ret = [True, True, True, True] # if one of the joints is not straight, return False
        # 5 6 7
        # 9 10 11
        # 13 14 15
        # 17 18 19
        for finger_id, finger_name in enumerate(["index", "middle", "ring", "pinky"]):
            for joint_name in ["metacarpal", "proximal", "intermediate", "distal"]:
                finger_joint_name = finger_name + "_" + joint_name
                if joint_name == "metacarpal":
                    if pose_data[finger_joint_name][2,1] > 0.8:
                        ret[finger_id] = False
                else:
                    parent_joint_name = finger_name + "_" + self.parent[joint_name]
                    rel_pose = np.linalg.inv(pose_data[parent_joint_name]) @ pose_data[finger_joint_name]
                    if rel_pose[2,1] > 0.8:
                        ret[finger_id] = False
        return ret