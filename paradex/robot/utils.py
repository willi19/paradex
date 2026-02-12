import os

from paradex.utils.path import rsc_path

def get_robot_urdf_path(arm_name=None, hand_name=None):
    if arm_name == "openarm":
        if hand_name is None:
            return os.path.join(rsc_path, "robot", "openarm", "openarm_bimanual.urdf")
        elif hand_name == "inspire_f1":
            return os.path.join(rsc_path, "robot", "openarm", "openarm_rh56f1.urdf")
    if arm_name == None:
        return os.path.join(rsc_path, "robot", hand_name+"_float.urdf")
    
    if hand_name == None:
        return os.path.join(rsc_path, "robot", arm_name+".urdf")
    
    return os.path.join(rsc_path, "robot", f"{arm_name}_{hand_name}.urdf")