import os
import numpy as np
from scipy.spatial.transform import Rotation as R

from paradex.utils.file_io import rsc_path

def to_quat(obj_pose):
    ret = np.zeros(7)   
    rotation_matrix = obj_pose[:3, :3]
    r = R.from_matrix(rotation_matrix)
    quat_xyzw = r.as_quat()  # scipy는 xyzw 순서로 반환
    ret[0] = quat_xyzw[3]
    ret[1:4] = quat_xyzw[0:3]
    ret[4:7] = obj_pose[:3, 3]
    return ret

def load_world_config(obj_dict):
    world_config_dict = {'mesh':{}, 'cuboid':{}}
    world_config_dict["cuboid"]["table"] = {
            "pose": [0.0, 0.0, -0.041, 0.0, 0.0, 0.0, 1.0],
            "dims": [1.0, 1.0, 0.082],
            "color": [0.8, 0.6, 0.4, 1.0]
        }
    
    for obj_name, obj_info in obj_dict.items():
        world_config_dict["mesh"][str(obj_name)] = {
                "pose": to_quat(obj_info["pose"]).tolist(),
                "file_path": obj_info["file_path"],
            }
        
    return world_config_dict