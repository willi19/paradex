import os
from typing import Dict
import numpy as np
import yaml
import pickle
from pathlib import Path

def find_latest_directory(directory):
    """
    Get the latest directory in the specified directory.

    Parameters:
    - directory: Directory containing timestamped directories.

    Returns:
    - latest_dir: Latest directory in the specified directory.
    """
    
    dirs = [d for d in os.listdir(directory)] 
    if not dirs:
        return None
        
    latest_dir = max(dirs, key=str)
    return latest_dir

def find_latest_index(directory):
    """
    Get the latest directory in the specified directory.

    Parameters:
    - directory: Directory containing timestamped directories.

    Returns:
    - latest_dir: Latest directory in the specified directory.
    """
    if not os.path.exists(directory):
        return -1
    
    dirs = [d for d in os.listdir(directory)] 

    if not dirs:
        return -1
        
    latest_dir = max(dirs, key=int)    
    return latest_dir

def is_image_file(file):
    return file.endswith(".png") or file.endswith(".jpg") or file.endswith(".jpeg")

def load_images(dir):
    return [os.path.join(dir, f) for f in os.listdir(dir) if is_image_file(f)]

# # File io
def load_obj_traj(demo_path):
    # Load object trajectory
    obj_traj = pickle.load(open(os.path.join(demo_path, "obj_traj.pickle"), "rb"))
    return obj_traj


def load_robot_traj(demo_path):
    # Load robot trajectory
    arm_traj = np.load(os.path.join(demo_path, "arm", "state.npy"))
    hand_traj = np.load(os.path.join(demo_path, "hand", "state.npy"))
    robot_traj = np.concatenate([arm_traj, hand_traj], axis=-1)

    return robot_traj

def load_robot_traj_prev(demo_path):
    # Load robot trajectory
    arm_traj = np.load(os.path.join(demo_path, "robot_qpos.npy"))
    # robot_traj = np.concatenate([arm_traj, hand_traj], axis=-1)

    return arm_traj


def load_robot_target_traj(demo_path):
    arm_traj = np.load(os.path.join(demo_path, "arm", "action.npy"))
    hand_traj = np.load(os.path.join(demo_path, "hand", "action.npy"))
    robot_traj = np.concatenate([arm_traj, hand_traj], axis=-1)
    return robot_traj

def load_contact_value(demo_path):
    contact_value = np.load(os.path.join(demo_path, "contact", "data.npy"))
    return contact_value

def load_mesh(obj_name):
    import open3d as o3d

    mesh = o3d.io.read_triangle_mesh(
        os.path.join(rsc_path, "object", obj_name, f"{obj_name}.obj")
    )
    return mesh

def load_yaml(file_path:str) -> Dict:
    """Load yaml file and return as dictionary. If file_path is a dictionary, return as is.

    Args:
        file_path: File path to yaml file or dictionary.

    Returns:
        Dict: Dictionary containing yaml file content.
    """
    if isinstance(file_path, str):
        with open(file_path) as file_p:
            yaml_params = yaml.load(file_p, Loader=yaml.Loader)
    return yaml_params

def remove_home(path):
    path = Path(path)
    return str(path.relative_to(Path.home()))