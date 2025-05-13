import os
import json
import numpy as np


rsc_path = os.path.join(
    os.path.dirname(__file__),
    "..",
    "..",
    "rsc",
)

home_dir = os.path.expanduser("~")
shared_dir = os.path.join(home_dir, "shared_data")
calib_path_list = [os.path.join(home_dir, f"captures{i}", "calibration") for i in range(1,3)]
capture_path_list = [os.path.join(home_dir, f"captures{i}") for i in range(1,3)]
download_dir = os.path.join(home_dir, "download")
cam_param_dir = os.path.join(shared_dir, "cam_param")
handeye_calib_path = os.path.join(shared_dir, "handeye_calibration")
config_dir = os.path.join(os.path.dirname(__file__), "..", "..", "config")

def get_video_list(video_dir):
    """
    Get a list of video files in the specified directory.

    Parameters:
    - video_dir: Directory containing video files.

    Returns:
    - video_list: List of video files in the directory.
    """
    video_list = []
    for f in os.listdir(video_dir):
        if f.endswith(".avi") or f.endswith(".mp4"):
            video_name = f.split("-")[0] # {serial_num}_{date}
            timestamp_path = os.path.join(video_dir, video_name+"_timestamp.json")
            if not os.path.exists(timestamp_path):
                video_list.append((os.path.join(video_dir, f), None))
                continue
            video_list.append((os.path.join(video_dir, f), timestamp_path))

    return video_list

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
        print("No valid directories found.")
        return "0"
        
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
        
    dirs = [d for d in os.listdir(directory)] 

    if not dirs:
        return -1
        
    latest_dir = max(dirs, key=int)    
    return latest_dir

def load_cam_param(name=None):
    if name == None:
        name = find_latest_directory(cam_param_dir)
    intrinsic_data = json.load(open(os.path.join(cam_param_dir, name, "intrinsics.json")))
    intrinsic = {}
    for serial, values in intrinsic_data.items():
        intrinsic[serial] = {
            "intrinsics_original": np.array(values["original_intrinsics"]).reshape(3, 3),
            "intrinsics_undistort": np.array(values["Intrinsics"]).reshape(3, 3),
            "intrinsics_warped": np.array(values["Intrinsics_warped"]).reshape(3, 3),
            "dist_params": np.array(values["dist_param"]),
            "height": values["height"],  # Scalar values remain unchanged
            "width": values["width"],
        }
    extrinsic_data = json.load(open(os.path.join(cam_param_dir, name, "extrinsics.json")))
    extrinsic = {}
    for serial, values in extrinsic_data.items():
        extrinsic[serial] = np.array(values).reshape(3, 4)
    return intrinsic, extrinsic

def load_cam_param_temp(name=None):
    if name == None:
        name = find_latest_directory(cam_param_dir)
    intrinsic_data = json.load(open(os.path.join(cam_param_dir, name, "intrinsics.json")))
    intrinsic = {}
    for serial, values in intrinsic_data.items():
        intrinsic[serial] = {
            "intrinsics_original": np.array(values["original_intrinsics"]).reshape(3, 3),
            "intrinsics_undistort": np.array(values["Intrinsics"]).reshape(3, 3),
            "intrinsics_warped": np.array(values["Intrinsics_warped"]).reshape(3, 3),
            "dist_params": np.array(values["dist_param"]),
            "height": values["height"],  # Scalar values remain unchanged
            "width": values["width"],
        }
    extrinsic_data = json.load(open(os.path.join(cam_param_dir, name, "extrinsics_temp.json")))
    extrinsic = {}
    for serial, values in extrinsic_data.items():
        extrinsic[serial] = np.array(values).reshape(3, 4)
    return intrinsic, extrinsic

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
        os.path.join(rsc_path, obj_name, f"{obj_name}.obj")
    )
    return mesh

def load_camparam(demo_path):
    intrinsic_data = json.load(open(os.path.join(demo_path, "cam_param", "intrinsics.json"), "r"))
    intrinsic = {}
    for serial, values in intrinsic_data.items():
        intrinsic[serial] = {
            "intrinsics_original": np.array(values["original_intrinsics"]).reshape(3, 3),
            "intrinsics_undistort": np.array(values["Intrinsics"]).reshape(3, 3),
            "intrinsics_warped": np.array(values["Intrinsics_warped"]).reshape(3, 3),
            "dist_params": np.array(values["dist_param"]),
            "height": values["height"],  # Scalar values remain unchanged
            "width": values["width"],
        }
    extrinsic_data = json.load(open(os.path.join(demo_path, "cam_param", "extrinsics.json"), "r"))
    extrinsic = {}
    for serial, values in extrinsic_data.items():
        extrinsic[serial] = np.array(values).reshape(3, 4)
    return intrinsic, extrinsic

def load_c2r(demo_path):
    C2R = np.load(os.path.join(demo_path, "C2R.npy"))
    return C2R