import argparse
import numpy as np
import os
from paradex.utils.file_io import get_robot_urdf_path, rsc_path
from paradex.geometry.coordinate import DEVICE2WRIST
from scipy.spatial.transform import Rotation as R
from paradex.robot import RobotWrapper
from paradex.robot.urdf import generate_urdf

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--arm", type=str, required=True)
    parser.add_argument("--hand", type=str, required=True)
    
    args = parser.parse_args()
    
    arm_urdf_path = get_robot_urdf_path(args.arm)
    hand_urdf_path = get_robot_urdf_path(args.hand)
    
    arm_model = RobotWrapper(arm_urdf_path)
    parent_link = arm_model.get_end_links()[0]
    
    hand_model = RobotWrapper(hand_urdf_path)
    child_link = "wrist"

    xacro_path = os.path.join(rsc_path, "robot", "robot_combined.urdf.xacro")
    output_path = os.path.join(rsc_path, "robot", f"{args.arm}_{args.hand}.urdf")
    
    arm2global = DEVICE2WRIST[args.arm]
    hand2global = DEVICE2WRIST[args.hand]
    
    arm2wrist = np.linalg.inv(arm2global) @ hand2global
    
    xyz = arm2wrist[:3, 3]
    rpy = R.from_matrix(arm2wrist[:3,:3]).as_euler('xyz')
    
    xyz_str = ' '.join(f'{v:.5f}' for v in xyz)
    rpy_str = ' '.join(f'{v:.5f}' for v in rpy)
    
    arg_dict = {
        "arm_file": arm_urdf_path,
        "hand_file": hand_urdf_path,
        "parent_link": parent_link,
        "child_link": child_link,
        "xyz": xyz_str,
        "rpy": rpy_str,
    }

    generate_urdf(xacro_path, output_path, arg_dict)
