import subprocess
import argparse
import numpy as np
import os
from paradex.utils.file_io import get_robot_urdf_path, rsc_path
from paradex.robot.urdf import get_end_links, get_root_links
from paradex.geometry.coordinate import DEVICE2WRIST

# currently not in use

def generate_urdf(xacro_path, output_path, args_dict):
    # Prepare command
    cmd = ["xacro", str(xacro_path)]

    # Add arguments
    for key, value in args_dict.items():
        cmd.append(f"{key}:={value}")

    # Write output to file
    with open(output_path, "w") as f:
        subprocess.run(cmd, stdout=f, check=True)

    print(f"Generated URDF saved to: {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--arm", type=str, default="arm1.urdf")
    parser.add_argument("--hand", type=str, default="hand1.urdf")
    
    args = parser.parse_args()
    
    arm_urdf_path = get_robot_urdf_path(args.arm)
    hand_urdf_path = get_robot_urdf_path(args.hand)
    
    parent_link = get_end_links(arm_urdf_path)
    child_link = get_root_links(hand_urdf_path)

    xacro_path = os.path.join(rsc_path, "robot", "robot_combined.urdf.xacro")
    output_path = os.path.join(rsc_path, "robot", f"{args.arm}_{args.hand}_gen.urdf")
    
    arm2global = DEVICE2WRIST[args.arm]
    hand2global = DEVICE2WRIST[args.hand]
    
    arm2wrist = np.linalg.inv(arm2global) @ hand2global
    print(arm2global)
    print()
    # 0  1 0
    # -1 0 0
    # 0  0 1

    arg_dict = {
        "arm_file": args.arm_file,
        "hand_file": args.hand_file,
        "parent_link": args.parent_link,
        "child_link": args.child_link,
        "xyz": args.xyz,
        "rpy": args.rpy,
    }

    # generate_urdf(xacro_path, output_path, arg_dict)
