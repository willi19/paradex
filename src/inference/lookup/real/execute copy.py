import time
import numpy as np
from paradex.inference.get_lookup_traj import get_traj

from paradex.io.robot_controller import XArmController, AllegroController, InspireController
import chime
import argparse

from paradex.geometry.coordinate import DEVICE2WRIST
from paradex.robot import RobotWrapper
import os
from paradex.utils.file_io import rsc_path

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--hand", required=True)
    parser.add_argument("--arm", required=True)
    
    args = parser.parse_args()
    
    robot = RobotWrapper(os.path.join(rsc_path, "robot", f"{args.arm}_{args.hand}.urdf"))
    palm_id = robot.get_link_index("palm_link")
    link6_id = robot.get_link_index("link6")
    
    palm_6d = robot.get_link_pose(palm_id)
    link6_6d = robot.get_link_pose(link6_id)
    
    LINK62WRIST = np.linalg.inv(palm_6d) @ link6_6d
        
    bottle_height = 0.07# 0.039
    pick_tx = 0.5
    pick_ty = -0.2

    place_tx = 0.5
    place_ty = 0.2

    pick_6D = np.eye(4)
    place_6D = np.eye(4)

    pick_6D[:3,3] = np.array([pick_tx, pick_ty, bottle_height])
    place_6D[:3,3] = np.array([place_tx, place_ty, bottle_height])

    demo_idx = 1

    pick_traj = np.load(f"lookup/bottle/{demo_idx}/pick.npy")
    place_traj = np.load(f"lookup/bottle/{demo_idx}/place.npy")

    pick_hand_traj = np.load(f"lookup/bottle/{demo_idx}/pick_hand.npy")
    place_hand_traj = np.load(f"lookup/bottle/{demo_idx}/place_hand.npy")

    traj, hand_traj = get_traj(pick_traj, pick_6D, place_traj, place_6D, pick_hand_traj, place_hand_traj)

    # hand_controller = None
    # if args.hand is not None:
    #     if args.hand == "allegro":
    #         hand_controller = AllegroController()
    #     elif args.hand == "inspire":
    #         hand_controller = InspireController()
    print(traj[0] @ LINK62WRIST)
    if args.arm == "xarm":
        arm_controller = XArmController()
        arm_controller.home_robot(traj[0] @ LINK62WRIST)
        
    home_start_time = time.time()
    while not arm_controller.is_ready():
        if time.time() - home_start_time > 0.5:
            chime.warning()
            home_start_time = time.time()
        time.sleep(0.01)
    
    chime.success()
        
    for i in range(len(traj)):
        l6_pose = traj[i] @ LINK62WRIST
        arm_controller.set_action(l6_pose)
        time.sleep(0.03)  # Simulate time taken for each action
    
    if args.arm is not None:
        arm_controller.quit()
        