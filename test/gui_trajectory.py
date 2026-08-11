import os
import numpy as np
import glob
import json
from paradex.utils.path import shared_dir
from paradex.utils.file_io import find_latest_directory
from paradex.io.robot_controller.gui_controller import RobotGUIController
from paradex.io.robot_controller import get_arm, get_hand
def convert(hand_pose):
    if hand_pose.shape == (16,):
        pose = hand_pose.copy()
        pose[:4] = hand_pose[12:]
        pose[4:] = hand_pose[:12]
    else:
        pose = hand_pose.copy()
        pose[:, :4] = hand_pose[:, 12:]
        pose[:, 4:] = hand_pose[:, :12]
    return pose
base_pose = np.array([[ 0.07842263, -0.16270738,  0.98355285,  0.4116784 ],
 [ 0.19462326, -0.96510918, -0.17517437, -0.42133483],
 [ 0.97773804,  0.20515989, -0.04401976,  0.41725355],
 [ 0.,          0.,          0.,          1.        ]])
obj_pose_idx = find_latest_directory(os.path.join(shared_dir, "inference", "bodex"))
approach_traj = np.load(os.path.join(shared_dir, "inference", "bodex", obj_pose_idx, "approach.npy"))
approach_traj[:, 6:] = convert(approach_traj[:, 6:])
pregrasp_pose = np.load(os.path.join(shared_dir, "inference", "bodex", obj_pose_idx, "pregrasp.npy"))
grasp_pose = np.load(os.path.join(shared_dir, "inference", "bodex", obj_pose_idx, "grasp_pose.npy"))
squeeze_pose = np.load(os.path.join(shared_dir, "inference", "bodex", obj_pose_idx, "squeeze_pose.npy"))
predefined_poses = {
    'base': base_pose
}
grasp_pose_dict = {
    'start': approach_traj[-1, 6:],
    'pregrasp': convert(pregrasp_pose),
    'grasp': convert(grasp_pose),
    'squeezed': convert(squeeze_pose)
}
arm = get_arm("xarm")
hand = get_hand("allegro")
print(">>> Controllers created.")
rgc = RobotGUIController(
    robot_controller=arm,
    hand_controller=hand,
    predefined_poses=predefined_poses,
    grasp_pose=grasp_pose_dict,
    approach_traj=approach_traj,
    lift_distance=100.0,
    place_distance=40.0
)
rgc.run()
input_info = json.load(open(os.path.join(shared_dir, "inference", "bodex", obj_pose_idx, "input_info.json"), 'r'))
result_file = os.path.join(shared_dir, "experiment", "squeeze_pose", input_info["obj_name"], input_info["file_name"], f"result_{input_info['squeeze_idx']}.json")
result = {}
while True:
    end = input("Success? (y/n): ")
    if end.lower() == 'y':
        result['success'] = True
        break
    elif end.lower() == 'n':
        result['success'] = False
        break
json.dump(result, open(result_file, 'w'))