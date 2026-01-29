import os
import numpy as np
import json
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parents[1]))
from paradex.utils.path import shared_dir
from paradex.utils.file_io import find_latest_directory
from jisoo_test.gui_controller import RobotGUIController
from paradex.io.robot_controller import get_arm, get_hand
# convert and make pregrasp, greaps, squeeze by copy the last pose of computed_trajectory

# /home/temp_idCOMPUTED_TRAJ_PATH = "/home/robot/shared_data/jisoo_test/computed_trajectory.pickle"
import pickle
trajectory_path = "/home/temp_id/shared_data/jisoo_test/spray/20260128_154035/grasp_trajectory/traj/['spray_20260128_154035_']_6_9"
traj_dict = {}
for traj_type in  ['approach', 'grasp_pose', 'squeeze_pose']:
    traj_dict[traj_type] = np.load(os.path.join(trajectory_path, f"{traj_type}.npy"))


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

def rearrange_joint_pose(cur_pose, cur_joint_names, target_joint_names):
    '''
        cur_pose: DxN
    '''
    new_pose = np.zeros((cur_pose.shape[0], len(target_joint_names)))
    for dix, dname in enumerate(target_joint_names):
        if dname in cur_joint_names:
            org_idx = cur_joint_names.index(dname)
            new_pose[:,dix] = cur_pose[:,org_idx]
    return new_pose

approach = traj_dict['approach']
grasp_pose = traj_dict['grasp_pose']
squeeze_pose = traj_dict['squeeze_pose']

approach_hand = traj_dict['approach'][:,6:]
grasp_hand = traj_dict['grasp_pose'][:,6:]
squeeze_hand = traj_dict['squeeze_pose'][:,6:]

start = np.tile(approach_hand[0,:], (6,1))
pregrasp = np.tile(approach_hand[-1,:], (6,1))
grasp = np.tile(grasp_hand[-1,:], (6,1))
squeeze = np.tile(squeeze_hand[-1,:], (6,1))

base_pose = np.array([[ 0.07842263, -0.16270738,  0.98355285,  0.4116784 ],
 [ 0.19462326, -0.96510918, -0.17517437, -0.42133483],
 [ 0.97773804,  0.20515989, -0.04401976,  0.41725355],
 [ 0.,          0.,          0.,          1.        ]])

predefined_poses = {
    'base': base_pose
}
grasp_pose_dict = {
    'start': start,
    'pregrasp': pregrasp,
    'grasp': grasp,
    'squeezed': squeeze
}
arm = get_arm("xarm")
hand = get_hand("inspire", ip=True)
print(">>> Controllers created.")
rgc = RobotGUIController(
    robot_controller=arm,
    hand_controller=hand,
    predefined_poses=predefined_poses,
    grasp_pose=grasp_pose_dict,
    approach_traj=approach,
    lift_distance=100.0,
    place_distance=40.0
)
rgc.run()