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
trajectory_path = "/home/temp_id/shared_data/jisoo_test/spray/20260130_054152/generated_trajectory_two_Steps/traj/['spray_20260130_054152_']_0"
traj_dict = {}
for traj_type in  ['approach', 'grasp_pose', 'squeeze_pose']:
    if os.path.exists(os.path.join(trajectory_path, f"{traj_type}.npy")):
        traj_dict[traj_type] = np.load(os.path.join(trajectory_path, f"{traj_type}.npy"))


def qpos_to_inspire_action(qpos):
    """Convert qpos (radians) to inspire integer action (0-1000).

    qpos order: [thumb_yaw, thumb_pitch, index, middle, ring, pinky]
    inspire action order: [pinky, ring, middle, index, thumb_pitch, thumb_yaw]

    Joint limits from URDF (inspire_hand_right.urdf):
        thumb_proximal_yaw_joint:   [0, 1.308]
        thumb_proximal_pitch_joint: [0, 0.6]
        index_proximal_joint:       [0, 1.47]
        middle_proximal_joint:      [0, 1.47]
        ring_proximal_joint:        [0, 1.47]
        pinky_proximal_joint:       [0, 1.47]
    """
    limits = np.array([1.308, 0.6, 1.47, 1.47, 1.47, 1.47])  # thumb_yaw, thumb_pitch, index, middle, ring, pinky

    if qpos.ndim == 1:
        q = qpos[:6]
        action = np.zeros(6, dtype=np.int32)
        normalized = np.clip(q / limits, 0.0, 1.0)
        # Map: qpos=0 -> action=1000 (open), qpos=limit -> action=0 (closed)
        action_float = (1.0 - normalized) * 1000.0
        # Reorder: qpos [thumb_yaw, thumb_pitch, index, middle, ring, pinky]
        #       -> action [pinky, ring, middle, index, thumb_pitch, thumb_yaw]
        action[0] = int(np.clip(action_float[5], 0, 1000))  # pinky
        action[1] = int(np.clip(action_float[4], 0, 1000))  # ring
        action[2] = int(np.clip(action_float[3], 0, 1000))  # middle
        action[3] = int(np.clip(action_float[2], 0, 1000))  # index
        action[4] = int(np.clip(action_float[1], 0, 1000))  # thumb_pitch
        action[5] = int(np.clip(action_float[0], 0, 1000))  # thumb_yaw
    else:
        q = qpos[:, :6]
        normalized = np.clip(q / limits, 0.0, 1.0)
        action_float = (1.0 - normalized) * 1000.0
        action = np.zeros((qpos.shape[0], 6), dtype=np.int32)
        action[:, 0] = np.clip(action_float[:, 5], 0, 1000).astype(int)  # pinky
        action[:, 1] = np.clip(action_float[:, 4], 0, 1000).astype(int)  # ring
        action[:, 2] = np.clip(action_float[:, 3], 0, 1000).astype(int)  # middle
        action[:, 3] = np.clip(action_float[:, 2], 0, 1000).astype(int)  # index
        action[:, 4] = np.clip(action_float[:, 1], 0, 1000).astype(int)  # thumb_pitch
        action[:, 5] = np.clip(action_float[:, 0], 0, 1000).astype(int)  # thumb_yaw
    return action


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

approach_hand = traj_dict['approach'][:,6:]
grasp_hand = traj_dict['grasp_pose'][:,6:]

if 'squeeze_pose' in traj_dict:
    squeeze_hand = traj_dict['squeeze_pose'][:,6:]
    squeeze_pose = traj_dict['squeeze_pose']
else:
    squeeze_hand = grasp_hand
    squeeze_pose = grasp_pose
    
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
approach[:,6:] = qpos_to_inspire_action(approach[:,6:])
for grasp_type in grasp_pose_dict:
    grasp_pose_dict[grasp_type] = qpos_to_inspire_action(grasp_pose_dict[grasp_type])[0]
print(">>> Grasp poses converted.")


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