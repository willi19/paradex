import os
import numpy as np
import json
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parents[2]))
from paradex.utils.path import shared_dir
from paradex.utils.file_io import find_latest_directory
from paradex.io.robot_controller.gui_controller_test import RobotGUIController
from paradex.io.robot_controller import get_arm, get_hand
# convert and make pregrasp, greaps, squeeze by copy the last pose of computed_trajectory

# /home/temp_id/shared_data/jisoo_test/spray/20260128_154035/grasp_trajectory/traj/['spray_20260128_154035_']_6_9

isaac_joint_names =  ['joint1', 'joint2', 'joint3', 'joint4', 'joint5', 'joint6', 'index_base', 'middle_base', 'ring_base', 'thumb_base', 'index_proximal', 'middle_proximal', 'ring_proximal', 'thumb_proximal', 'index_medial', 'middle_medial', 'ring_medial', 'thumb_medial', 'index_distal', 'middle_distal', 'ring_distal', 'thumb_distal']
curobo_joint_names =  ['joint1', 'joint2', 'joint3', 'joint4', 'joint5', 'joint6',  'thumb_base', 'thumb_proximal', 'thumb_medial', 'thumb_distal', 'index_base', 'index_proximal', 'index_medial', 'index_distal', 'middle_base', 'middle_proximal', 'middle_medial', 'middle_distal', 'ring_base', 'ring_proximal', 'ring_medial', 'ring_distal']

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

COMPUTED_TRAJ_PATH = "/home/robot/shared_data/jisoo_test/computed_trajectory.pickle"
import pickle
computed_trajectory = pickle.load(open(COMPUTED_TRAJ_PATH, 'rb'))
idx=1
computed_trajectory = computed_trajectory[idx]
traj = computed_trajectory['traj']
traj = rearrange_joint_pose(traj, isaac_joint_names, curobo_joint_names)
# traj[:, 6:] = convert(traj[:, 6:])
hand_traj = traj[:, 6:]

start = np.tile(hand_traj[0,:], (6,1))
pregrasp = np.tile(hand_traj[-1,:], (6,1))
grasp = np.tile(hand_traj[-1,:], (6,1))
squeeze = np.tile(hand_traj[-1,:], (6,1))

base_pose = np.array([[ 0.07842263, -0.16270738,  0.98355285,  0.4116784 ],
 [ 0.19462326, -0.96510918, -0.17517437, -0.42133483],
 [ 0.97773804,  0.20515989, -0.04401976,  0.41725355],
 [ 0.,          0.,          0.,          1.        ]])

# pregrasp_pose = np.load(os.path.join(shared_dir, "inference", "bodex", obj_pose_idx, "pregrasp.npy"))
# grasp_pose = np.load(os.path.join(shared_dir, "inference", "bodex", obj_pose_idx, "grasp_pose.npy"))
# squeeze_pose = np.load(os.path.join(shared_dir, "inference", "bodex", obj_pose_idx, "squeeze_pose.npy"))
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
hand = get_hand("allegro")
print(">>> Controllers created.")
rgc = RobotGUIController(
    robot_controller=arm,
    hand_controller=hand,
    predefined_poses=predefined_poses,
    grasp_pose=grasp_pose_dict,
    approach_traj=traj,
    lift_distance=100.0,
    place_distance=40.0
)
rgc.run()