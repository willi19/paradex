import numpy as np
import transforms3d as t3d


def load_robotwrist_pose(target_state):
    # target_traj = load_robot_target_traj(...)
    # target_state = target_traj[step]

    # target_state is the pid input for the robot
    translation = target_state[:3]
    rotation = target_state[3:6]

    # Normalize the vector to get the axis and angle
    angle = np.linalg.norm(rotation)  # Magnitude is the rotation angle
    if angle != 0:
        axis = rotation / angle  # Direction is the axis
    else:
        axis = np.array([1.0, 0.0, 0.0])  # Default axis if angle is 0

    rotmat = t3d.axangles.axangle2mat(axis, angle)

    target_pose = np.zeros((4, 4))
    target_pose[:3, 3] = translation
    target_pose[:3, :3] = rotmat
    return target_pose


def get_hand_tip_pos(hand_pos):
    # tip index: 4, 9, 14, 19, 24
    tip_idx = [4, 9, 14, 19, 0]
    tip_pos = hand_pos[:, tip_idx]
    return tip_pos
