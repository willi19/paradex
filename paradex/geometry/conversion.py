from scipy.spatial.transform import Rotation as R
import numpy as np
import transforms3d as t3d

def aa_to_rotmat(aa):
    # target_traj = load_robot_target_traj(...)
    # target_state = target_traj[step]

    # target_state is the pid input for the robot
    translation = aa[:3]
    rotation = aa[3:]

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