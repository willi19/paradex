import numpy as np
import os
from paradex.utils.io import handeye_calib_path, find_latest_directory
import argparse
import numpy as np
from scipy.linalg import sqrtm
from numpy.linalg import inv
import numpy as np
from scipy.spatial.transform import Rotation as R
from paradex.utils.math import rigid_transform_3D
from dex_robot.utils.robot_wrapper import RobotWrapper
from dex_robot.utils.file_io import rsc_path
from dex_robot.utils.file_io import shared_path, load_camparam, load_c2r, download_path, rsc_path

def logR(T):
    R = T[0:3, 0:3]
    theta = np.arccos((np.trace(R) - 1)/2)
    logr = np.array([R[2,1] - R[1,2], R[0,2] - R[2,0], R[1,0] - R[0,1]]) * theta / (2*np.sin(theta))
    return logr

def Calibrate(A, B):
    n_data = len(A)
    M = np.zeros((3,3))
    C = np.zeros((3*n_data, 3))
    d = np.zeros((3*n_data, 1))
    A_ = np.array([])
    for i in range(n_data-1):
        alpha = logR(A[i])
        beta = logR(B[i])
        alpha2 = logR(A[i+1])
        beta2 = logR(B[i+1])
        alpha3 = np.cross(alpha, alpha2)
        beta3  = np.cross(beta, beta2)
        M1 = np.dot(beta.reshape(3,1),alpha.reshape(3,1).T)
        M2 = np.dot(beta2.reshape(3,1),alpha2.reshape(3,1).T)
        M3 = np.dot(beta3.reshape(3,1),alpha3.reshape(3,1).T)
        M = M1+M2+M3
    theta = np.dot(sqrtm(inv(np.dot(M.T, M))), M.T)
    for i in range(n_data):
        rot_a = A[i][0:3, 0:3]
        rot_b = B[i][0:3, 0:3]
        trans_a = A[i][0:3, 3]
        trans_b = B[i][0:3, 3]
        C[3*i:3*i+3, :] = np.eye(3) - rot_a
        d[3*i:3*i+3, 0] = trans_a - np.dot(theta, trans_b)
    b_x  = np.dot(inv(np.dot(C.T, C)), np.dot(C.T, d))
    return theta, b_x

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", type=str, default=None, help="Name of the calibration directory.")
    args = parser.parse_args()
    if args.name is None:
        args.name = find_latest_directory(handeye_calib_path)
    
    he_calib_path = os.path.join(handeye_calib_path, args.name)
    index_list = os.listdir(os.path.join(he_calib_path))

    A_list = []
    B_list = []

    robot = RobotWrapper(
        os.path.join(rsc_path, "xarm6", "xarm6_allegro_wrist_mounted_rotate.urdf")
    )
    finger_id_list = [11,13,14]
    finger_marker = {11:"ring_proximal", 13:"middle_proximal", 14:"index_proximal", 10:"thumb_proximal"}
    finger_index = {f"{finger_name}_proximal":robot.get_link_index(f"{finger_name}_proximal") for finger_name in ["thumb", "index", "middle", "ring"]}

    link_index = robot.get_link_index("link6")
    wrist_index = robot.get_link_index("palm_link")

    c2r = load_c2r(os.path.join(he_calib_path, "0"))


    for i in range(len(index_list)-1):
        idx1 = index_list[i]
        idx2 = index_list[i+1]

        robot_action1 = np.load(os.path.join(he_calib_path, idx1, "robot.npy"))
        robot_action2 = np.load(os.path.join(he_calib_path, idx2, "robot.npy"))

        marker_3d1 = np.load(os.path.join(he_calib_path, idx1, "marker_3d.npy"), allow_pickle=True).item()
        marker_3d2 = np.load(os.path.join(he_calib_path, idx2, "marker_3d.npy"), allow_pickle=True).item()

        for finger_id in finger_id_list:
            if finger_id not in marker_3d1 or finger_id not in marker_3d2:
                continue
            if marker_3d1[finger_id] is None or marker_3d2[finger_id] is None:
                continue
            A = rigid_transform_3D(marker_3d2[finger_id], marker_3d1[finger_id])
            

            robot.compute_forward_kinematics(robot_action1)
            T_r1 = robot.get_link_pose(link_index)
            T_h1 = robot.get_link_pose(finger_index[finger_marker[finger_id]])
            T_w1 = robot.get_link_pose(wrist_index)

            T_h1 = np.linalg.inv(T_w1) @ T_h1

            robot.compute_forward_kinematics(robot_action2)
            T_r2 = robot.get_link_pose(link_index)
            T_h2 = robot.get_link_pose(finger_index[finger_marker[finger_id]])
            T_w2 = robot.get_link_pose(wrist_index)

            T_h2 = np.linalg.inv(T_w2) @ T_h2

            A = np.linalg.inv(T_r1) @ np.linalg.inv(c2r) @ A @ c2r @ T_r2
            B = T_h1 @ np.linalg.inv(T_h2)

            A_list.append(A)
            B_list.append(B)
            # print(A@c2r - c2r@B)

                # err = A @ B - np.eye(4)
                # if np.max(np.abs(err)) < 0.05:
                #     print("Error: ", np.max(np.abs(err)))
                #     print(A)
                #     print(B)
                #     print(np.linalg.inv(A) @ B)


    X = np.eye(4)
    theta, b_x = Calibrate(A_list, B_list)
    X[0:3, 0:3] = theta
    X[0:3, -1] = b_x.flatten()
    print(X)

    link6 = robot.get_link_pose(link_index)
    wrist = robot.get_link_pose(wrist_index)
    print(np.linalg.inv(link6) @ wrist)

    print(A_list[0] @ X - X @ B_list[0])
    marker_pos = {}


    for idx in index_list:
        robot_action1 = np.load(os.path.join(he_calib_path, idx, "robot.npy"))

        marker_3d = np.load(os.path.join(he_calib_path, idx, "marker_3d.npy"), allow_pickle=True).item()()

        for finger_id in finger_id_list:
            if finger_id not in marker_3d:
                continue
            if marker_3d[finger_id] is None:
                continue

            T_h = robot.get_link_pose(finger_index[finger_marker[finger_id]])
            import pdb; pdb.set_trace()
            marker_pos[finger_id].append(np.linalg.inv(T_h) @ np.linalg.inv(c2r) @ marker_3d[finger_id].T)

    for mid in marker_pos:
        marker_pos[mid] = np.mean(marker_pos[mid], axis=0)
        
    np.save(os.path.join(he_calib_path, "0", "finger_pose.npy"), marker_pos)