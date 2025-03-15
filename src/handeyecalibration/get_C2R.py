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

    for i in range(len(index_list)-1):
        idx1 = index_list[i]
        idx2 = index_list[i+1]

        robot1 = np.load(os.path.join(he_calib_path, idx1, "link5.npy"))
        robot2 = np.load(os.path.join(he_calib_path, idx2, "link5.npy"))
        B_list.append(robot1 @ np.linalg.inv(robot2))

        marker_dict1 = np.load(os.path.join(he_calib_path, idx1, "marker_3d.npy"), allow_pickle=True).item()
        marker_dict2 = np.load(os.path.join(he_calib_path, idx2, "marker_3d.npy"), allow_pickle=True).item()

        marker1 = []
        marker2 = []
        for mid in marker_dict1:
            if mid in marker_dict2:
                marker1.append(marker_dict1[mid])
                marker2.append(marker_dict2[mid])
        
        marker1 = np.vstack(marker1)
        marker2 = np.vstack(marker2)

        A_list.append(rigid_transform_3D(marker2, marker1))


    X = np.eye(4)
    theta, b_x = Calibrate(A_list, B_list)
    X[0:3, 0:3] = theta
    X[0:3, -1] = b_x.flatten()
    print(A_list[0] @ X - X @ B_list[0])
    np.save(os.path.join(he_calib_path, "0", "C2R.npy"), X)