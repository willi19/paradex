import numpy as np
import os
from paradex.utils.file_io import handeye_calib_path, find_latest_directory
import argparse
import numpy as np
from numpy.linalg import inv
import numpy as np
from scipy.spatial.transform import Rotation as R
from paradex.geometry.math import rigid_transform_3D
from paradex.geometry.Tsai_Lenz import solve

marker_id_list = [261,262,263,264,265,266]


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

        robot1 = np.load(os.path.join(he_calib_path, idx1, "link6.npy"))
        robot2 = np.load(os.path.join(he_calib_path, idx2, "link6.npy"))
        B_list.append(robot1 @ np.linalg.inv(robot2))

        marker_dict1 = np.load(os.path.join(he_calib_path, idx1, "marker_3d.npy"), allow_pickle=True).item()
        marker_dict2 = np.load(os.path.join(he_calib_path, idx2, "marker_3d.npy"), allow_pickle=True).item()

        marker1 = []
        marker2 = []
        for mid in marker_dict1:
            if mid in marker_dict2:
                if mid in marker_id_list:
                    marker1.append(marker_dict1[mid])
                    marker2.append(marker_dict2[mid])
        
        marker1 = np.vstack(marker1)
        marker2 = np.vstack(marker2)

        A_list.append(rigid_transform_3D(marker2, marker1))


    X = np.eye(4)
    theta, b_x = solve(A_list, B_list)
    X[0:3, 0:3] = theta
    X[0:3, -1] = b_x.flatten()
    for i in range(len(index_list)-1):
        print(A_list[i] @ X - X @ B_list[i])
    np.save(os.path.join(he_calib_path, "0", "C2R.npy"), X)

    marker_pos = {}
    marker_id_list = [261,263,264,265,266]

    for idx in index_list:
        robot = np.load(os.path.join(he_calib_path, idx, "link6.npy"))
        marker_dict = np.load(os.path.join(he_calib_path, idx, "marker_3d.npy"), allow_pickle=True).item()

        for mid in marker_dict:
            if mid not in marker_pos:
                marker_pos[mid] = []
            # marker_dict[mid] :4x3
            marker_pos[mid].append(np.linalg.inv(robot) @ np.linalg.inv(X) @ np.hstack((marker_dict[mid], np.ones((marker_dict[mid].shape[0], 1)))).T)
            
    # for mid in marker_pos:
    #     # import pdb; pdb.set_trace()
    #     print(np.std(marker_pos[mid], axis=0))
    #     marker_pos[mid] = np.mean(marker_pos[mid], axis=0)
        
    np.save(os.path.join(he_calib_path, "0", "marker_pos.npy"), marker_pos)