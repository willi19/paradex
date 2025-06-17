import argparse
import os
import json
import yaml
from itertools import chain
from multiprocessing import Pool
from glob import glob
import cv2
from paradex.utils.file_io import find_latest_directory, home_path, download_dir, load_colmap_camparam, shared_dir
import numpy as np
from paradex.geometry.triangulate import ransac_triangulation
import tqdm

extrinsic_dir = os.path.join(shared_dir,"extrinsic")
cam_param_dir = os.path.join(shared_dir, "cam_param")
dir = "config"

if __name__ == "__main__":
    import time
    start_time = time.time()
    parser = argparse.ArgumentParser(description="Manage timestamped directories.")
    parser.add_argument("--name", type=str, help="Name of the directory to detect keypoint.")
    parser.add_argument("--latest", action="store_true", help="Split the latest video files.")
    
    args = parser.parse_args()
    
    if not args.latest and not args.name:
        print("Please specify either --latest or --name.")
        exit()
    
    if args.latest:
        name = find_latest_directory(extrinsic_dir)
    else:
        name = args.name

    root_dir = os.path.join(extrinsic_dir, name)
    index_list = os.listdir(root_dir)

    index_list.sort()
    if len(index_list) == 0:
        print("No valid directories found.")
        exit()

    intrinsics, extrinsics = load_colmap_camparam(root_dir)

    length = []
    proj_err = {}

    offset = 0

    for index in index_list:
        file_list = os.listdir(os.path.join(root_dir, index))
        max_id = -1
        kypt_dict = {}

        for kypt_file in file_list:
            if "cor" not in kypt_file:
                continue
            serial_num = kypt_file.split("_")[0]

            ids = np.load(os.path.join(root_dir, index, f"{serial_num}_id.npy"))
            if len(ids) == 0:
                continue
            max_id = max(max_id, np.max(ids))

            # ids += offset
            kypt = np.load(os.path.join(root_dir, index, kypt_file))

            if serial_num not in intrinsics.keys():
                print(serial_num)
                continue
            
            int_mat = np.array(intrinsics[serial_num]['original_intrinsics'])
            ext_mat = np.array(extrinsics[serial_num])
            int_dist = np.array(intrinsics[serial_num]['dist_params'])

            int_undist = np.array(intrinsics[serial_num]['intrinsics_undistort'])

            if len(kypt) == 0:
                continue
            normalized = cv2.undistortPoints(kypt, int_mat, int_dist)
            kypt = normalized.squeeze() * np.array(
                [[int_undist[0, 0], int_undist[1, 1]]]
            ) + np.array(
                [[int_undist[0, 2], int_undist[1, 2]]]
            )

            for i in range(ids.shape[0]):
                id = ids[i][0]
                cor = kypt[i]
                if id not in kypt_dict.keys():
                    kypt_dict[id] = {"2d": [], "projection":[]}
                
                kypt_dict[id]["2d"].append(cor)
                kypt_dict[id]["projection"].append(int_undist @ ext_mat)

        kypt_3d = {}
        for i in kypt_dict.keys():
            proj_mat = np.array(kypt_dict[i]["projection"])
            kypt_2d = np.array(kypt_dict[i]["2d"])
            pt3d = ransac_triangulation(kypt_2d, proj_mat)
            if pt3d is None:
                continue
            kypt_3d[i] = pt3d

        idx_list = list(kypt_3d.keys())
        idx_list.sort()

        for i in idx_list:
            if i-1 in list(kypt_3d.keys()) and (i-idx_list[0]) % 10 != 0:
                length.append(np.linalg.norm(kypt_3d[i] - kypt_3d[i-1]))
            if i+10 in list(kypt_3d.keys()):
                length.append(np.linalg.norm(kypt_3d[i] - kypt_3d[i+10]))
            


        
        for kypt_file in file_list:
            if "cor" not in kypt_file:
                continue

            kypt = np.load(os.path.join(root_dir, index, kypt_file))
            ids = np.load(os.path.join(root_dir, index, f"{kypt_file.split('_')[0]}_id.npy"))
            serial_num = kypt_file.split("_")[0]

            int_mat = np.array(intrinsics[serial_num]['original_intrinsics'])
            ext_mat = np.array(extrinsics[serial_num])
            int_dist = np.array(intrinsics[serial_num]['dist_params'])
            int_undist = np.array(intrinsics[serial_num]['intrinsics_undistort'])

            if len(kypt) == 0:
                continue

            normalized = cv2.undistortPoints(kypt, int_mat, int_dist)
            kypt = normalized.squeeze() * np.array(
                [[int_undist[0, 0], int_undist[1, 1]]]
            ) + np.array(
                [[int_undist[0, 2], int_undist[1, 2]]]
            )

            if serial_num not in proj_err.keys():
                proj_err[serial_num] = []

            for i in range(ids.shape[0]):
                id = ids[i][0]
                cor = kypt[i]
                if id not in kypt_3d.keys():
                    continue

                pt3d = kypt_3d[id]
                pt3d_h = np.hstack((pt3d, np.ones((1))))

                proj = int_undist @ ext_mat @ pt3d_h
                proj = proj[:2] / proj[2]

                err = np.linalg.norm(proj - cor)
                proj_err[serial_num].append(err)
                    
    print(np.std(length))
    print(np.mean(length))

    for serial_num, proj in proj_err.items():
        print(serial_num, np.mean(proj))

    new_extrinsics = {}
    for serial_num, extrinsic in extrinsics.items():
        new_extrinsic = np.array(extrinsic)
        new_extrinsic[:3, 3] *= (0.025 / np.mean(length))
        new_extrinsics[serial_num] = new_extrinsic.tolist()

    os.makedirs(os.path.join(cam_param_dir, name), exist_ok=True)
        
    with open(os.path.join(cam_param_dir, name, "extrinsics.json"), "w") as f:
        json.dump(new_extrinsics, f, indent=4)

    for serial_num, intrinsic in intrinsics.items():
        new_intrinsic = np.array(intrinsic['intrinsics_undistort'])
        intrinsics[serial_num]['intrinsics_undistort'] = new_intrinsic.tolist()
        intrinsics[serial_num]['original_intrinsics'] = np.array(intrinsic['original_intrinsics']).tolist()
        intrinsics[serial_num]['dist_params'] = np.array(intrinsic['dist_params']).tolist()

    with open(os.path.join(cam_param_dir, name, "intrinsics.json"), "w") as f:
        json.dump(intrinsics, f, indent=4)