import argparse
import os
import json
import yaml
from itertools import chain
from multiprocessing import Pool
from glob import glob
import cv2
from paradex.utils.io import find_latest_directory, home_dir, download_dir, load_cam_param_temp, cam_param_dir
import numpy as np
from paradex.triangulate.traingulate import ransac_triangulation
import tqdm

download_dir = os.path.join(download_dir,"calibration")
config_dir = "config"

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
        name = find_latest_directory(download_dir)
    else:
        name = args.name

    root_dir = os.path.join(download_dir, name)
    index_list = os.listdir(root_dir)
    index_list.sort()
    if len(index_list) == 0:
        print("No valid directories found.")
        exit()

    keypoint_path_list = []
    for index in index_list:
        frame_dir = os.path.join(root_dir, index, "keypoints")
        keypoint_path_list += [os.path.join(frame_dir, d) for d in os.listdir(frame_dir) if int(d) % 5 == 0]

    # Initial camera intrinsics
    width, height = 2048, 1536
    cx, cy = width // 2, height // 2
    num_cameras = 24

    intrinsics, extrinsics = load_cam_param_temp(name)
    # tot_kypt_dict = {}
    # tot_kypt_matches = {}
    length = []
    proj_err = {}

    for kypt_path in tqdm.tqdm(keypoint_path_list):
        kypt_file_list = os.listdir(os.path.join(root_dir, kypt_path))
        kypt_dict = {}

        for kypt_file in kypt_file_list:
            if "ids" in kypt_file:
                continue
            serial_num = kypt_file.split("_")[0]

            ids = np.load(os.path.join(root_dir, kypt_path, f"{serial_num}_ids.npy"))
            kypt = np.load(os.path.join(root_dir, kypt_path, kypt_file))
            int_mat = np.array(intrinsics[serial_num]['intrinsics_original'])
            ext_mat = np.array(extrinsics[serial_num])
            
            for i in range(ids.shape[0]):
                id = ids[i][0]
                cor = kypt[i]
                if id not in kypt_dict.keys():
                    kypt_dict[id] = {"2d": [], "projection":[]}
                
                kypt_dict[id]["2d"].append(cor)
                kypt_dict[id]["projection"].append(int_mat @ ext_mat)

        kypt_3d = {}
        for i in list(kypt_dict.keys()):
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

        for kypt_file in kypt_file_list:
            if "ids" in kypt_file:
                continue
            serial_num = kypt_file.split("_")[0]
            if serial_num not in proj_err.keys():
                proj_err[serial_num] = []

            ids = np.load(os.path.join(root_dir, kypt_path, f"{serial_num}_ids.npy"))
            kypt = np.load(os.path.join(root_dir, kypt_path, kypt_file))

            int_mat = np.array(intrinsics[serial_num]['intrinsics_original'])
            ext_mat = np.array(extrinsics[serial_num])
            
            for i in range(ids.shape[0]):
                id = ids[i][0]
                cor = kypt[i]
                if id not in kypt_3d.keys():
                    continue

                pt3d = kypt_3d[id]
                pt3d_h = np.hstack((pt3d, np.ones((1))))

                proj = int_mat @ ext_mat @ pt3d_h
                proj = proj[:2] / proj[2]

                err = np.linalg.norm(proj - cor)
                proj_err[serial_num].append(err)


    print(np.std(length))
    print(np.mean(length))

    new_extrinsics = {}
    for serial_num, extrinsic in extrinsics.items():
        new_extrinsic = np.array(extrinsic)
        new_extrinsic[:3, 3] *= (0.025 / np.mean(length))
        new_extrinsics[serial_num] = new_extrinsic.tolist()

    json.dump(new_extrinsics, open(os.path.join(cam_param_dir, name, "extrinsics.json"), "w"), indent=4)

    for serial_num, err_list in proj_err.items():
        print(serial_num, np.mean(err_list))