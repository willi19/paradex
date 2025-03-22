import argparse
import os
import json
import yaml
from itertools import chain
from multiprocessing import Pool
from glob import glob
import cv2
from paradex.utils.io import find_latest_directory, home_dir, download_dir, load_cam_param

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

    extrinsics, intrinsics = load_cam_param(name)
    # tot_kypt_dict = {}
    # tot_kypt_matches = {}

    for kypt_path in keypoint_path_list:
        kypt_file_list = os.listdir(os.path.join(root_dir, kypt_path))
        kypt_dict = {}

        for kypt_file in kypt_file_list:
            if "ids" in kypt_file:
                continue
            serial_num = kypt_file.split("_")[0]

            ids = np.load(os.path.join(root_dir, kypt_path, f"{serial_num}_ids.npy"))
            kypt = np.load(os.path.join(root_dir, kypt_path, kypt_file))

            for i in range(ids.shape[0]):
                id = ids[i]
                cor = kypt[i]
                print(cor)
                if id not in kypt_dict.keys():
                    kypt_dict[id] = {"2d": [], "projection":[]}
                kypt_dict[id]["2d"].append(cor)
                kypt_dict[id]["projection"].append(np.dot(intrinsics[serial_num] @ extrinsics[serial_num]))

            kypt_3d = "asdf"#
        
        for serial_num, kypt_list in kypt_dict.items():
            kypt_list = np.vstack(kypt_list)
            kypt_dict[serial_num] = kypt_list
