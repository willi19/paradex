import argparse
import os
import json
import yaml
from itertools import chain
from multiprocessing import Pool
from glob import glob
import pycolmap
import cv2
from cv2 import aruco
import multiprocessing as mp
from paradex.calibration.database import *
from paradex.calibration.colmap import get_two_view_geometries
from paradex.utils.io import find_latest_directory, home_dir, download_dir, shared_dir
import tqdm

download_dir = os.path.join(download_dir,"calibration")
config_dir = "config"

chessboard_size = (10, 7)  # number of inner corners
square_size = 0.025  # in meters or whatever unit you choose

objp = np.zeros((np.prod(chessboard_size), 3), np.float32)
objp[:, :2] = np.indices(chessboard_size).T.reshape(-1, 2)
objp *= square_size

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

    num_cam = 24
    keypoint_path_list = []
    for index in index_list:
        frame_dir = os.path.join(root_dir, index, "keypoints")    
        keypoint_path_list += [os.path.join(frame_dir, d) for d in os.listdir(frame_dir) if int(d) % 50 == 1]
    
    with open(f"{config_dir}/camera_index.json", "r") as f:
        camera_index = json.load(f)  # {camera_serial : camera_index}

    # with open(f"{config_dir}/lens_info.json", "r") as f:
    #     lens_info = json.load(f)

    camera_index_inv = dict()
    for k,v in camera_index.items():
        camera_index_inv[v] = k

    # Initial camera intrinsics
    width, height = 2048, 1536
    cx, cy = width // 2, height // 2
    num_cameras = 24

    tot_kypt_dict = {}
    tot_id_dict = {}
    tot_obj_dict = {}

    tot_kypt_matches = {}

    for kypt_path in tqdm.tqdm(keypoint_path_list):
        kypt_file_list = os.listdir(os.path.join(root_dir, kypt_path))
        kypt_dict = {}
        for kypt_file in kypt_file_list:
            if "ids" in kypt_file:
                continue
            serial_num = kypt_file.split("_")[0]

            ids = np.load(os.path.join(root_dir, kypt_path, f"{serial_num}_ids.npy")) 
            if ids.shape[0] < 50:
                continue
            
            id_min = (np.min(ids) // 70) * 70
            ids = ids[:,0] - id_min

            corner = np.load(os.path.join(root_dir, kypt_path, kypt_file))
            obj = objp[ids]


            if serial_num not in tot_kypt_dict:
                tot_kypt_dict[serial_num] = []
                tot_id_dict[serial_num] = []
                tot_obj_dict[serial_num] = []
            
            # tot_id_dict[serial_num].append(ids
            tot_kypt_dict[serial_num].append(corner)
            tot_obj_dict[serial_num].append(obj)

    intrinsics = {}
    for serial_num in tqdm.tqdm(list(tot_kypt_dict.keys())):
        # tot_kypt_dict[serial_num] = np.vstack(tot_kypt_dict[serial_num])
        # tot_obj_dict[serial_num] = np.vstack(tot_obj_dict[serial_num])
        # tot_id_dict[serial_num] = np.hstack(tot_id_dict[serial_num])
        ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
            objectPoints=tot_obj_dict[serial_num],
            imagePoints=tot_kypt_dict[serial_num],
            imageSize=(width, height),
            cameraMatrix=None,
            distCoeffs=None
        )
        print(ret, serial_num)

        intrinsics[serial_num] = {}
        intrinsics[serial_num]["original_intrinsics"] = camera_matrix.tolist()
        intrinsics[serial_num]["dist_param"] = dist_coeffs.tolist()
        intrinsics[serial_num]["height"] = height
        intrinsics[serial_num]["width"] = width

        new_cammtx, roi = cv2.getOptimalNewCameraMatrix(camera_matrix, dist_coeffs, (width, height), 1, (width, height))
        intrinsics[serial_num]["Intrinsics"] = new_cammtx.reshape(-1).tolist()
    
    os.makedirs(f"{shared_dir}/cam_param/{name}", exist_ok=True)
    json.dump(intrinsics, open(f"{shared_dir}/cam_param/{name}/intrinsics_init.json", "w"), indent='\t')