import os
import cv2
import argparse
import numpy as np
import json
import tqdm

from paradex.image.aruco import detect_aruco, triangulate_marker, draw_aruco
from paradex.image.undistort import undistort_img
from paradex.utils.file_io import find_latest_directory, home_path, download_dir, shared_dir, load_intrinsic, cam_param_dir



parser = argparse.ArgumentParser()
parser.add_argument("--name", type=str, default=None, help="Name of the calibration directory.")

args = parser.parse_args()
if args.name is None:
    args.name = find_latest_directory(os.path.join(download_dir, "all_in_one"))

name = args.name
makrker_offset = np.load("data/marker_offset.npy", allow_pickle=True).item()
marker_id = [261, 263, 264, 265, 266]

root_path = os.path.join(download_dir, "all_in_one", name)
index_list = os.listdir(root_path)
index_list.sort()

intrinsics = load_intrinsic()

cam_keypoint = {}

for idx in tqdm.tqdm(index_list):
    last_link_pose = np.load(os.path.join(root_path, idx, "robot.npy"))
    marker_robot_pose = {id :last_link_pose @ makrker_offset[id].T for id in marker_id}
      
    img_list = os.listdir(os.path.join(root_path, idx, "image"))
    for img_name in img_list:
        img = cv2.imread(os.path.join(root_path, idx, "image", img_name))
        serial_num = img_name.split(".")[0]
        undist_img = undistort_img(img, intrinsics[serial_num])
        kypts, ids = detect_aruco(undist_img)
        
        if ids is None:
            continue
        
        kypt_dict = {id[0]:kypt for id, kypt in zip(ids, kypts)}
        os.makedirs(os.path.join(shared_dir, "all_in_one", name, str(idx), "keypoint"), exist_ok=True)
        np.save(os.path.join(shared_dir, "all_in_one", name, str(idx), "keypoint", f"{serial_num}.npy"), kypt_dict)
        
        # kypts = kypts[:,0]
        ids = ids[:,0]
        
        if serial_num not in cam_keypoint:
            cam_keypoint[serial_num] = {"2d":[], "3d":[]}
        
        for id, cor in zip(ids, kypts):
            if id not in marker_id:
                continue
            cam_keypoint[serial_num]["2d"].append(cor[0])
            cam_keypoint[serial_num]["3d"].append(marker_robot_pose[id][:3].T)        
   
extrinsics = {}
     
for serial_num, cam_kypt in cam_keypoint.items():
    robot_cor = np.concatenate(cam_kypt["3d"])
    cam_cor = np.concatenate(cam_kypt["2d"])
    retval, rvec, tvec, _ = cv2.solvePnPRansac(robot_cor, cam_cor, intrinsics[serial_num]['intrinsics_undistort'], np.zeros(4))
    
    rvec = np.array([[0.1], [0.2], [0.3]])
    tvec = np.array([[10], [20], [30]])

    # 1. rvec을 3x3 회전 행렬로 변환
    R, _ = cv2.Rodrigues(rvec)  # shape: (3, 3)

    # 2. Extrinsic matrix 만들기: 4x4
    extrinsic = np.eye(4)
    extrinsic[:3, :3] = R
    extrinsic[:3, 3] = tvec.flatten()
    
    extrinsics[serial_num] = extrinsic[:3,:].tolist()

new_intrinsics = {}
for serial_num, intrinsic in intrinsics.items():
    new_intrinsics[serial_num] = {}
    new_intrinsics[serial_num]['intrinsics_undistort'] = intrinsic['intrinsics_undistort'].tolist()
    new_intrinsics[serial_num]['original_intrinsics'] = np.array(intrinsic['original_intrinsics']).tolist()
    new_intrinsics[serial_num]['dist_params'] = np.array(intrinsic['dist_params']).tolist()
    new_intrinsics[serial_num]['width'] = intrinsic['width']
    new_intrinsics[serial_num]['height'] = intrinsic['height']
        
os.makedirs(os.path.join(cam_param_dir, name), exist_ok=True)
with open(os.path.join(cam_param_dir, name, "intrinsics.json"), "w") as f:
    json.dump(new_intrinsics, f, indent=4)

with open(os.path.join(cam_param_dir, name, "extrinsics.json"), "w") as f:
        json.dump(extrinsics, f, indent=4)
    