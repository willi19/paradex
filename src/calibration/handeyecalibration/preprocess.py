import os
import cv2
import argparse
import numpy as np

from paradex.image.aruco import detect_aruco, triangulate_marker, draw_aruco
from paradex.image.undistort import undistort_img
from paradex.geometry.Tsai_Lenz import solve
from paradex.geometry.conversion import project
from paradex.utils.file_io import handeye_calib_path, find_latest_directory, load_cam_param, rsc_path, handeye_calib_path
from paradex.utils.cam_param import get_cammtx

marker_id = [261, 262, 263, 264, 265, 266]

parser = argparse.ArgumentParser()
parser.add_argument("--name", type=str, default=None, help="Name of the calibration directory.")

args = parser.parse_args()
if args.name is None:
    args.name = find_latest_directory(handeye_calib_path)

name = args.name
he_calib_path = os.path.join(handeye_calib_path, name)

intrinsic, extrinsic = load_cam_param(os.path.join(he_calib_path, "0", "cam_param"))
cammtx = get_cammtx(intrinsic, extrinsic)

index_list = os.listdir(he_calib_path)

robot_cor = []
cam_cor = []

for idx in index_list:
    img_dir = os.path.join(he_calib_path, idx, "image")
    
    img_dict = {}
    for img_name in os.listdir(img_dir):
        img_dict[img_name.split(".")[0]] = cv2.imread(os.path.join(img_dir, img_name))
        
    cor_3d = triangulate_marker(img_dict, intrinsic, extrinsic)

    for serial_num, img in img_dict.items():
        if serial_num not in cammtx:
            continue
        
        undist_img = undistort_img(img, intrinsic[serial_num])
        undist_kypt, ids = detect_aruco(undist_img)
        
        if ids is None:
            continue
        draw_aruco(undist_img, undist_kypt, ids, (0, 0, 255))
        
        for mid in marker_id:
            if mid not in ids or cor_3d[mid] is None:
                continue
            pt_2d = project(cammtx[serial_num], cor_3d[mid])
            draw_aruco(undist_img, [pt_2d], None, (255, 0, 0))
            
        os.makedirs(os.path.join(he_calib_path, idx, "debug"), exist_ok=True)
        cv2.imwrite(os.path.join(he_calib_path, idx, "debug", f"{serial_num}.png"), undist_img)

    marker_3d = {}
    for mid in marker_id:
        if mid not in cor_3d or cor_3d[mid] is None:
            continue
        marker_3d[mid] = cor_3d[mid]
    
    cam_cor.append(marker_3d)
    robot_cor.append(np.load(os.path.join(he_calib_path, idx, "robot.npy")))
