import os
import cv2
import argparse
import numpy as np

from paradex.image.aruco import detect_aruco, triangulate_marker, draw_aruco
from paradex.image.undistort import undistort_img
from paradex.geometry.Tsai_Lenz import solve
from paradex.geometry.conversion import project, to_homo
from paradex.utils.file_io import handeye_calib_path, find_latest_directory, load_camparam
from paradex.image.projection import get_cammtx
from paradex.geometry.math import rigid_transform_3D

marker_id = [261, 262, 263, 264, 265, 266]

parser = argparse.ArgumentParser()
parser.add_argument("--name", type=str, default=None, help="Name of the calibration directory.")

args = parser.parse_args()
if args.name is None:
    args.name = find_latest_directory(handeye_calib_path)

name = args.name
he_calib_path = os.path.join(handeye_calib_path, name)

intrinsic, extrinsic = load_camparam(os.path.join(he_calib_path, "0"))
cammtx = get_cammtx(intrinsic, extrinsic)

index_list = os.listdir(he_calib_path)

robot_cor = []
cam_cor = []

for idx in index_list:
    robot_cor.append(np.load(os.path.join(he_calib_path, idx, "robot.npy")))
    
    # if os.path.exists(os.path.join(he_calib_path, idx, "marker_3d.npy")):
    #     marker_3d = np.load(os.path.join(he_calib_path, idx, "marker_3d.npy"), allow_pickle=True).item()
    #     cam_cor.append(marker_3d)
    #     continue
        
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
    np.save(os.path.join(he_calib_path, idx, "marker_3d.npy"), marker_3d)
    cam_cor.append(marker_3d)

A_list = []
B_list = []

for i in range(len(index_list)-1):
    B_list.append(robot_cor[i] @ np.linalg.inv(robot_cor[i+1]))
    
    marker1 = []
    marker2 = []
    for mid in cam_cor[i]:
        if mid in cam_cor[i+1]:
            if mid in marker_id:
                marker1.append(cam_cor[i][mid])
                marker2.append(cam_cor[i+1][mid])
    
    marker1 = np.vstack(marker1)
    marker2 = np.vstack(marker2)
    A_list.append(rigid_transform_3D(marker2, marker1))

X = np.eye(4)
theta, b_x = solve(A_list, B_list)
X[0:3, 0:3] = theta
X[0:3, -1] = b_x.flatten()
# for i in range(len(index_list)-1):
#     print(A_list[i] @ X - X @ B_list[i], "error")
print(X)
np.save(os.path.join(he_calib_path, "0", "C2R.npy"), X)
marker_pos = {}

for idx in range(len(index_list)):
    for mid in cam_cor[idx]:
        if mid not in marker_pos:
            marker_pos[mid] = []
        marker_cam_pose = to_homo(cam_cor[idx][mid])
        marker_pos[mid].append((np.linalg.inv(robot_cor[idx]) @ np.linalg.inv(X) @ marker_cam_pose.T).T)
        
for mid in marker_pos:
    print(np.std(np.array(marker_pos[mid]), axis=0), "marker offset error")
    marker_pos[mid] = np.mean(marker_pos[mid], axis=0)
    
np.save(os.path.join(he_calib_path, "0", "marker_pos.npy"), marker_pos)