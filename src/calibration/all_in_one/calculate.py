import os
import cv2
import argparse
import numpy as np
import json
import tqdm

from paradex.image.aruco import detect_aruco, triangulate_marker, draw_aruco
from paradex.image.undistort import undistort_img
from paradex.utils.file_io import find_latest_directory, home_path, download_dir, shared_dir, load_intrinsic, cam_param_dir, load_camparam
from paradex.geometry.math import rigid_transform_3D

parser = argparse.ArgumentParser()
parser.add_argument("--name", type=str, default=None, help="Name of the calibration directory.")

args = parser.parse_args()
if args.name is None:
    args.name = find_latest_directory(os.path.join(download_dir, "all_in_one"))

name = args.name
makrker_offset = np.load("data/marker_pos.npy", allow_pickle=True).item()
marker_id = [261, 263, 264, 265, 266]

root_path = os.path.join(shared_dir, "all_in_one", name)
index_list = os.listdir(root_path)
index_list.sort()

intrinsics = load_intrinsic()

cam_keypoint = {}

for idx in tqdm.tqdm(index_list[:50]):
    last_link_pose = np.load(os.path.join(root_path, idx, "robot.npy"))
    # intrinsic, extrinsic = load_camparam(os.path.join(root_path, idx))
    
    marker_robot_pose = {id :last_link_pose @ makrker_offset[id].T for id in marker_id}
      
    kypt_list = os.listdir(os.path.join(root_path, idx, "keypoint"))
    for kypt_name in kypt_list:
        serial_num = kypt_name.split(".")[0]
        kypt_dict = np.load(os.path.join(root_path, idx, "keypoint", kypt_name), allow_pickle=True).item()
        
        if serial_num not in cam_keypoint:
            cam_keypoint[serial_num] = {"2d":[], "3d":[]}
        
        for id, cor in kypt_dict.items():
            if id not in marker_id:
                continue
            cam_keypoint[serial_num]["2d"].append(cor[0])
            cam_keypoint[serial_num]["3d"].append(marker_robot_pose[id][:3].T)        
   
extrinsics = {}
     
for serial_num, cam_kypt in cam_keypoint.items():
    robot_cor = np.concatenate(cam_kypt["3d"])
    cam_cor = np.concatenate(cam_kypt["2d"])
    
    retvals, rvecs, tvecs, _ = cv2.solvePnPGeneric(robot_cor, cam_cor, intrinsics[serial_num]['intrinsics_undistort'], np.zeros(4))
    
    robot_cor_h = np.concatenate([robot_cor, np.ones((robot_cor.shape[0], 1))], axis=1)
    rvec, tvec = rvecs[0], tvecs[0]
    R, _ = cv2.Rodrigues(rvec)  # shape: (3, 3)

    # 2. Extrinsic matrix 만들기: 4x4
    extrinsic = np.eye(4)
    extrinsic[:3, :3] = R
    extrinsic[:3, 3] = tvec.flatten()
    
    robot_2_cam = (extrinsic @ robot_cor_h.T).T[:,:3]
    # robot_2_cam = robot_2_cam[:,:3] / robot_2_cam[:,3:]
    
    robot_2_cam = (intrinsics[serial_num]['intrinsics_undistort'] @ robot_2_cam.T).T
    robot_2_cam = robot_2_cam[:,:2] / robot_2_cam[:,2:]
    # tmp = np.array([[-0.0365371,   0.998842,    0.0313001, 0],
    #                 [ 0.80097246,  0.04799988, -0.59677393, 0],
    #                 [-0.59758527,  0.00326613, -0.80179871, 0],
    #                 [ 0,      0,          0,          1.        ]])
    
    # robot_2_cam = (extrinsic @ tmp @ robot_cor_h.T).T[:,:3]
    # # robot_2_cam = robot_2_cam[:,:3] / robot_2_cam[:,3:]
    
    # robot_2_cam = (intrinsics[serial_num]['intrinsics_undistort'] @ robot_2_cam.T).T
    # robot_2_cam = robot_2_cam[:,:2] / robot_2_cam[:,2:]
    
    extrinsics[serial_num] = extrinsic[:3,:].tolist()

new_intrinsics = {}
for serial_num, intrinsic in intrinsics.items():
    new_intrinsics[serial_num] = {}
    new_intrinsics[serial_num]['intrinsics_undistort'] = intrinsic['intrinsics_undistort'].tolist()
    new_intrinsics[serial_num]['original_intrinsics'] = np.array(intrinsic['original_intrinsics']).tolist()
    new_intrinsics[serial_num]['dist_params'] = np.array(intrinsic['dist_params']).tolist()
    new_intrinsics[serial_num]['width'] = intrinsic['width']
    new_intrinsics[serial_num]['height'] = intrinsic['height']
print(name)
os.makedirs(os.path.join(cam_param_dir, name), exist_ok=True)
with open(os.path.join(cam_param_dir, name, "intrinsics.json"), "w") as f:
    json.dump(new_intrinsics, f, indent=4)

with open(os.path.join(cam_param_dir, name, "extrinsics.json"), "w") as f:
        json.dump(extrinsics, f, indent=4)
      
#     robot_cor.append(np.load(os.path.join(he_calib_path, idx, "robot.npy")))
    
#     if os.path.exists(os.path.join(he_calib_path, idx, "marker_3d.npy")):
#         marker_3d = np.load(os.path.join(he_calib_path, idx, "marker_3d.npy"), allow_pickle=True).item()
#         cam_cor.append(marker_3d)
#         continue
        
#     img_dir = os.path.join(he_calib_path, idx, "image")
    
#     img_dict = {}
#     for img_name in os.listdir(img_dir):
#         img_dict[img_name.split(".")[0]] = cv2.imread(os.path.join(img_dir, img_name))
        
#     cor_3d = triangulate_marker(img_dict, intrinsic, extrinsic)

#     for serial_num, img in img_dict.items():
#         if serial_num not in cammtx:
#             continue
        
#         undist_img = undistort_img(img, intrinsic[serial_num])
#         undist_kypt, ids = detect_aruco(undist_img)
        
#         if ids is None:
#             continue
#         draw_aruco(undist_img, undist_kypt, ids, (0, 0, 255))
        
#         for mid in marker_id:
#             if mid not in ids or cor_3d[mid] is None:
#                 continue
#             pt_2d = project(cammtx[serial_num], cor_3d[mid])
#             draw_aruco(undist_img, [pt_2d], None, (255, 0, 0))
            
#         os.makedirs(os.path.join(he_calib_path, idx, "debug"), exist_ok=True)
#         cv2.imwrite(os.path.join(he_calib_path, idx, "debug", f"{serial_num}.png"), undist_img)

#     marker_3d = {}
#     for mid in marker_id:
#         if mid not in cor_3d or cor_3d[mid] is None:
#             continue
#         marker_3d[mid] = cor_3d[mid]
#     np.save(os.path.join(he_calib_path, idx, "marker_3d.npy"), marker_3d)
#     cam_cor.append(marker_3d)

# A_list = []
# B_list = []

# for i in range(len(index_list)-1):
#     B_list.append(robot_cor[i] @ np.linalg.inv(robot_cor[i+1]))
    
#     marker1 = []
#     marker2 = []
#     for mid in cam_cor[i]:
#         if mid in cam_cor[i+1]:
#             if mid in marker_id:
#                 marker1.append(cam_cor[i][mid])
#                 marker2.append(cam_cor[i+1][mid])
    
#     marker1 = np.vstack(marker1)
#     marker2 = np.vstack(marker2)
#     A_list.append(rigid_transform_3D(marker2, marker1))

# X = np.eye(4)
# theta, b_x = solve(A_list, B_list)
# X[0:3, 0:3] = theta
# X[0:3, -1] = b_x.flatten()
# for i in range(len(index_list)-1):
#     print(A_list[i] @ X - X @ B_list[i], "error")
    
# np.save(os.path.join(he_calib_path, "0", "C2R.npy"), X)
# print(extrinsic['22640993'])
# marker_pos = {}

# for idx in range(len(index_list)):
#     for mid in cam_cor[idx]:
#         if mid not in marker_pos:
#             marker_pos[mid] = []
#         marker_cam_pose = to_homo(cam_cor[idx][mid])
#         marker_pos[mid].append((np.linalg.inv(robot_cor[idx]) @ np.linalg.inv(X) @ marker_cam_pose.T).T)
        
# for mid in marker_pos:
#     print(np.std(np.array(marker_pos[mid]), axis=0), "marker offset error")
#     marker_pos[mid] = np.mean(marker_pos[mid], axis=0)
    
# np.save(os.path.join(he_calib_path, "0", "marker_pos.npy"), marker_pos)