import os
import cv2
import numpy as np

from paradex.geometry.math import rigid_transform_3D
from paradex.image.aruco import detect_aruco
from paradex.utils.file_io import load_camparam, load_c2r, shared_dir
from paradex.image.aruco import triangulate_marker

grasp_policy_path = os.path.join("data", "ramen")
marker_offset = {}

for idx in os.listdir(grasp_policy_path):
    intrinsic, extrinsic = load_camparam(os.path.join(grasp_policy_path, idx))    
    c2r = load_c2r(os.path.join(grasp_policy_path, idx))

    img_dict = {}
    
    for img_name in os.listdir(os.path.join(grasp_policy_path, idx, "grasp", "start", "image")):
        serial_num = img_name.split(".")[0]
        img_dict[serial_num] = cv2.imread(os.path.join(grasp_policy_path, idx, "grasp", "start", "image", img_name))
    
    cor_3d = triangulate_marker(img_dict, intrinsic, extrinsic)
    marker_offset = np.load(os.path.join(grasp_policy_path, idx, "marker_offset.npy"), allow_pickle=True).item()
    marker_id = list(marker_offset.keys())

    A = []
    B = []
    for mid in marker_id:
        if mid not in cor_3d or cor_3d[mid] is None:
            continue
        
        A.append(marker_offset[mid])
        B.append(cor_3d[mid])
    
    A = np.concatenate(A)
    B = np.concatenate(B)
    obj_T = rigid_transform_3D(A, B)
    obj_T = np.linalg.inv(c2r) @ obj_T
    obj_T[:3, :3] = np.eye(3)  # Special normalization for ramen

    wrist_se3 = np.load(os.path.join(grasp_policy_path, idx, "grasp", "wristSe3.npy"))

    np.save(os.path.join(grasp_policy_path, idx, "grasp", "obj_wristSe3.npy"), np.linalg.inv(obj_T) @ wrist_se3)