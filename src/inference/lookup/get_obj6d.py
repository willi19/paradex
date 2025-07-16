import numpy as np
from paradex.utils.file_io import find_latest_directory, handeye_calib_path, shared_dir, load_current_camparam, find_latest_index
import os
from paradex.geometry.math import rigid_transform_3D
from paradex.geometry.triangulate import ransac_triangulation
import cv2
from paradex.image.undistort import undistort_img
from paradex.image.aruco import detect_aruco

obj_name = "pringles"

dir_name = find_latest_directory(handeye_calib_path)
C2R = np.load(os.path.join(handeye_calib_path, dir_name, "0", "C2R.npy"))

camparam_dir = os.path.join(shared_dir, "cam_param")
camparam_name = find_latest_directory(camparam_dir)
camparam_path = os.path.join(shared_dir, "cam_param", camparam_name)

intrinsic, extrinsic = load_current_camparam(os.path.join(camparam_path))

cammat = {}
for serial_num in list(intrinsic.keys()):
    int_mat = intrinsic[serial_num]["intrinsics_undistort"]
    ext_mat = extrinsic[serial_num]
    cammat[serial_num] = int_mat @ ext_mat
    
# inf_id = find_latest_index(os.path.join(shared_dir, "inference", obj_name))

root_dir = os.path.join(shared_dir, "inference", obj_name, "0")
image_dir = os.path.join(root_dir, "image")
undist_img_dir = os.path.join(shared_dir, "marker_offset", obj_name, "0", "undistort_image")

id_cor = {}
for img_name in os.listdir(image_dir):
    img = cv2.imread(os.path.join(image_dir, img_name))
    serial_num = img_name.split(".")[0]
    undist_img = undistort_img(img, intrinsic[serial_num])
    cv2.imwrite(os.path.join(undist_img_dir, img_name), undist_img)

    undist_kypt, ids = detect_aruco(undist_img) # Tuple(np.ndarray(1, 4, 2)), np.ndarray(N, 1)
    paper_undist_kypt, paper_ids = detect_aruco(undist_img, dict_type="4X4_50")
    
    if ids is None and paper_ids is None:
        continue
    
    elif ids is None:
        undist_kypt = paper_undist_kypt
        ids = paper_ids
            
    elif paper_ids is not None:
        undist_kypt = np.concatenate([undist_kypt, paper_undist_kypt])
        ids = np.concatenate([ids, paper_ids])    
    
    ids = ids.reshape(-1)
    for id, k in zip(ids,undist_kypt):
        k = k.squeeze()
        
        if id not in id_cor:
            id_cor[id] = {"2d": [], "cammtx": []}
        id_cor[id]["2d"].append(k)
        id_cor[id]["cammtx"].append(cammat[serial_num]) 
        
cor_3d = {id:ransac_triangulation(np.array(id_cor[id]["2d"]), np.array(id_cor[id]["cammtx"])) for id in id_cor}

marker_offset = np.load(os.path.join(shared_dir, "marker_offset", obj_name, "0", "marker_offset.npy"), allow_pickle=True).item()
marker_id = list(marker_offset.keys())

A = np.concatenate([marker_offset[id] for id in marker_id])
B = np.concatenate([cor_3d[id] for id in marker_id])

obj_6D = np.linalg.inv(C2R) @ rigid_transform_3D(A, B)

np.save(os.path.join(root_dir, "obj_6D.npy"), obj_6D)
print(obj_6D)