from paradex.image.aruco import detect_aruco
from paradex.image.undistort import undistort_img

from paradex.utils.file_io import shared_dir, find_latest_directory, load_current_camparam
import os
import cv2
from paradex.geometry.triangulate import ransac_triangulation
import numpy as np

obj_name = "pringles"

camparam_dir = os.path.join(shared_dir, "cam_param")
camparam_name = find_latest_directory(camparam_dir)
camparam_path = os.path.join(shared_dir, "cam_param", camparam_name)

intrinsic, extrinsic = load_current_camparam(os.path.join(camparam_path))
cammat = {}
for serial_num in list(intrinsic.keys()):
    int_mat = intrinsic[serial_num]["intrinsics_undistort"]
    ext_mat = extrinsic[serial_num]
    cammat[serial_num] = int_mat @ ext_mat
    
root_dir = os.path.join(shared_dir, "marker_offset", obj_name, "0")
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
marker_id = [176, 177, 178, 1, 2, 3, 4] #, 11, 13, 14]

debug = False
if debug == True:
    for img_name in os.listdir(image_dir):
        img = cv2.imread(os.path.join(image_dir, img_name))
        serial_num = img_name.split(".")[0]
        undist_img = undistort_img(img, intrinsic[serial_num])
        undist_kypt, ids = detect_aruco(undist_img)
        
        paper_undist_kypt, paper_ids = detect_aruco(undist_img, dict_type="4X4_50")
        
        undist_kypt = np.concatenate([undist_kypt, paper_undist_kypt])
        ids = np.concatenate([ids, paper_ids])
        
        if ids is None:
            continue
        img_tmp = undist_img.copy()
        # if 266 not in ids:
        #     continue
        for id, corner in zip(ids, undist_kypt):
            corner = corner.squeeze().astype(int)
            cv2.putText(img_tmp, str(id), tuple(corner[0]), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            for i in range(4):
                cv2.circle(img_tmp, tuple(corner[i]), 3, (0, 0, 255), -1) # red circle for corners

        for mid in marker_id:
            if mid not in ids or cor_3d[mid] is None:
                continue
            proj_mtx = cammat[serial_num]
            pt_3d = cor_3d[mid]
            
            pt_3d_hom = np.concatenate([pt_3d, np.ones((4, 1))], axis=1)

            for i in range(4):
                pt_2d = proj_mtx @ pt_3d_hom[i:i+1].T
                pt_2d = (pt_2d / pt_2d[2])[:2]
                cv2.circle(img_tmp, (int(pt_2d[0]),int(pt_2d[1])), 3, (255, 0, 0), -1) # blue circle for 3D points
        
        os.makedirs(os.path.join(root_dir, "debug"), exist_ok=True)
        cv2.imwrite(os.path.join(root_dir, "debug", img_name), img_tmp)

marker_3d = {}
for mid in marker_id:
    if mid not in cor_3d or cor_3d[mid] is None:
        continue
    marker_3d[mid] = cor_3d[mid]
# np.save(os.path.join(root_dir, "marker_3d.npy"), marker_3d)

p1 = np.mean(marker_3d[1],axis=0)
p2 = np.mean(marker_3d[2],axis=0)
p3 = np.mean(marker_3d[3],axis=0)
p4 = np.mean(marker_3d[4],axis=0)

obj_rx = (p3+p4-p2-p1)
obj_rx = obj_rx / np.linalg.norm(obj_rx)

obj_ry = (p2+p4-p1-p3)
obj_ry = obj_ry / np.linalg.norm(obj_ry)

obj_rz = np.cross(obj_rx, obj_ry)
obj_rz = obj_rz / np.linalg.norm(obj_rz)

obj_trans = (p1+p2+p3+p4) / 4 + obj_rz * 0.05

obj_T = np.eye(4)
obj_T[:3,0] = obj_rx
obj_T[:3,1] = obj_ry
obj_T[:3,2] = obj_rz
obj_T[:3,3] = obj_trans

marker_offset = {}
for id in [176,177,178]:
    marker_offset[id] = (np.linalg.inv(obj_T)[:3,:3] @ marker_3d[id].T + np.linalg.inv(obj_T)[:3,3:4]).T
    print(marker_offset[id])
np.save(os.path.join(root_dir, "marker_offset.npy"), marker_offset)