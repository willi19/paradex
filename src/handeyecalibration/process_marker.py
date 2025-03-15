from paradex.utils.marker import detect_aruco, triangulate, ransac_triangulation
import os
from paradex.utils.io import handeye_calib_path, find_latest_directory, load_cam_param
import argparse
from paradex.utils.image_util import undistort_img, undistort_points
import cv2
import numpy as np


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", type=str, default=None, help="Name of the calibration directory.")
    parser.add_argument("--cam_param_dir", type=str, default=None, help="Directory containing camera parameters.")

    args = parser.parse_args()
    if args.name is None:
        args.name = find_latest_directory(handeye_calib_path)

    he_calib_path = os.path.join(handeye_calib_path, args.name)
    intrinsic, extrinsic = load_cam_param(args.cam_param_dir)
    cammat = {}
    for serial_num in list(intrinsic.keys()):
        int_mat = intrinsic[serial_num]["intrinsics_undistort"]
        ext_mat = extrinsic[serial_num]
        cammat[serial_num] = int_mat @ ext_mat

    index_list = os.listdir(os.path.join(he_calib_path))

    for idx in index_list:
        
        img_dir = os.path.join(he_calib_path, idx, "image")
        undist_img_dir = os.path.join(he_calib_path, idx, "undist_image")
        
        img_list = os.listdir(img_dir)
        id_cor = {}
        for img_name in img_list:
            img = cv2.imread(os.path.join(img_dir, img_name))
            serial_num = img_name.split(".")[0]
            undist_img = undistort_img(img, intrinsic[serial_num])
            
            undist_kypt, ids = detect_aruco(undist_img) # Tuple(np.ndarray(1, 4, 2)), np.ndarray(N, 1)

            if ids is None:
                continue
            
            ids = ids.reshape(-1)
            for id, k in zip(ids,undist_kypt):
                k = k.squeeze()
                if id not in id_cor:
                    id_cor[id] = {"2d": [], "cammtx": []}
                id_cor[id]["2d"].append(k)
                id_cor[id]["cammtx"].append(cammat[serial_num])
        
        cor_3d = {id:ransac_triangulation(np.array(id_cor[id]["2d"]), np.array(id_cor[id]["cammtx"])) for id in id_cor}

        marker_id = [262,263,264,265,266]

        # for img_name in img_list:
        #     img = cv2.imread(os.path.join(img_dir, img_name))
        #     serial_num = img_name.split(".")[0]
        #     undist_img = undistort_img(img, intrinsic[serial_num])
        #     undist_kypt, ids = detect_aruco(undist_img)
        #     if ids is None:
        #         continue
        #     img_tmp = undist_img.copy()
        #     if 266 not in ids:
        #         continue
        #     for id, corner in zip(ids, undist_kypt):
        #         corner = corner.squeeze().astype(int)
        #         cv2.putText(img_tmp, str(id), tuple(corner[0]), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        #         for i in range(4):
        #             cv2.circle(img_tmp, tuple(corner[i]), 3, (0, 0, 255), -1)

        #     for mid in marker_id:
        #         if mid not in ids or cor_3d[mid] is None:
        #             continue
        #         proj_mtx = cammat[serial_num]
        #         pt_3d = cor_3d[mid]
        #         pt_3d_hom = np.concatenate([pt_3d, np.ones((4, 1))], axis=1)

        #         for i in range(4):
        #             pt_2d = proj_mtx @ pt_3d_hom[i].T
        #             pt_2d = (pt_2d / pt_2d[2])[:2]
        #             cv2.circle(img_tmp, (int(pt_2d[0]),int(pt_2d[1])), 3, (255, 0, 0), -1)
                
  
        #     img_tmp = cv2.resize(img_tmp, (1024, 768))
        #     cv2.imshow("original", img_tmp)
        #     cv2.waitKey(0)
        marker_3d = {}
        for mid in marker_id:
            if mid not in cor_3d or cor_3d[mid] is None:
                continue
            marker_3d[mid] = cor_3d[mid]
        np.save(os.path.join(he_calib_path, idx, "marker_3d.npy"), marker_3d)