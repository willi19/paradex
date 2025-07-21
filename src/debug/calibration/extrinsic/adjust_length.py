import argparse
import os
import cv2
from paradex.utils.file_io import find_latest_directory, home_path, download_dir, shared_dir, load_current_camparam, load_current_camparam_fixed
import numpy as np
from paradex.geometry.triangulate import triangulate
from paradex.image.merge import merge_image
from paradex.image.aruco import draw_keypoint
from paradex.image.projection import get_cammtx, project_point

extrinsic_dir = os.path.join(shared_dir,"extrinsic")
cam_param_dir = os.path.join(shared_dir, "cam_param")
dir = "config"

if __name__ == "__main__":
    name_list = ['20250720_024117', '20250720_025258', '20250720_031957', '20250720_033814']
    for name in name_list:
        intrinsics, extrinsics = load_current_camparam(name)
        cammtx = get_cammtx(intrinsics, extrinsics)
        
        # intrinsic, extrinsic = load_current_camparam_fixed(name)
        # cammtx_fixed = get_cammtx(intrinsic_fix, extrinsic_fix)
        
        root_dir = os.path.join(shared_dir, "extrinsic_test")
        index_list = os.listdir(root_dir)

        index_list.sort()
        if len(index_list) == 0:
            print("No valid directories found.")
            exit()

        proj_err_tot = []
        proj_err_serial = {}
        
        for index in index_list:
            file_list = os.listdir(os.path.join(root_dir, index, "keypoint"))
            kypt_dict = {}
            image_dict = {}

            kypt_2d_dict = {}
            proj_err = {}
            
            for kypt_file in file_list:
                if "cor" not in kypt_file:
                    continue
                serial_num = kypt_file.split("_")[0]
                
                image_dict[serial_num] = cv2.imread(os.path.join(shared_dir, "extrinsic_test", index, "image", f"{serial_num}.png"))
                kypt_2d_dict[serial_num] = {}
                
                cv2.putText(image_dict[serial_num], serial_num, (80, 120), 1, 10, (255,255,255), 3)
                
                ids = np.load(os.path.join(root_dir, index, "keypoint", f"{serial_num}_id.npy"))
                if len(ids) == 0:
                    continue
                kypt = np.load(os.path.join(root_dir, index, "keypoint", kypt_file))

                int_mat = np.array(intrinsics[serial_num]['original_intrinsics'])
                ext_mat = np.array(extrinsics[serial_num])
                int_dist = np.array(intrinsics[serial_num]['dist_params'])

                int_undist = np.array(intrinsics[serial_num]['intrinsics_undistort'])

                if len(kypt) == 0:
                    continue
                normalized = cv2.undistortPoints(kypt, int_mat, int_dist)
                
                kypt = normalized.squeeze() * np.array(
                    [[int_undist[0, 0], int_undist[1, 1]]]
                ) + np.array(
                    [[int_undist[0, 2], int_undist[1, 2]]]
                )
                image_dict[serial_num] = draw_keypoint(image_dict[serial_num], kypt, color=(0,0,255))
                for i in range(ids.shape[0]):
                    id = ids[i][0]
                    cor = kypt[i]
                    
                    if id not in kypt_dict.keys():
                        kypt_dict[id] = {"2d": [], "projection":[]}
                    
                    kypt_dict[id]["2d"].append(cor)
                    kypt_dict[id]["projection"].append(cammtx[serial_num])
                    kypt_2d_dict[serial_num][id] = cor
                    
            kypt_3d = {}
            for i in kypt_dict.keys():
                proj_mat = np.array(kypt_dict[i]["projection"])
                kypt_2d = np.array(kypt_dict[i]["2d"])
                pt3d = triangulate(kypt_2d, proj_mat)
                if pt3d is None:
                    continue
                kypt_3d[i] = pt3d

            max_proj_err = 0
            for serial_num, kypt_2d in kypt_2d_dict.items():
                proj_err[serial_num] = []

                for id in kypt_2d.keys():
                    cor = kypt_2d[id]
                    
                    if id not in kypt_3d.keys():
                        continue

                    pt3d = kypt_3d[id]
                    pt3d_h = np.hstack((pt3d, np.ones((1))))

                    proj =  cammtx[serial_num] @ pt3d_h
                    proj = proj[:2] / proj[2]
                    # cv2.circle(image_dict[serial_num], (int(proj[0]), int(proj[1])), radius=1, color=(0,255,0), thickness=-1)
                    
                    draw_keypoint(image_dict[serial_num], proj, (0,255,0))
                    err = np.linalg.norm(proj - cor)
                    
                    proj_err[serial_num].append(err)
                    if serial_num not in proj_err_serial:
                        proj_err_serial[serial_num] = []
                    proj_err_serial[serial_num].append(err)
                    proj_err_tot.append(err)
                
                if len(kypt_2d) == 0:
                    continue
            
        for serial_num, v in proj_err_serial.items():
            if len(v) != 0:
                print(serial_num, np.mean(v), np.max(v))
        
        print(name, np.mean(proj_err_tot), np.max(proj_err_tot))