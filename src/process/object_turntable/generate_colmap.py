import argparse
import os
import pycolmap
import multiprocessing as mp
import tqdm
import numpy as np
from scipy.spatial.transform import Rotation

from paradex.calibration.colmap import *
from paradex.calibration.utils import load_camparam 
from paradex.utils.path import home_path, shared_dir

def add_camera(db, intrinsics_dict, extrinsics_dict, rot_mat, img_dir):
    for serial_num in intrinsics_dict.keys(): # add 50 cameras
        intrinsic = intrinsics_dict[serial_num]
        width = intrinsic["width"]
        height = intrinsic["height"]
        fx = intrinsic["intrinsics_undistort"][0][0]
        fy = intrinsic["intrinsics_undistort"][1][1]
        k1 = 0
        k2 = 0
        p1 = 0
        p2 = 0
        cx = intrinsic["intrinsics_undistort"][0][2]
        cy = intrinsic["intrinsics_undistort"][1][2]

        camera_id = db.add_camera(4, width, height, np.array([fx,fy, cx, cy, k1, k2, p1, p2]), 0)
        
        extrinsic = np.eye(4)
        extrinsic[:3] = extrinsics_dict[serial_num]
        
        for idx in range(rot_mat.shape[0]): # add images
            img_path = f"{serial_num}/frame_{idx+1:06d}.jpg"
            T = rot_mat[idx] @ extrinsic
            
            R = T[:3, :3]
            t = T[:3, 3]
            quat = Rotation.from_matrix(R).as_quat()
            # COLMAP은 [w, x, y, z] 순서
            qvec = np.array([quat[3], quat[0], quat[1], quat[2]])
            image_id = db.add_image(img_path, camera_id, qvec, t)

    return db

def add_image(db, img_dir, camera_id, extrinsic, rot):
    for img_path in os.listdir(img_dir):
        img_full_path =  os.path.join(img_dir, img_path)
        # set extrinsic
        R = extrinsic[:3, :3]
        t = extrinsic[:3, 3]
        qvec = pycolmap.qvec_from_rotmat(R)
        image_id = db.add_image(img_path, camera_id, qvec, t)
        
    return db
    
def generate_db(database_path):
    image_dir = os.path.join(home_path, "paradex_download", demo_path, "images")

    database_path = os.path.join(home_path, "paradex_download", demo_path, "colmap", "database.db")
    os.makedirs(os.path.dirname(database_path), exist_ok=True)
    
    intrinsics, extrinsics = load_camparam(os.path.join(shared_dir, demo_path))
    rot_mat = np.load(os.path.join(shared_dir, demo_path, "rot.npy"))
        
    if os.path.exists(database_path):
        os.remove(database_path)
        
    db = COLMAPDatabase.connect(database_path)
    db.create_tables()

    db = add_camera(db, intrinsics, extrinsics, rot_mat, image_dir)

    db.commit()
    db.close()


root_dir = os.path.join(home_path, "paradex_download/capture/object_turntable")
obj_list = ['pepper_tuna']

for obj_name in obj_list:
    index_list = os.listdir(os.path.join(root_dir, obj_name))
    for index in index_list:
        demo_path = os.path.join("capture/object_turntable", obj_name, index)

        
        generate_db(demo_path)
        
        print(f"Generated COLMAP database for {demo_path}")