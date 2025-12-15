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
import subprocess
import logging
logging.getLogger("pycolmap").setLevel(logging.WARNING)

def add_camera(db, intrinsics_dict):
    camera_id = {}
    for serial_num in intrinsics_dict.keys():
        intrinsic = intrinsics_dict[serial_num]
        width = intrinsic["width"]
        height = intrinsic["height"]
        fx = intrinsic["intrinsics_undistort"][0][0]
        fy = intrinsic["intrinsics_undistort"][1][1]
        cx = intrinsic["intrinsics_undistort"][0][2]
        cy = intrinsic["intrinsics_undistort"][1][2]
        camera_id[serial_num] = db.add_camera(1, width, height, np.array([fx,fy, cx, cy]), 0)
    return db, camera_id

def add_image(db, demo_path, camera_id_dict, extrinsics_dict):
    image_dir = os.path.join(demo_path, "masked_images")  # masked_images 디렉토리
    
    rot_mat = {}
    for rot_file in os.listdir(os.path.join(demo_path, "rotation")):
        frame_idx = rot_file.split(".npy")[0]
        rot_mat[int(frame_idx)] = np.load(os.path.join(demo_path, "rotation", rot_file))
        
    for serial_num in os.listdir(image_dir):
        img_path = os.path.join(image_dir, serial_num)
        img_files = sorted(os.listdir(img_path))
        extrinsic = extrinsics_dict[serial_num]
        camera_id = camera_id_dict[serial_num]
        print(f"Adding images for {serial_num}... {camera_id}")
        
        for img_file in img_files:
            frame_idx = int(img_file.split(".")[0].split("_")[-1])
            if frame_idx not in rot_mat:
                print(f"  Warning: No rotation found for frame {frame_idx} in {demo_path}, skipping.")
                continue
            
            T =  np.concatenate([extrinsic, np.array([[0, 0, 0, 1]])], axis=0) @ rot_mat[int(frame_idx)]
            R = T[:3, :3]
            t = T[:3, 3]
            quat = Rotation.from_matrix(R).as_quat()
            qvec = np.array([quat[0], quat[1], quat[2], quat[3]])
            
            # mask 추가
            image_id = db.add_image(
                os.path.join(serial_num, img_file), 
                camera_id, 
                qvec, 
                t
            )
    
    return db

def export_initial_poses(demo_path):
    colmap_dir = os.path.join(demo_path, "colmap")
    database_path = os.path.join(colmap_dir, "database.db")
    sparse_dir = os.path.join(colmap_dir, "sparse", "tmp_initial")
    os.makedirs(sparse_dir, exist_ok=True)
    
    # DB에서 직접 cameras.bin, images.bin 생성
    reconstruction = pycolmap.Reconstruction()
    
    # DB에서 카메라와 이미지 읽기
    db = COLMAPDatabase.connect(database_path)
    
    # 카메라 추가
    for camera_id in db.execute("SELECT camera_id, model, width, height, params FROM cameras"):
        cam = pycolmap.Camera(
            camera_id=camera_id[0],
            model=camera_id[1],
            width=camera_id[2],
            height=camera_id[3],
            params=np.frombuffer(camera_id[4], dtype=np.float64)
        )
        reconstruction.add_camera(cam)
    
    # 이미지 추가
    for img_row in db.execute("SELECT image_id, name, camera_id, prior_qw, prior_qx, prior_qy, prior_qz, prior_tx, prior_ty, prior_tz FROM images"):
        img = pycolmap.Image(
            id=img_row[0],
            name=img_row[1],
            camera_id=img_row[2],
            cam_from_world=pycolmap.Rigid3d(
                rotation=pycolmap.Rotation3d([img_row[3], img_row[4], img_row[5], img_row[6]]),
                translation=np.array([img_row[7], img_row[8], img_row[9]])
            )
        )
        reconstruction.add_image(img)
    
    db.close()
    
    # Binary 형식으로 저장
    reconstruction.write(sparse_dir)
        
def run_point_triangulator(demo_path):
    colmap_dir = os.path.join(demo_path, "colmap")
    database_path = os.path.join(colmap_dir, "database.db")
    images_dir = os.path.join(demo_path, "masked_images")
    output_dir = os.path.join(colmap_dir, "sparse")
    input_dir = os.path.join(colmap_dir, "sparse", "tmp_initial")
    
    os.makedirs(output_dir, exist_ok=True)
    
    export_initial_poses(demo_path)
    
    cmd = [
        "colmap", "point_triangulator",
        "--database_path", database_path,
        "--image_path", images_dir,
        "--input_path", input_dir,
        "--output_path", output_dir,
        "--Mapper.tri_max_transitivity", "10",
        "--Mapper.tri_create_max_angle_error", "10.0",
        "--Mapper.tri_continue_max_angle_error", "10.0",
        "--Mapper.tri_merge_max_reproj_error", "16.0",
        "--Mapper.tri_complete_max_reproj_error", "16.0",
        "--Mapper.tri_re_max_angle_error", "10.0",
        "--Mapper.tri_re_min_ratio", "0.05",
        "--Mapper.tri_min_angle", "0.1",
    ]
    
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, # 표준 출력을 무시
    stderr=subprocess.DEVNULL  # 표준 에러 출력을 무시 (경고나 오류도 안 보임))
    )
    
def generate_db(demo_path):
    database_path = os.path.join(demo_path, "colmap", "database.db")
    os.makedirs(os.path.dirname(database_path), exist_ok=True)
    
    intrinsics, extrinsics = load_camparam(demo_path)
        
    if os.path.exists(database_path):
        os.remove(database_path)
        
    db = COLMAPDatabase.connect(database_path)
    db.create_tables()
    db, camera_id = add_camera(db, intrinsics)
    db = add_image(db, demo_path, camera_id, extrinsics)
    db.commit()
    db.close()

def run_colmap(demo_path):
    """Run COLMAP feature extraction and matching"""
    colmap_dir = os.path.join(demo_path, "colmap")
    database_path = os.path.join(colmap_dir, "database.db")
    image_path = os.path.join(demo_path, "masked_images")
    mask_path = os.path.join(demo_path, "masks")
    
    reader_options = pycolmap.ImageReaderOptions(
        mask_path=mask_path # ImageReaderOptions를 통해 mask_path 전달
    )

    # Feature extraction
    pycolmap.extract_features(
        database_path=database_path,
        image_path=image_path,
        # 2. reader_options 인수에 전달
        reader_options=reader_options, 
        sift_options={
            "max_num_features": 8192,
        }
    )
    
    # Sequential이 exhaustive보다 훨씬 빠르고, turntable처럼 순차적인 경우 적합
    pycolmap.match_sequential(
        database_path=database_path,
        # sequential_options을 matching_options으로 변경
        matching_options={ 
            "overlap": 10,  # 각 이미지당 앞뒤 10개와 매칭
            "quadratic_overlap": True,  # 시작/끝 부분 더 많이 매칭
        }
    )
    
    
root_dir = os.path.join(home_path, "paradex_download/capture/object_turntable")
obj_list = sorted(os.listdir(root_dir))
err_list = []
for obj_name in tqdm.tqdm(['big_green_spray']):
    index_list = os.listdir(os.path.join(root_dir, obj_name))
    for index in index_list:
        try:
            demo_path = os.path.join(root_dir, obj_name, index)

            print(f"Generating COLMAP database for {obj_name}/{index}...")
            generate_db(demo_path)

            print(f"Running COLMAP reconstruction...")
            run_colmap(demo_path)
            
            print(f"✓ Done: {obj_name}/{index}")
            run_point_triangulator(demo_path)

        except Exception as e:
            err_list.append((obj_name, index))
            print(f"Error occurred for {obj_name}/{index}: {e}")

print("Errors occurred for the following objects/indices:")
for obj_name, index in err_list:
    print(f"- {obj_name}/{index}")
            