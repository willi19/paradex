import numpy as np
import os
import trimesh

from paradex.visualization.visualizer.viser import ViserViewer
from paradex.calibration.utils import load_camparam, load_c2r
from paradex.utils.path import home_path

def add_image(demo_path):
    intrinsics_dict, extrinsics_dict = load_camparam(os.path.join(demo_path))
    c2r = load_c2r(os.path.join(demo_path))
    
    image_dir = os.path.join(demo_path, "masked_images")  # masked_images 디렉토리
    vis = ViserViewer()
    obj= trimesh.creation.icosphere(radius=0.02, center=[0,0,0])
    
    rot_mat = {}
    for rot_file in os.listdir(os.path.join(demo_path, "rotation")):
        frame_idx = rot_file.split(".npy")[0]
        rot_mat[int(frame_idx)] = np.load(os.path.join(demo_path, "rotation", rot_file))
        
    for serial_num in os.listdir(image_dir)[:1]:
        img_path = os.path.join(image_dir, serial_num)
        img_files = sorted(os.listdir(img_path))
        extrinsic = extrinsics_dict[serial_num]
        for img_file in img_files:
            frame_idx = int(img_file.split(".")[0].split("_")[-1])
            if frame_idx % 200 != 1 or frame_idx < 160:
                continue
            # if int(frame_idx) < 60:
            #     continue
            rot_extrinsic = np.concatenate([extrinsic, np.array([[0, 0, 0, 1]])], axis=0) @ rot_mat[int(frame_idx)] 
            # T = np.concatenate([T, np.array([[0,0,0,1]])], axis=0)
            T =  np.linalg.inv(rot_extrinsic) # @ np.linalg.inv(rot_mat[int(frame_idx)])
            vis.add_camera(name=f"{serial_num}_{img_file}", extrinsic=T, intrinsic=intrinsics_dict[serial_num])

    return vis
    

demo_path = os.path.join(home_path, "paradex_download/capture/object_turntable/alarm")
index = os.listdir(demo_path)[0]
demo_path = os.path.join(demo_path, index)
intrinsics_dict, extrinsics_dict = load_camparam(os.path.join(demo_path))
vis = add_image(demo_path)
vis.start_viewer()