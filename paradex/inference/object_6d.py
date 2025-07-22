import os
import cv2
import numpy as np

from paradex.io.capture_pc.camera_main import RemoteCameraController
from paradex.image.aruco import triangulate_marker
from paradex.geometry.math import rigid_transform_3D
from paradex.io.capture_pc.connect import run_script
from paradex.utils.env import get_pcinfo
from paradex.utils.file_io import load_current_camparam, home_path, shared_dir, load_latest_C2R

def get_current_object_6d(obj_name):
    image_path = f'shared_data/inference/obj_6D/image'
    pc_info = get_pcinfo()
    pc_list = list(pc_info.keys())
    run_script(f"python src/capture/camera/image_client.py", pc_list)

    camera_loader = RemoteCameraController("image", None)
    camera_loader.start(image_path)
    camera_loader.end()
    camera_loader.quit()
    image_list = os.listdir(os.path.join(home_path, image_path))

    img_dict = {}
    for img_name in image_list:
        img_dict[img_name.split(".")[0]] = cv2.imread(os.path.join(home_path, image_path, img_name))

    intrinsic, extrinsic = load_current_camparam()
    c2r = load_latest_C2R()
    cor_3d = triangulate_marker(img_dict, intrinsic, extrinsic)

    marker_offset = np.load(os.path.join(shared_dir, "marker_offset", obj_name, "0", "marker_offset.npy"), allow_pickle=True).item()
    marker_id = list(marker_offset.keys())

    A = []
    B = []
    for mid in marker_id:
        if mid not in cor_3d:
            continue
        
        A.append(marker_offset[mid])
        B.append(cor_3d[mid])
        
    A = np.concatenate(A)
    B = np.concatenate(B)

    pick_6D = np.linalg.inv(c2r) @ rigid_transform_3D(A, B)
    
    return pick_6D