import pickle
import os
import trimesh
import numpy as np
import cv2
import copy

from paradex.utils.env import get_pcinfo
from paradex.utils.file_io import shared_dir, copy_calib_files, load_latest_C2R, load_current_camparam, rsc_path, find_latest_index
from paradex.io.capture_pc.connect import run_script
from paradex.image.projection import get_cammtx, project_mesh_nvdiff, project
from paradex.image.aruco import triangulate_marker, draw_aruco, detect_aruco, undistort_img
from paradex.visualization_.renderer import BatchRenderer
from paradex.image.overlay import overlay_mask
from paradex.io.capture_pc.camera_main import RemoteCameraController
from paradex.geometry.math import rigid_transform_3D
from paradex.inference.object_6d import get_current_object_6d, get_book_goal_position

os.makedirs(os.path.join(shared_dir, "erasethis/marker"), exist_ok=True)
pc_info = get_pcinfo()
pc_list = list(pc_info.keys())

run_script(f"python src/capture/camera/image_client.py", pc_list)
camera = RemoteCameraController("image", None,debug=True)

index = int(find_latest_index(os.path.join(shared_dir, "erasethis/marker")))+1
os.makedirs(os.path.join(shared_dir, f"erasethis/marker/{index}/image"), exist_ok=True)
camera.start(f"shared_data/erasethis/marker/{index}/image")
camera.end()
camera.quit()

img_dir = os.path.join(shared_dir, f"erasethis/marker/{index}/image")
img_dict = {img_name.split(".")[0]:cv2.imread(os.path.join(img_dir, img_name)) for img_name in os.listdir(img_dir)}
serial_list = list(img_dict.keys())
serial_list.sort()

intrinsic, extrinsic = load_current_camparam()
cammtx = get_cammtx(intrinsic, extrinsic)

c2r = load_latest_C2R()
r2c = np.linalg.inv(c2r)

marker1 = triangulate_marker(img_dict, intrinsic, extrinsic,'4X4_50')
marker2 = triangulate_marker(img_dict, intrinsic, extrinsic)
marker_3d = {}

for id, cor in marker1.items():
    if cor is None or cor[0][0] is None:
        continue
    marker_3d[id] = cor

for id, cor in marker2.items():
    if cor is None or cor[0][0] is None:
        continue
    marker_3d[id] = cor
        
os.makedirs(os.path.join(shared_dir, f"erasethis/marker/{index}/debug"), exist_ok=True)

intrinsic_list = []
extrinsic_list = []
for serial_name in serial_list:
    extmat = extrinsic[serial_name]
    extrinsic_list.append(extmat @ c2r)    
    intrinsic_list.append(intrinsic[serial_name]['intrinsics_undistort'])
    
renderer = BatchRenderer(intrinsic_list, extrinsic_list, width=2048, height=1536, device='cuda')

debug_img_dict = {}

for i, serial_num in enumerate(serial_list):
    debug_img_dict[serial_num] = undistort_img(img_dict[serial_num], intrinsic[serial_num])
    robot_2d = []
    
    for id, cor in marker_3d.items():
        robot_2d.append(project(cammtx[serial_num], cor))
    draw_aruco(debug_img_dict[serial_num], robot_2d, list(marker_3d.keys()))

for i, serial_num in enumerate(serial_list):
    cv2.imwrite(os.path.join(shared_dir, f"erasethis/marker/{index}/debug/{serial_num}.png"), debug_img_dict[serial_num])
