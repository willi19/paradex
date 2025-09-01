import pickle
import os
import trimesh
import numpy as np
import cv2

from paradex.utils.env import get_pcinfo
from paradex.utils.file_io import shared_dir, copy_calib_files, load_latest_C2R, load_current_camparam, rsc_path, find_latest_index
from paradex.io.capture_pc.connect import run_script
from paradex.image.projection import get_cammtx, project_mesh_nvdiff, project
from paradex.image.aruco import triangulate_marker, draw_aruco, detect_aruco, undistort_img
from paradex.visualization_.renderer import BatchRenderer
from paradex.image.overlay import overlay_mask
from paradex.io.capture_pc.camera_main import RemoteCameraController
from paradex.geometry.math import rigid_transform_3D
from paradex.inference.object_6d import get_current_object_6d

os.makedirs(os.path.join(shared_dir, "object/marker_offset/book"), exist_ok=True)
pc_info = get_pcinfo()
pc_list = list(pc_info.keys())

# 박스 크기 (mm 단위)
width = 26.2 / 100   # x 축
height = 17.0 / 100 # y 축  
depth = 4.3 / 100    # z 축

# trimesh로 박스 생성 (크기를 지정)
mesh = trimesh.creation.box(extents=[width, height, depth])
mesh.export(os.path.join(rsc_path, "object", "book", "book.obj"))
capture_image = False
if capture_image:
    run_script(f"python src/capture/camera/image_client.py", pc_list)
    camera = RemoteCameraController("image", None)

    index = int(find_latest_index(os.path.join(shared_dir, "object/marker_offset/book")))+1
    os.makedirs(os.path.join(shared_dir, f"object/marker_offset/book/{index}/image"), exist_ok=True)
    camera.start(f"shared_data/object/marker_offset/book/{index}/image")
    camera.end()
    camera.quit()

else:
    index = int(find_latest_index(os.path.join(shared_dir, "object/marker_offset/book")))

img_dir = os.path.join(shared_dir, f"object/marker_offset/book/{index}/image")
img_dict = {img_name.split(".")[0]:cv2.imread(os.path.join(img_dir, img_name)) for img_name in os.listdir(img_dir)}
serial_list = list(img_dict.keys())
serial_list.sort()

intrinsic, extrinsic = load_current_camparam()
cammtx = get_cammtx(intrinsic, extrinsic)

c2r = load_latest_C2R()
r2c = np.linalg.inv(c2r)

floor_cor_3d = triangulate_marker(img_dict, intrinsic, extrinsic,'4X4_50')
obj_cor_3d = triangulate_marker(img_dict, intrinsic, extrinsic)
marker_index = [650, 651, 652, 653]

marker_3d = {}

for id, cor in floor_cor_3d.items():
    if cor is None or cor[0][0] is None:
        continue
    marker_3d[id] = cor
    
for id, cor in obj_cor_3d.items():
    marker_3d[id] = cor

x_dir = np.mean(((marker_3d[1]+marker_3d[3]) - (marker_3d[2]+marker_3d[4])), axis=0) / 2
y_dir  = np.mean(((marker_3d[3]+marker_3d[4]) - (marker_3d[1]+marker_3d[2])), axis=0) / 2

x_dir /= np.linalg.norm(x_dir)
y_dir /= np.linalg.norm(y_dir)
z_dir = np.cross(x_dir, y_dir)
z_dir /= np.linalg.norm(z_dir)

middle = np.mean((marker_3d[3]+marker_3d[4]+marker_3d[1]+marker_3d[2]), axis=0) / 4
obj_cor = np.eye(4)

obj_cor[:3, 0] = y_dir
obj_cor[:3, 1] = z_dir
obj_cor[:3, 2] = x_dir
obj_cor[:3, 3] = middle + (height-0.001) / 2 * z_dir - 0.002 * y_dir
obj_T = np.linalg.inv(obj_cor)

marker_offset = {}
for marker_id in marker_index:
    marker_offset[marker_id] = ((obj_T[:3,:3] @ marker_3d[marker_id].T) + obj_T[:3,3:]).T

np.save(os.path.join(shared_dir, f"object/marker_offset/book/{index}/marker_offset.npy"), marker_offset)

mesh.apply_transform(obj_cor)

intrinsic_list = []
extrinsic_list = []
for serial_name in serial_list:
    extmat = extrinsic[serial_name]
    extrinsic_list.append(extmat)    
    intrinsic_list.append(intrinsic[serial_name]['intrinsics_undistort'])
    
renderer = BatchRenderer(intrinsic_list, extrinsic_list, width=2048, height=1536, device='cuda')
frame, mask = project_mesh_nvdiff(mesh, renderer)
mask = mask.detach().cpu().numpy()[:,:,:,0].astype(np.bool_)

os.makedirs(os.path.join(shared_dir, f"object/marker_offset/book/{index}/debug"), exist_ok=True)

for i, serial_num in enumerate(serial_list):
    img = undistort_img(img_dict[serial_num], intrinsic[serial_num])
    
    robot_2d = []
    
    for id, cor in obj_cor_3d.items():
        if cor is not None:
            robot_2d.append(project(cammtx[serial_num], cor))
    draw_aruco(img, robot_2d, list(obj_cor_3d.keys()))
    
    robot_2d = []
    
    for id, cor in floor_cor_3d.items():
        if cor is not None:
            robot_2d.append(project(cammtx[serial_num], cor))
    draw_aruco(img, robot_2d, list(floor_cor_3d.keys()))
    
    overlay_mask(img, mask[i], np.array((255,0, 0)), 0.7)
    
    cv2.imwrite(os.path.join(shared_dir, f"object/marker_offset/book/{index}/debug/{serial_num}.png"), img)
    