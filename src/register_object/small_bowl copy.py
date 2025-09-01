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

pc_info = get_pcinfo()
pc_list = list(pc_info.keys())

file_path = os.path.join(shared_dir, "Icra/rsc/smallbowl2/aruco_objctr.pickle")

# pickle 파일 로드 (with문 사용 - 권장)
with open(file_path, 'rb') as f:
    marker_offset = pickle.load(f)


mesh = trimesh.load(os.path.join(shared_dir, "Icra/rsc/smallbowl2/smallbowl2.obj"))
capture_image = False
if capture_image:
    run_script(f"python src/capture/camera/image_client.py", pc_list)
    camera = RemoteCameraController("image", None)

    index = int(find_latest_index(os.path.join(shared_dir, "object/marker_offset/smallbowl2")))+1
    os.makedirs(os.path.join(shared_dir, f"object/marker_offset/smallbowl2/{index}/image"), exist_ok=True)
    camera.start(f"shared_data/object/marker_offset/smallbowl2/{index}/image")
    camera.end()
    camera.quit()

else:
    index = int(find_latest_index(os.path.join(shared_dir, "object/marker_offset/smallbowl2")))

img_dir = os.path.join(shared_dir, f"object/marker_offset/smallbowl2/{index}/image")
img_dict = {img_name.split(".")[0]:cv2.imread(os.path.join(img_dir, img_name)) for img_name in os.listdir(img_dir)}
serial_list = list(img_dict.keys())
serial_list.sort()

intrinsic, extrinsic = load_current_camparam()
cammtx = get_cammtx(intrinsic, extrinsic)

c2r = load_latest_C2R()
r2c = np.linalg.inv(c2r)

obj_cor_3d = triangulate_marker(img_dict, intrinsic, extrinsic)
marker_index = [608, 609, 606]

A = []
for marker_id in marker_index:
    marker_pos = np.zeros((4, 3))
    for i in range(4):
        marker_pos[i] = marker_offset[marker_id][i]
    A.append(marker_pos)
        
A = np.concatenate(A)
B = np.concatenate([obj_cor_3d[marker_id] for marker_id in marker_index])
##3333333333333333  obj_T = rigid_transform_3D(A, B)

obj_T = get_current_object_6d("servingbowl_small", False, img_dict)
mesh.apply_transform(obj_T)

intrinsic_list = []
extrinsic_list = []
for serial_name in serial_list:
    extmat = extrinsic[serial_name]
    extrinsic_list.append(extmat)    
    intrinsic_list.append(intrinsic[serial_name]['intrinsics_undistort'])
    
renderer = BatchRenderer(intrinsic_list, extrinsic_list, width=2048, height=1536, device='cuda')
frame, mask = project_mesh_nvdiff(mesh, renderer)
mask = mask.detach().cpu().numpy()[:,:,:,0].astype(np.bool_)

os.makedirs(os.path.join(shared_dir, f"object/marker_offset/smallbowl2/{index}/debug"), exist_ok=True)

for i, serial_num in enumerate(serial_list):
    img = undistort_img(img_dict[serial_num], intrinsic[serial_num])
    
    robot_2d = []
    
    for id, cor in obj_cor_3d.items():
        if cor is not None:
            robot_2d.append(project(cammtx[serial_num], cor))
    draw_aruco(img, robot_2d, list(obj_cor_3d.keys()))
    overlay_mask(img, mask[i], np.array((255,0, 0)), 0.7)
    
    cv2.imwrite(os.path.join(shared_dir, f"object/marker_offset/smallbowl2/{index}/debug/{serial_num}.png"), img)
    