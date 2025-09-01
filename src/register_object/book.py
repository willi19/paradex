import os
import cv2
import numpy as np
import trimesh

from paradex.io.capture_pc.camera_main import RemoteCameraController
from paradex.io.capture_pc.connect import run_script

from paradex.utils.env import get_pcinfo
from paradex.utils.file_io import shared_dir, copy_calib_files, load_latest_C2R, load_current_camparam, rsc_path, find_latest_index
from paradex.image.aruco import triangulate_marker, draw_aruco, detect_aruco, undistort_img
from paradex.image.projection import get_cammtx, project_mesh_nvdiff, project
from paradex.visualization_.renderer import BatchRenderer
from paradex.image.overlay import overlay_mask

pc_info = get_pcinfo()
pc_list = list(pc_info.keys())

run_script(f"python src/capture/camera/image_client.py", pc_list)
camera = RemoteCameraController("image", None)

index = int(find_latest_index(os.path.join(shared_dir, "object/marker_offset/pringles")))+1
os.makedirs(os.path.join(shared_dir, f"object/marker_offset/pringles/{index}/image"))
camera.start(f"shared_data/object/marker_offset/pringles/{index}/image")
camera.end()
camera.quit()
img_dir = os.path.join(shared_dir, f"object/marker_offset/pringles/{index}/image")
# img_dir = os.path.join(shared_dir, f"object/marker_offset/pringles/{index}/image")
img_dict = {img_name.split(".")[0]:cv2.imread(os.path.join(img_dir, img_name)) for img_name in os.listdir(img_dir)}

intrinsic, extrinsic = load_current_camparam()
cammtx = get_cammtx(intrinsic, extrinsic)

c2r = load_latest_C2R()
r2c = np.linalg.inv(c2r)

floor_cor_3d = triangulate_marker(img_dict, intrinsic, extrinsic,'4X4_50')
obj_cor_3d = triangulate_marker(img_dict, intrinsic, extrinsic)

robot_3d = {}
marker_3d = {}

for id, cor in floor_cor_3d.items():
    if cor is None or cor[0][0] is None:
        continue
    print(cor, id)
    robot_3d[id] = np.mean((r2c[:3,:3] @ cor.T + r2c[:3,3:]).T,axis=0)
    marker_3d[id] = (r2c[:3,:3] @ cor.T + r2c[:3,3:]).T
    
for id, cor in obj_cor_3d.items():
    if cor is None or cor[0][0] is None:
        continue
    print(cor, id)
    robot_3d[id] = np.mean((r2c[:3,:3] @ cor.T + r2c[:3,3:]).T,axis=0)
    marker_3d[id] = (r2c[:3,:3] @ cor.T + r2c[:3,3:]).T
    
x_dir = ((robot_3d[1]+robot_3d[3]) - (robot_3d[2]+robot_3d[4])) / 2
y_dir  = ((robot_3d[3]+robot_3d[4]) - (robot_3d[1]+robot_3d[2])) / 2

x_dir /= np.linalg.norm(x_dir)
y_dir /= np.linalg.norm(y_dir)
z_dir = np.cross(x_dir, y_dir)
z_dir /= np.linalg.norm(z_dir)

middle = (robot_3d[3]+robot_3d[4]+robot_3d[1]+robot_3d[2]) / 4
obj_cor = np.eye(4)

obj_cor[:3, 0] = x_dir
obj_cor[:3, 1] = y_dir
obj_cor[:3, 2] = z_dir
obj_cor[:3, 3] = middle + 0.096 * z_dir

marker_index = [181]
marker_offset = {}

obj_T = np.linalg.inv(obj_cor)
for marker_id in marker_index:
    marker_offset[marker_id] = ((obj_T[:3,:3] @ marker_3d[marker_id].T) + obj_T[:3,3:]).T
np.save(os.path.join(shared_dir, f"object/marker_offset/pringles/{index}/marker_offset.npy"), marker_offset)

os.makedirs(os.path.join(shared_dir, f"object/marker_offset/pringles/{index}/debug"), exist_ok=True)

mesh = trimesh.load(os.path.join(rsc_path, "object", "pringles", "pringles.obj"))
mesh.apply_transform(obj_cor)

extrinsic_list = []
intrinsic_list = []
cammtx_list = []

serial_list = list(img_dict.keys())
serial_list.sort()

for serial_name in serial_list:
    extmat = extrinsic[serial_name]
    extrinsic_list.append(extmat @ c2r)    
    intrinsic_list.append(intrinsic[serial_name]['intrinsics_undistort'])
    cammtx_list.append(intrinsic_list[-1] @ extrinsic_list[-1])


renderer = BatchRenderer(intrinsic_list, extrinsic_list, width=2048, height=1536, device='cuda')
frame, mask = project_mesh_nvdiff(mesh, renderer)
mask = mask.detach().cpu().numpy()[:,:,:,0].astype(np.bool_)

    
for i, serial_num in enumerate(serial_list):
    img = undistort_img(img_dict[serial_num], intrinsic[serial_num])
    
    robot_2d = []
    
    for id, cor in marker_3d.items():
        robot_2d.append(project(cammtx_list[i], cor))
    draw_aruco(img, robot_2d, list(marker_3d.keys()))
    overlay_mask(img, mask[i], np.array((255,0, 0)), 0.7)
    
    cv2.imwrite(os.path.join(shared_dir, f"object/marker_offset/pringles/{index}/debug/{serial_num}.png"), img)
