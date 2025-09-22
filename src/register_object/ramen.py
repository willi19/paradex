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

capture_image = False
if capture_image:
    os.makedirs(os.path.join(shared_dir, "object/marker_offset/ramen"), exist_ok=True)
    pc_info = get_pcinfo()
    pc_list = list(pc_info.keys())
    run_script(f"python src/capture/camera/image_client.py", pc_list)
    camera = RemoteCameraController("image", None, debug=True)

    index = int(find_latest_index(os.path.join(shared_dir, "object/marker_offset/ramen")))+1
    os.makedirs(os.path.join(shared_dir, f"object/marker_offset/ramen/{index}/image"), exist_ok=True)
    camera.start(f"shared_data/object/marker_offset/ramen/{index}/image")
    camera.end()
    camera.quit()

else:
    index = int(find_latest_index(os.path.join(shared_dir, "object/marker_offset/ramen")))
    
img_dir = os.path.join(shared_dir, f"object/marker_offset/ramen/{index}/image")
img_dict = {img_name.split(".")[0]:cv2.imread(os.path.join(img_dir, img_name)) for img_name in os.listdir(img_dir)}

intrinsic, extrinsic = load_current_camparam()
cammtx = get_cammtx(intrinsic, extrinsic)

c2r = load_latest_C2R()
r2c = np.linalg.inv(c2r)

floor_cor_3d = np.array([[ 0.69231381,  0.06524541, -0.03741436],
       [ 0.72438389,  0.08510476, -0.03718648],
       [ 0.74424467,  0.05325171, -0.03719116],
       [ 0.71232081,  0.0334489 , -0.03765076]])
                        # triangulate_marker(img_dict, intrinsic, extrinsic,'4X4_50')
obj_cor_3d = triangulate_marker(img_dict, intrinsic, extrinsic)
marker_3d = {}

for id, cor in obj_cor_3d.items():
    if cor is None or cor[0][0] is None:
        continue
    marker_3d[id] = (r2c[:3,:3] @ cor.T + r2c[:3,3:]).T

marker_3d[3] = floor_cor_3d

# x_dir = ((floor_cor_3d[0]+floor_cor_3d[3]) - (floor_cor_3d[2]+floor_cor_3d[4])) / 2
# y_dir  = ((floor_cor_3d[3]+floor_cor_3d[4]) - (floor_cor_3d[1]+floor_cor_3d[2])) / 2

# x_dir /= np.linalg.norm(x_dir)
# y_dir /= np.linalg.norm(y_dir)
# z_dir = np.cross(x_dir, y_dir)
# z_dir /= np.linalg.norm(z_dir)

x_dir = np.array([1,0,0])
y_dir = np.array([0,1,0])
z_dir = np.array([0,0,1])

middle = np.mean(floor_cor_3d, axis=0)
obj_cor = np.eye(4)

obj_cor[:3, 0] = x_dir
obj_cor[:3, 1] = y_dir
obj_cor[:3, 2] = z_dir
obj_cor[:3, 3] = middle

marker_index = [178]
marker_offset = {}

obj_T = np.linalg.inv(obj_cor)
for marker_id in marker_index:
    marker_offset[marker_id] = ((obj_T[:3,:3] @ marker_3d[marker_id].T) + obj_T[:3,3:]).T
np.save(os.path.join(shared_dir, f"object/marker_offset/ramen/{index}/marker_offset.npy"), marker_offset)

os.makedirs(os.path.join(shared_dir, f"object/marker_offset/ramen/{index}/debug"), exist_ok=True)

# # mesh = trimesh.load(os.path.join(rsc_path, "object", "ramen", "ramen.obj"))
# # mesh.apply_transform(obj_cor)

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


# # renderer = BatchRenderer(intrinsic_list, extrinsic_list, width=2048, height=1536, device='cuda')
# # frame, mask = project_mesh_nvdiff(mesh, renderer)
# # mask = mask.detach().cpu().numpy()[:,:,:,0].astype(np.bool_)

    
for i, serial_num in enumerate(serial_list):
    img = undistort_img(img_dict[serial_num], intrinsic[serial_num])
    
    robot_2d = []
    
    for id, cor in marker_3d.items():
        robot_2d.append(project(cammtx_list[i], cor))
    draw_aruco(img, robot_2d, list(marker_3d.keys()))
    # overlay_mask(img, mask[i], np.array((255,0, 0)), 0.7)
    
    cv2.imwrite(os.path.join(shared_dir, f"object/marker_offset/ramen/{index}/debug/{serial_num}.png"), img)
