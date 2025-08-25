import os
import cv2
import numpy as np
import shutil
import torch
import copy

from paradex.io.capture_pc.camera_main import RemoteCameraController
from paradex.image.aruco import triangulate_marker
from paradex.geometry.math import rigid_transform_3D
from paradex.io.capture_pc.connect import run_script
from paradex.utils.env import get_pcinfo
from paradex.utils.file_io import load_current_camparam, home_path, shared_dir, load_latest_C2R
from paradex.pose_utils.optimize_initial_frame import object6d_silhouette
from paradex.pose_utils.io import get_obj_info
from paradex.pose_utils.vis_utils_nvdiff import BatchRenderer
from paradex.pose_utils.renderer_utils import show_res_cv2, grid_image

def get_current_object_6d_marker(obj_name):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    image_path = f'shared_data/inference/obj_6D/image'
    scene_path = f'{shared_dir}/inference/obj_6D/'
    if os.path.exists(os.path.join(shared_dir, "inference", "obj_6D")):
        shutil.rmtree(os.path.join(shared_dir, "inference", "obj_6D"))
        os.makedirs(os.path.join(shared_dir, "inference", "obj_6D", "image"))
    
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
    
    
    #### Object pose estimation ####
    cor_3d = triangulate_marker(img_dict, intrinsic, extrinsic)

    marker_offset = np.load(os.path.join(shared_dir, "marker_offset", obj_name, "0", "marker_offset.npy"), allow_pickle=True).item()
    marker_id = list(marker_offset.keys())

    A = []
    B = []
    for mid in marker_id:
        if mid not in cor_3d or cor_3d[mid] is None:
            continue
        
        A.append(marker_offset[mid])
        B.append(cor_3d[mid])
    
    A = np.concatenate(A)
    B = np.concatenate(B)
    
    obj_T = rigid_transform_3D(A, B)
    obj_T = np.linalg.inv(c2r) @ obj_T
    
    return obj_T

def get_current_object_6d(obj_name):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    image_path = f'shared_data/inference/obj_6D/image'
    scene_path = f'{shared_dir}/inference/obj_6D/'
    if os.path.exists(os.path.join(shared_dir, "inference", "obj_6D")):
        shutil.rmtree(os.path.join(shared_dir, "inference", "obj_6D"))
        os.makedirs(os.path.join(shared_dir, "inference", "obj_6D", "image"))
    
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
    
    
    #### Object pose estimation ####
    # cor_3d = triangulate_marker(img_dict, intrinsic, extrinsic)

    # marker_offset = np.load(os.path.join(shared_dir, "marker_offset", obj_name, "0", "marker_offset.npy"), allow_pickle=True).item()
    # marker_id = list(marker_offset.keys())

    # A = []
    # B = []
    # for mid in marker_id:
    #     if mid not in cor_3d or cor_3d[mid] is None:
    #         continue
        
    #     A.append(marker_offset[mid])
    #     B.append(cor_3d[mid])
    
    # A = np.concatenate(A)
    # B = np.concatenate(B)
    object6d_silhouette(
        scene_path,
        obj_name, 
        renderer_type = 'nvdiffrast',
        wo_simplify = False,
        rescale_factor = 0.25,
        confidence = 0.002,
        learning_rate = 1e-2,
        iou_weight = 0.5,
        tg_weight = 10.0,
        use_rgb = False,
        image_dir = os.path.join(home_path, image_path)
    )
    obj_mesh, obj_T, _ = get_obj_info(scene_path, obj_name, obj_status_path=None)
    proj_matrix = {}
    camera_params = {}
    for cam_id in extrinsic:
        proj_matrix[cam_id] = np.array(intrinsic[cam_id]['intrinsics_undistort']).reshape(3,3) @ extrinsic[cam_id]
        extrinsic_np = np.array(extrinsic[cam_id])
        intrinsic_np = np.array(intrinsic[cam_id]['intrinsics_undistort'])
        K = intrinsic_np
        R = extrinsic_np[:, :3]  # 3x3 회전행렬
        t = extrinsic_np[:, 3] 
        camera_params[cam_id] = {'K': K, 'R': R, 't': t}
    
    renderer_dict = {}

    for cam_id in camera_params.keys():
        R = np.array(camera_params[cam_id]["R"])
        t = np.array(camera_params[cam_id]["t"])[:, np.newaxis]
        K = np.array(camera_params[cam_id]["K"])

        cam_extr = np.hstack([R, t])  # (3, 4)
        renderer_dict[cam_id] = BatchRenderer(
            opengl=False,
            cam_intrinsics=[K],
            cam_extrinsics=[cam_extr],
            width=2048,
            height=1536,
            near=0.01,
            far=2.0,
            device=device,
        )
      
    with torch.no_grad():
        transformed_obj_mesh = copy.deepcopy(obj_mesh).apply_transform(obj_T)
        verts = torch.tensor(transformed_obj_mesh.vertices, dtype=torch.float32).to(device).unsqueeze(0)
        faces_tensor = torch.tensor(transformed_obj_mesh.faces, dtype=torch.int32).to(device)
        vtx_col = torch.ones_like(verts)  # (1, V, 3)
        col_idx = faces_tensor  # (F, 3)

        obj_dict = {
        'type': 'vertex_color',
        'verts': verts,
        'faces': faces_tensor,
        'vtx_col': vtx_col,
        'col_idx': col_idx,
        }
        
        mask_images = []
        for cam_id in renderer_dict.keys():
            target_image = os.path.join(home_path, image_path, str(cam_id) + ".png")    
            image = None 
            if os.path.exists(target_image):
                image = cv2.imread(target_image)
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            else:
                image = np.zeros((1536, 2048, 3), dtype=np.uint8)

            _, mask_soft = renderer_dict[cam_id].render(obj_dict)  # mask_soft: (1, H, W, 1)
            mask_np = mask_soft[0].squeeze(-1).cpu().detach().numpy()  # (H, W)
            overlay_image = show_res_cv2(cam_id = cam_id, image = image, masks = mask_np, scores = [1.0], alpha = 0.5)

            mask_images.append(overlay_image)
        grid_image(mask_images, "{:05d}.jpg".format(int(0)), scene_path, "object_projection")
        print(f"initial object projection saved")
        
    

    # obj_T = np.linalg.inv(c2r) @ obj_T
    # shutil.rmtree(os.path.join(home_path, image_path))
    
    return obj_T # camera space
