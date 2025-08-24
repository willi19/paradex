from paradex.pose_utils.optimize_initial_frame import object6d_silhouette
from paradex.pose_utils.io import get_optimal_T, get_camera_params, get_obj_info
from paradex.pose_utils.vis_utils_nvdiff import BatchRenderer
from paradex.pose_utils.renderer_utils import show_res_cv2, grid_image

import pickle
import numpy as np
import torch
import copy
import cv2, os

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

scene_path = '/home/temp_id/shared_data/inference_/lookup/pringles/stand_allegro/34/'
obj_name = 'pringles'
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
    use_rgb = False
)

obj_mesh, obj_T, _ = get_obj_info(scene_path, obj_name, obj_status_path=None, device=device)


# obj_initial_status = get_optimal_T(scene_path, obj_name) 
# obj_optim_output = pickle.load(open(obj_initial_status,'rb'))
# obj_R = obj_optim_output['R'].detach().cpu().numpy()
# obj_t = obj_optim_output['t'].detach().cpu().numpy()
# obj_T0 = np.eye(4)
# obj_T0[:3,:3] = obj_R
# obj_T0[:3,3] = obj_t

_, P, camera_params = get_camera_params(scene_path)
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
        target_image = os.path.join(scene_path, "rgb_extracteds", str(cam_id), "{:05d}.jpeg".format(0))    
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