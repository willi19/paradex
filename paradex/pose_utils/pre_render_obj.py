import os

import numpy as np
import cv2
import json
from pathlib import Path
import copy
import sys
import numpy as np
import torch
PROJECT_DIR = Path(__file__).absolute().parent.parent
print(f'PROJECT_DIR {PROJECT_DIR}')
sys.path.insert(0, str(PROJECT_DIR))
from scipy.spatial.transform import Rotation as R

from paradex.pose_utils.optimize_module import get_transformed_obj
from paradex.pose_utils.geometry import get_visualhull_ctr, deterministic_so3_fibonacci, uniform_so3_fibonacci, matrix_to_rotation_6d, get_bbox_center,\
                    rotation_6d_to_matrix, matrix_to_rotation_6d
from paradex.pose_utils.vis_utils import crop_and_resize_by_mask, make_grid_image_np, parse_objectmesh_objdict, putText

device = torch.device("cuda:0")

nas_path = Path(os.environ['NAS_PATH'])
mesh_template_dir = nas_path/'mesh_rendered'
if not os.path.exists(mesh_template_dir):
    os.makedirs(mesh_template_dir)

def get_rendered_obj(obj_nm, renderer_type='nvdiff', device='cuda'):
    import pickle
    N = 1000

    save_path = mesh_template_dir/f'{obj_nm}.pickle'
    if os.path.exists(save_path):
        return pickle.load(open(save_path, 'rb'))

    intrinsic = np.array([
			3590.9048430272187,
			0.0,
			1042.6834668507586,
			0.0,
			3592.940557422954,
			806.0097027886786,
			0.0,
			0.0,
			1.0
		]
    ).reshape((3,3))

    # Setting renderer
    from paradex.pose_utils.vis_utils_nvdiff import BatchRenderer

    width, height = 2048, 1536

    renderer = BatchRenderer(opengl=False, cam_intrinsics = [intrinsic], cam_extrinsics=[np.eye(4)[:3]], \
                                            width=width, height=height, device=device)

    obj_dict = parse_objectmesh_objdict(obj_nm, renderer_type=renderer_type, device=device, simplify=False)
    scaled = obj_dict['scaled']

    assert scaled, 'You should use scaled mesh'

    # Make Candidates
    # uniformly_sampled_rotmat_tensor = torch.tensor(uniform_so3_fibonacci(N).as_matrix(), device=device).float().
    import scipy
    random_rotations = torch.tensor(scipy.spatial.transform.Rotation.random(N).as_matrix(), device=device).float()
    uniformly_sampled_6d_tensor = matrix_to_rotation_6d(random_rotations)  # in quaternions

    rendered_rgbs = []
    rendered_sils = []
    for ridx in range(0, N):
        trasnformed_obj_dict = get_transformed_obj(obj_dict, 1.0, uniformly_sampled_6d_tensor[ridx],\
                                                    torch.tensor([0,0,1.5], device=device).float(), )
        rendered_rgb_batched, rendered_silhouette_batched = renderer.render(trasnformed_obj_dict, render_rgb=True)
        # cv2.imwrite('test.png',rendered_rgb_batched[0].detach().cpu().numpy()*255)

        rendered_rgb = crop_and_resize_by_mask(rendered_rgb_batched[0].detach().cpu().numpy()*255,  \
                                    rendered_silhouette_batched[0,...,-1].detach().cpu().numpy()*255, \
                                    124)
        rendered_sil = crop_and_resize_by_mask(rendered_silhouette_batched[0,...,-1].unsqueeze(-1).expand(-1,-1,3).detach().cpu().numpy()*255,  \
                                    rendered_silhouette_batched[0,...,-1].detach().cpu().numpy()*255, \
                                    124)
        
        rendered_rgbs.append(rendered_rgb)
        rendered_sils.append(rendered_sil)

    rendered_rgbs = np.stack(rendered_rgbs)
    rendered_sils = np.stack(rendered_sils)

    result_dict = {'rendered_rgbs':rendered_rgbs, 'rendered_sils':rendered_sils, 'rotmats':random_rotations, '6d':uniformly_sampled_6d_tensor}
    pickle.dump(result_dict, open(save_path,'wb'))
    return result_dict

if __name__ == "__main__":
    get_rendered_obj(obj_nm='box_pink', device=device)