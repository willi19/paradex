import os

import numpy as np
import cv2
import json
from pathlib import Path
import copy
import sys
import time
import numpy as np
import torch
from torch.nn import functional as F
PROJECT_DIR = Path(__file__).absolute().parent.parent
print(f'PROJECT_DIR {PROJECT_DIR}')
sys.path.insert(0, str(PROJECT_DIR))
from scipy.spatial.transform import Rotation as R

from paradex.object_detection.optimize_module import get_transformed_obj
from paradex.object_detection.obj_utils.geometry import matrix_to_rotation_6d, matrix_to_rotation_6d
from paradex.object_detection.obj_utils.vis_utils import parse_objectmesh_objdict, \
                            crop_and_resize_by_mask, crop_and_resize_by_mask_tensor

device = torch.device("cuda:0")

nas_path = Path(os.environ['NAS_PATH'])
mesh_template_dir = PROJECT_DIR/'mesh_rendered'
if not os.path.exists(mesh_template_dir):
    os.makedirs(mesh_template_dir)

def get_rendered_obj(obj_nm, N_rotation=1000, image_size=124, renderer_type='nvdiff', dino_module=None, device='cuda'):
    import pickle
    st_time = time.time()

    save_path = mesh_template_dir/f'{obj_nm}.pickle'
    if os.path.exists(save_path):
        rendered_dict = pickle.load(open(save_path, 'rb'))
        for key in ['rendered_rgbs','rendered_sils']: # resize image
            if rendered_dict[key].shape[1]!=image_size:
                rendered_dict[key] = \
                    F.interpolate(torch.tensor(rendered_dict[key]).permute(0,3,1,2), \
                                size=image_size).permute(0,2,3,1).detach().cpu().numpy()
    else:
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
        from paradex.object_detection.obj_utils.vis_utils_nvdiff import BatchRenderer
        width, height = 2048, 1536
        renderer = BatchRenderer(opengl=False, cam_intrinsics = [intrinsic], cam_extrinsics=[np.eye(4)[:3]], \
                                                width=width, height=height, device=device)
        obj_dict = parse_objectmesh_objdict(obj_nm, renderer_type=renderer_type, device=device, simplify=False)
        scaled = obj_dict['scaled']
        assert scaled, 'You should use scaled mesh'
        # Make Candidates
        # uniformly_sampled_rotmat_tensor = torch.tensor(uniform_so3_fibonacci(N).as_matrix(), device=device).float().
        import scipy
        random_rotations = torch.tensor(scipy.spatial.transform.Rotation.random(N_rotation).as_matrix(), device=device).float()
        uniformly_sampled_6d_tensor = matrix_to_rotation_6d(random_rotations)  # in quaternions

        rendered_rgbs = []
        rendered_sils = []
        for ridx in range(0, N_rotation):
            trasnformed_obj_dict = get_transformed_obj(obj_dict, 1.0, uniformly_sampled_6d_tensor[ridx],\
                                                        torch.tensor([0,0,1.5], device=device).float(), )
            rendered_rgb_batched, rendered_silhouette_batched = renderer.render(trasnformed_obj_dict, render_rgb=True)
            # cv2.imwrite('test.png',rendered_rgb_batched[0].detach().cpu().numpy()*255)

            rendered_rgb = crop_and_resize_by_mask(rendered_rgb_batched[0].detach().cpu().numpy(),  \
                                        rendered_silhouette_batched[0,...,-1].detach().cpu().numpy(), \
                                        image_size)
            rendered_sil = crop_and_resize_by_mask(rendered_silhouette_batched[0,...,-1].unsqueeze(-1).expand(-1,-1,3).detach().cpu().numpy(),  \
                                        rendered_silhouette_batched[0,...,-1].detach().cpu().numpy(), \
                                        image_size)
            
            rendered_rgb[rendered_sil<=1] = 0 # make black not rendered_area
            rendered_rgbs.append(rendered_rgb)
            rendered_sils.append(rendered_sil)

        rendered_rgbs = np.stack(rendered_rgbs)
        rendered_sils = np.stack(rendered_sils)
        rendered_dict = {'rendered_rgbs':rendered_rgbs, 'rendered_sils':rendered_sils, \
                    'rotmats':random_rotations, '6d':uniformly_sampled_6d_tensor}
        
        pickle.dump(rendered_dict, open(save_path,'wb'))

    if dino_module is not None:
        feature_save_path = mesh_template_dir/f'{obj_nm}_feature.pickle'
        if os.path.exists(feature_save_path):
            prerendered_feature = pickle.load(open(feature_save_path,'rb'))
            rendered_dict['prerendered_features'] = prerendered_feature
        else:
            prerendered_features = {key:[] for key in ['x_norm_clstoken','x_norm_patchtokens']}
            for rendered_rgb in rendered_dict['rendered_rgbs']:
                prerendered_feature = dino_module.extract_feature(torch.tensor(rendered_rgb,device=device)/255)
                for key in ['x_norm_clstoken','x_norm_patchtokens']:
                    prerendered_features[key].append(prerendered_feature[key].detach().cpu())

            if len(prerendered_features['x_norm_clstoken'])>0:
                for key in ['x_norm_clstoken','x_norm_patchtokens']:
                    prerendered_features[key] = torch.stack(prerendered_features[key])
            else:
                prerendered_features = None
            rendered_dict['prerendered_features'] = prerendered_features
            pickle.dump(rendered_dict['prerendered_features'], open(str(feature_save_path),'wb'))

    ed_time = time.time()
    print(f"Loading rendered. {ed_time-st_time} sec")
    return rendered_dict

import scipy
mesh_template_nview_dir = PROJECT_DIR/'mesh_nview_rendered'
os.makedirs(mesh_template_nview_dir, exist_ok=True)
def get_nview_rendered_obj(scene, obj_dict, initial_translate, N_rotation=1000, default_scale=1.0, image_size=124, rerender=False, dino_module=None):
    st_time = time.time()

    import pickle
    save_path = mesh_template_nview_dir/f'{scene.obj_nm}.pt'
    '''
        Load Rendered Templates and Rotation Candidates
    '''

    if os.path.exists(save_path) and not rerender:
        rendered_dict = torch.load(save_path)
        for key in ['rendered_rgbs_all_Rs','rendered_sils_all_Rs']: # check the dimension.
            if rendered_dict[key][0].shape[1]!=image_size:
                for ridx, rgb_imgs in enumerate(rendered_dict[key]): 
                    rendered_dict[key][ridx] = \
                        F.interpolate(torch.tensor(rgb_imgs).permute(0,3,1,2), \
                                    size=image_size).permute(0,2,3,1).detach().cpu().numpy()
    else:
        # prerendering.
        randomly_sampled_rotmat_tensor = torch.tensor(scipy.spatial.transform.Rotation.random(N_rotation).as_matrix(), device=device).float()
        randomly_sampled_6d_tensor = matrix_to_rotation_6d(randomly_sampled_rotmat_tensor)  # in quaternions

        rendered_rgbs_allRs = []
        rendered_sils_allRs = []
        for ridx in range(0, N_rotation):
            print(f"Making template {ridx}")
            trasnformed_obj_dict = get_transformed_obj(obj_dict, default_scale, randomly_sampled_6d_tensor[ridx],\
                                                        torch.tensor(initial_translate).to(device).float())
            rendered_rgb_batched, rendered_silhouette_batched = scene.batch_render(trasnformed_obj_dict)

            # Parse Rendered Image
            rendered_rgbs, rendered_sils = [], []
            for rendered_idx in range(rendered_rgb_batched.shape[0]):
                rendered_rgb_tensor = crop_and_resize_by_mask_tensor(rendered_rgb_batched[rendered_idx]*255, \
                                                                     rendered_silhouette_batched[rendered_idx,...,-1]*255, image_size)
                rendered_sil_tensor = crop_and_resize_by_mask_tensor(rendered_silhouette_batched[rendered_idx,...,-1].unsqueeze(-1).expand(-1,-1,3)*255, \
                                                                     rendered_silhouette_batched[rendered_idx,...,-1]*255, image_size)
                rendered_rgb_tensor[rendered_sil_tensor<=1]=0
                rendered_rgbs.append(rendered_rgb_tensor)
                rendered_sils.append(rendered_sil_tensor)
                
            rendered_rgbs = torch.stack(rendered_rgbs)
            rendered_sils = torch.stack(rendered_sils)

            rendered_rgbs_allRs.append(rendered_rgbs.detach().cpu().numpy())
            rendered_sils_allRs.append(rendered_sils.detach().cpu().numpy())

        rendered_dict = {'randomly_sampled_6d_tensor':randomly_sampled_6d_tensor,
                         'randomly_sampled_rotmat_tensor':randomly_sampled_rotmat_tensor,
                         'rendered_rgbs_all_Rs': rendered_rgbs_allRs,
                         'rendered_sils_all_Rs': rendered_sils_allRs}
        torch.save(rendered_dict, save_path)

    if dino_module is not None:
        feature_save_path = mesh_template_nview_dir/f'{scene.obj_nm}_feature.pt'
        if os.path.exists(feature_save_path):
            prerendered_feature = torch.load(feature_save_path)
            rendered_dict['prerendered_features'] = prerendered_feature
        else:
            rendered_dict['prerendered_features'] = {key:[ ] for key in ['x_norm_clstoken','x_norm_patchtokens']}
            for rendered_rgbs in rendered_dict['rendered_rgbs_all_Rs']:
                prerendered_feature = dino_module.extract_feature(torch.tensor(rendered_rgbs,device=device)/255)
                for key in ['x_norm_clstoken','x_norm_patchtokens']:
                    rendered_dict['prerendered_features'][key].append(prerendered_feature[key].detach().cpu())
            for key in ['x_norm_clstoken','x_norm_patchtokens']:
                    rendered_dict['prerendered_features'][key] = torch.stack(rendered_dict['prerendered_features'][key] )
            torch.save(rendered_dict['prerendered_features'], feature_save_path)
        
    
    ed_time = time.time()
    print(f"Loading N view rendered. {ed_time-st_time} sec")

    return rendered_dict


if __name__ == "__main__":
    get_rendered_obj(obj_nm='box_pink', device=device)