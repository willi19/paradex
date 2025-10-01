import sys, os
from pathlib import Path
PROJECT_PATH = Path(__file__).parent.parent
sys.path.append(str(PROJECT_PATH))
import numpy as np
import pickle
import cv2
from copy import deepcopy

import torch
from paradex.object_detection.obj_utils.scene import Scene
from paradex.object_detection.obj_utils.io import get_optimal_T
from paradex.object_detection.obj_utils.vis_utils import parse_objectmesh_objdict, make_grid_image_np, overlay_mask
# from utils.geometry import project_3d_to_2d_tensor, project_3d_to_2d
# Initialize the matcher with default settings

device = 'cuda'
class Template:
    '''
        Template is a set of images and matched 6D pose of object 
    '''
    def __init__(self, root_dir:str, obj_name=None, render_template_path='./check_template.jpeg', render_template=False):
        assert root_dir is not None and os.path.exists(root_dir), f'Should check directory {root_dir}'

        if obj_name is None:
            obj_name = root_dir.split("/")[-2] 

        template_path = str(Path(root_dir)/'initial_template.pkl')

        # pickle.dump({'img2face':self.img2face, 'img2point3d':self.img2point3d}, open(template_path, 'wb'))
        obj_dict = parse_objectmesh_objdict(obj_name, min_vertex_num=1000, \
                                            remove_uv=True, renderer_type='nvdiffrast', device=device)
        # scale mesh
        if not obj_dict['canonicalized']:
            obj_scale = obj_optim_output['scale'].detach().cpu().numpy().item(0)
            obj_dict['verts']*=obj_scale
        else:
            obj_scale = 1.0
        self.obj_dict = obj_dict

        if os.path.exists(template_path):
            tmp_dict = pickle.load(open(template_path, 'rb'))
            self.img2face = tmp_dict['img2face']
            self.img2point3d = tmp_dict['img2point3d']
            obj_optim_output = tmp_dict['obj_optim_output']
            self.img_template = tmp_dict['img_template']
            self.mask_template = tmp_dict['mask_template']
        
        else:
            self.scene = Scene(scene_path=Path(root_dir), rescale_factor=0.5, mask_dir_nm=f'mask_hq/{obj_name}')
            self.img_template = {cam_id : self.scene.get_image(cam_id, 0) for cam_id in self.scene.cam_ids}
            # self.mask_template = {cam_id : self.scene.get_mask(cam_id, 0, fill_value=255.0) for cam_id in self.scene.cam_ids} # In mask checking
            # cv2.imwrite('test.png', make_grid_image_np(np.array([self.mask_template[cam_id]*255 for cam_id in self.mask_template]), 4,6))
            # self.cam2extr = self.scene.cam2extr
            # self.cam2intr = self.scene.cam2intr
            

            obj_initial_status = get_optimal_T(root_dir) # find optimal T in first frame. 
            self.obj_initial_T = None
            if obj_initial_status is not None:
                obj_optim_output = pickle.load(open(obj_initial_status,'rb'))
                for key in obj_optim_output:
                    if torch.is_tensor(obj_optim_output[key]):
                        obj_optim_output[key] = obj_optim_output[key].to(device)

            self.render_template(self.obj_dict, obj_optim_output, render_template_path)

            obj_T = np.eye(4)
            obj_T[:3,:3] = obj_optim_output['R'].detach().cpu().numpy()
            obj_T[:3,3] = obj_optim_output['t'].detach().cpu().numpy()
            self.obj_initial_T = obj_T
            print(f'Initial object R:{obj_T[:3,:3]}, obj_t {obj_T[:3,3]}')

            self.init_objtemplate(deepcopy(obj_dict), obj_optim_output, template_path)
            debug=True
            if debug:
                import matplotlib.cm as cm
                from matplotlib.colors import Normalize
                cmap = cm.get_cmap('viridis')
                imgs = []
                for cam_id in self.img2face:
                    canvas = np.zeros((self.img2face[cam_id].shape[0], self.img2face[cam_id].shape[1],3))
                    for r in range(self.img2face[cam_id].shape[0]):
                        for c in range(self.img2face[cam_id].shape[1]):
                            if self.img2face[cam_id][r][c]>=0:
                                face_float = self.img2face[cam_id][r][c]/len(self.obj_dict['faces'])
                                canvas[r][c]=cmap(face_float)[:3]
                    imgs.append(canvas*255)
                cv2.imwrite(self.scene.scene_path/'template_faces.png', make_grid_image_np(np.stack(imgs), 4, 6))

            if render_template:
                self.render_template(self.obj_dict, obj_optim_output, render_template_path)

    def render_template(self, obj_dict, obj_optim_output, render_template_path):
        self.scene.get_batched_renderer(self.scene.cam_ids)
        # Transform object
        transformed_obj = deepcopy(obj_dict)
        org_scaled_verts = transformed_obj['verts'].detach()
        transformed_verts = torch.einsum('mn, bjn -> bjm', obj_optim_output['R'], (org_scaled_verts)) + obj_optim_output['t']
        # transformed_obj = deepcopy(obj_dict)
        transformed_obj['verts'] = transformed_verts

        batch_rendered = self.scene.batch_render(transformed_obj)

        rendered_rgb, rendered_sil = batch_rendered
        rendered_sil = rendered_sil.squeeze()

        imgs = []
        # visualize on image
        for cidx, cam_id in enumerate(self.scene.cam_ids):
            bgr_img = self.scene.get_image(cam_id, fidx=0)
    
            mask=rendered_sil[cidx].detach().cpu().numpy()
            overlaid = overlay_mask(bgr_img, mask=(mask>0))
            imgs.append(overlaid)

        cv2.imwrite(render_template_path, cv2.cvtColor(make_grid_image_np(np.stack(imgs), 4, 6),cv2.COLOR_BGR2RGB))
    

    def init_objtemplate(self, obj_dict, obj_optim_output, template_path):
        self.scene.get_batched_renderer(self.scene.cam_ids)
        # Transform object
        transformed_obj = deepcopy(obj_dict)
        org_scaled_verts = transformed_obj['verts'].detach()
        transformed_verts = torch.einsum('mn, bjn -> bjm', obj_optim_output['R'], (org_scaled_verts)) + obj_optim_output['t']
        # transformed_obj = deepcopy(obj_dict)
        transformed_obj['verts'] = transformed_verts

        pix2face = self.scene.batch_rasterize(transformed_obj).cpu().numpy() # CAM_NUMB, HEIGHT, WIDTH, 1 (-1 if not rendered, number of face index)

        # make 2d to 3d coorespondence from template to 3d points 
        self.img2face = {}
        self.img2point3d = {}
        self.mask_template = {}
        for cidx, cam_id in enumerate(self.scene.cam_ids):
            self.img2face[cam_id] = pix2face[cidx]
            self.img2point3d[cam_id] = np.zeros((self.img2face[cam_id].shape[0],self.img2face[cam_id].shape[1],3))
            # array uint8 
            tmp_mask = (self.img2face[cam_id]>0).astype(np.uint8)*255
            self.mask_template[cam_id] = np.tile(tmp_mask[...,np.newaxis],(1,1,3))
            ys, xs = np.where(self.img2face[cam_id]!=-1)
            for x,y in zip(xs,ys):
                vertice_numbs = list(obj_dict['faces'][int(self.img2face[cam_id][y,x])].detach().cpu().numpy())
                pos3d = torch.mean(obj_dict['verts'][0, vertice_numbs], axis=0)
                # for vidx in vertice_numbs:
                self.img2point3d[cam_id][y,x] = pos3d.detach().cpu().numpy()
                
        # obj_optim_output, img_template, mask_template
        pickle.dump({'img2face':self.img2face, 'img2point3d':self.img2point3d,\
                    'obj_optim_output':obj_optim_output, 'img_template':self.img_template,\
                    'mask_template':self.mask_template}, open(template_path, 'wb'))
        
        # cv2.imwrite('test.png', make_grid_image_np(np.array([self.mask_template[cam_id] for cam_id in self.mask_template]), 4,6))
