import cv2
import numpy as np
import sys
from pathlib import Path
import os

PROJECT_DIR = Path(__file__).absolute().parent.parent
sys.path.insert(0, str(PROJECT_DIR))

from paradex.utils.file_io import shared_dir, load_current_camparam
import torch

NAS_IMG_SAVEDIR = os.path.join(shared_dir, 'current_img')

class MultiCamScene:
    def __init__(self, rescale_factor=0.5, device='cuda'):
        intrinsic, extrinsic = load_current_camparam()
        
        # parsing intrinsic, extrinsic
        self.cam2extr = extrinsic
        self.cam2intr = {}
        self.cam2extr_t = {}
        self.cam2intr_t = {}
        self.proj_matrix = {}
        self.cam_params = {}
        
        self.device=device
        
        
        for cam_id in intrinsic:
            intrinsic[cam_id]['intrinsics_undistort'][:2]*=rescale_factor     
            self.cam2intr[cam_id] = intrinsic[cam_id]['intrinsics_undistort']
            
            self.cam2extr_t[cam_id] = torch.tensor(extrinsic[cam_id], device=self.device).float()
            self.cam2intr_t[cam_id] = torch.tensor(self.cam2intr[cam_id], device=self.device).float()
            
            self.proj_matrix[cam_id] = self.cam2intr[cam_id] @ self.cam2extr[cam_id]
            
            self.cam_params[cam_id] = {
                'intrinsic': self.cam2intr[cam_id],
                'extrinsic': self.cam2extr[cam_id],
                'dist':intrinsic[cam_id]['dist_params']
            }
            
        self.cam_ids = list(self.cam2intr.keys())
        self.height, self.width = int(intrinsic[cam_id]['height']*rescale_factor), \
                                    int(intrinsic[cam_id]['width']*rescale_factor)
                                    
    

    def get_renderer(self, renderer_type='nvdiffrast'):
        if renderer_type == 'pytorch3d':
            from paradex.object_detection.obj_utils.renderer_utils import Batched_RGB_Silhouette_Renderer

            renderer_dict = {}
            for cam_id in self.cam_ids:
                cam_extrinsic = torch.tensor([self.cam_params[cam_id]['extrinsic']], dtype=torch.float32, device=self.device)
                cam_intrinsic = torch.tensor([self.cam_params[cam_id]['intrinsic']], dtype=torch.float32, device=self.device)
                img_sizes = torch.tensor([[self.height, self.width]], device=self.device)
                
                renderer = Batched_RGB_Silhouette_Renderer(cam_extrinsic, cam_intrinsic, img_sizes, self.device)
                renderer_dict[cam_id] = renderer

            self.renderer_dict = renderer_dict

            return self.renderer_dict
        elif renderer_type == 'nvdiffrast':
            from paradex.object_detection.obj_utils.vis_utils_nvdiff import BatchRenderer
            # intrinsic should be same.
            renderer_dict = {}
            for cam_id in self.cam_ids:
                renderer_dict[cam_id] = BatchRenderer(opengl=False, cam_intrinsics = [self.cam2intr[cam_id]], cam_extrinsics=[self.cam2extr[cam_id]], \
                                                  width=self.width, height=self.height, device=self.device)
                
            self.renderer_dict = renderer_dict
            return renderer_dict
        

    def get_batched_renderer(self, tg_cam_list, renderer_type='nvdiffrast'):
        
        if renderer_type == 'pytorch3d':
            from paradex.object_detection.obj_utils.renderer_utils import Batched_RGB_Silhouette_Renderer

            stacked_extrinsic = []
            stacked_intrinsic = []
            stacked_imgsize = []

            for cam_id in tg_cam_list:
                stacked_extrinsic.append(self.cam_params[cam_id]['extrinsic'])
                stacked_intrinsic.append(self.cam_params[cam_id]['intrinsic'])
                stacked_imgsize.append(np.array([self.height, self.width])) # height width

            self.stacked_extrinsic = np.stack(stacked_extrinsic) # BX3X4
            self.stacked_intrinsic = np.stack(stacked_intrinsic)
            stacked_imgsize = np.stack(stacked_imgsize)

            self.batched_renderer = Batched_RGB_Silhouette_Renderer(torch.tensor(self.stacked_extrinsic, dtype=torch.float32, device=self.device), \
                                                            torch.tensor(self.stacked_intrinsic, dtype=torch.float32, device=self.device), \
                                                            torch.tensor(stacked_imgsize, dtype=torch.float32, device=self.device), \
                                                            device=self.device)
            
        elif renderer_type == 'nvdiffrast':
            from paradex.object_detection.obj_utils.vis_utils_nvdiff import BatchRenderer
            # intrinsic should be same.
            intrinsics = []
            extrinsics = []
            for cam_id in tg_cam_list:
                intrinsics.append(self.cam2intr[cam_id])
                extrinsics.append(self.cam2extr[cam_id])

            self.batched_renderer = BatchRenderer(opengl=False, cam_intrinsics = intrinsics, cam_extrinsics=extrinsics, width=self.width, height=self.height, device=self.device)

        return self.batched_renderer
    
    def get_image(self, cam_id, rgb=False):
        assert cam_id in self.cam_params, f'{cam_id} not in the camera list'
        if os.path.exists(os.path.join(NAS_IMG_SAVEDIR, f'{cam_id}.jpeg')):
            try:
            resized_img = cv2.resize(cv2.imread(os.path.join(NAS_IMG_SAVEDIR, f'{cam_id}.jpeg')), (self.width, self.height))
            if rgb:
                resized_img = cv2.cvtColor(resized_img, cv2.COLOR_BGR2RGB)
        else:
            print(f"[Warning] No image for {cam_id}! \
                check the path: {os.path.join(NAS_IMG_SAVEDIR, f'{cam_id}.jpeg')}")
            resized_img = np.zeros((self.height, self.width, 3), dtype=np.uint8)
                
        return resized_img


    def render(self, cam_id, obj_dict, render_rgb=True):
        rendered_rgb, rendered_silhouette = self.renderer_dict[cam_id].render(obj_dict, render_rgb=render_rgb)

        return rendered_rgb, rendered_silhouette
    
    def batch_render(self, obj_dict, render_rgb=True):
        
        rendered_rgb, rendered_silhouette = self.batched_renderer.render(obj_dict, render_rgb=render_rgb)
        return rendered_rgb, rendered_silhouette
    

    def get_batch_rendered_mask(self, obj_dict_list):
        return self.batched_renderer.render_id(obj_dict_list)

    def rasterize(self, cam_id, obj_dict):
        return self.renderer_dict[cam_id].get_fragments(obj_dict)
    
    def batch_rasterize(self, obj_dict):
        return self.batched_renderer.get_rendered_faces(obj_dict)
    