import cv2
import numpy as np
import sys
import glob
from pathlib import Path
import json
import os

PROJECT_DIR = Path(__file__).absolute().parent.parent
sys.path.insert(0, str(PROJECT_DIR))

import torch

def read_frame(video_path, frame_number=0):
    supposed_extracted_path = os.path.join(str(video_path).replace('video','rgb_extracted').split(".")[0],'%05d.jpeg'%(frame_number))
    if os.path.exists(supposed_extracted_path):
        return cv2.imread(supposed_extracted_path)
    else:

        import imageio.v3 as iio
        # 비디오 읽기img_np        
        # 총 프레임 수 확인
        total_frames = iio.improps(video_path).n_images
        
        if frame_number >= total_frames:
            print(f"Error: Requested frame {frame_number} is beyond total frames.")
            return None
        
        # n번째 프레임 읽기
        frame = iio.imread(video_path, index=frame_number)
        print(f"Total frames in video: {total_frames}")

        os.makedirs(str(video_path).replace('video','rgb_extracted').split(".")[0], exist_ok=True)
        cv2.imwrite(supposed_extracted_path, frame)

        return frame

def get_frame_number(video_path):
    cap = cv2.VideoCapture(str(video_path))
    cv2.VideoCapture


    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")
    
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return frame_count


class Scene:
    def __init__(self, root_path:Path, rescale_factor=1., mask_dir_nm='mask_hq', device='cuda'):
        extrinsics_dict = json.load(open(root_path/'cam_param'/'extrinsics.json'))
        intrinsics_dict = json.load(open(root_path/'cam_param'/'intrinsics.json'))

        self.device=device

        self.video_root_dir = root_path/'video'
        self.extracted_img_root_dir = root_path/'rgb_extracted'
        if mask_dir_nm is not None:
            self.mask_root_dir = root_path/mask_dir_nm
        else:
            self.mask_root_dir = None

        hand_action = np.load(root_path/'hand'/'action.npy') # FX16
        hand_state = np.load(root_path/'hand'/'state.npy') # FX16
        arm_action = np.load(root_path/'arm'/'action.npy') # FX6
        arm_state = np.load(root_path/'arm'/'state.npy') # FX6

        self.robot_traj = np.concatenate([arm_state, hand_state], axis=1) # FX22

        self.C2R = np.load(root_path/'C2R.npy') # FX6
        self.R2C = np.linalg.inv(self.C2R)

        if os.path.exists(root_path/'C2M.npy'):
            self.C2M = np.load(root_path/'C2M.npy')
        else:
            self.C2M = None

        contact_path = root_path/'contact'/'data.npy'
        self.contact = None
        if os.path.exists(contact_path):
            self.contact = np.load(contact_path) # TX15

        self.rescale_factor = rescale_factor
        height, width = None, None
        self.proj_matrix = {}
        self.cam_params = {}

        self.cam2extr = {}
        self.cam2intr = {}
        self.cam2extr_t = {}
        self.cam2intr_t = {}

        self.camera_centers = {}

        # Setting Camera Parameters
        for cam_id in extrinsics_dict:
            extrinsic_np = np.array(extrinsics_dict[cam_id]) # 3X4
            intrinsic_np = np.array(intrinsics_dict[cam_id]['Intrinsics']).reshape(3,3)
            intrinsic_np[:2]*=rescale_factor
            dist = intrinsics_dict[cam_id]['dist_param']
            self.cam_params[cam_id] = {'extrinsic':extrinsic_np, 'intrinsic':intrinsic_np.reshape(3,3), 'dist':dist}
            self.cam2extr[cam_id] = extrinsic_np
            self.cam2intr[cam_id] = intrinsic_np
            self.cam2extr_t[cam_id] = torch.tensor(extrinsic_np, device=self.device).float()
            self.cam2intr_t[cam_id] = torch.tensor(intrinsic_np, device=self.device).float()

            cam_center = -np.linalg.inv(extrinsic_np[:3,:3])@extrinsic_np[:3,3]
            self.camera_centers[cam_id] = cam_center

            self.proj_matrix[cam_id] = self.cam_params[cam_id]['intrinsic']@self.cam_params[cam_id]['extrinsic']
            if height is None and width is None:
                if os.path.exists(self.video_root_dir/f'{cam_id}.avi'):
                    frame = read_frame(self.video_root_dir/f'{cam_id}.avi',0)
                    height, width = frame.shape[:2]

        self.cam_ids = [cam_id for cam_id in self.cam_params]
        self.ttl_frame_length = 0

        # first images
        self.height, self.width = int(height*rescale_factor), int(width*rescale_factor)

        for cam_id in self.cam_ids:
            if os.path.exists(self.video_root_dir/f'{cam_id}.avi'):
                images_numb = get_frame_number(self.video_root_dir/f'{cam_id}.avi')
                if self.ttl_frame_length is None or self.ttl_frame_length<images_numb:
                    self.ttl_frame_length = images_numb
            else:
                self.cam_ids.remove(cam_id) # remove if image not exist

        # self.cam_ids = self.cam_ids[:4]

        self.renderer_dict = None
        self.batched_renderer = None
        self.stacked_extrinsic = None
        self.stacked_intrinsic = None


    def get_renderer(self, renderer_type='nvdiffrast'):
        if renderer_type == 'pytorch3d':
            from utils.renderer_utils import Batched_RGB_Silhouette_Renderer

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
            from utils.vis_utils_nvdiff import BatchRenderer
            # intrinsic should be same.
            renderer_dict = {}
            for cam_id in self.cam_ids:
                renderer_dict[cam_id] = BatchRenderer(opengl=False, cam_intrinsics = [self.cam2intr[cam_id]], cam_extrinsics=[self.cam2extr[cam_id]], \
                                                  width=self.width, height=self.height, device=self.device)
                
            self.renderer_dict = renderer_dict
            return renderer_dict
        

    def get_batched_renderer(self, tg_cam_list, renderer_type='nvdiffrast'):
        
        if renderer_type == 'pytorch3d':
            from utils.renderer_utils import Batched_RGB_Silhouette_Renderer

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
            from utils.vis_utils_nvdiff import BatchRenderer
            # intrinsic should be same.
            intrinsics = []
            extrinsics = []
            for cam_id in tg_cam_list:
                intrinsics.append(self.cam2intr[cam_id])
                extrinsics.append(self.cam2extr[cam_id])

            self.batched_renderer = BatchRenderer(opengl=False, cam_intrinsics = intrinsics, cam_extrinsics=extrinsics, width=self.width, height=self.height, device=self.device)

        return self.batched_renderer
    
    def get_image(self, cam_id, fidx):
        assert fidx<self.ttl_frame_length, f'{fidx} not in the range'
        video_path = self.video_root_dir/f'{cam_id}.avi'
        if os.path.exists(video_path):
            image_np = read_frame(video_path, fidx)
            image_np = cv2.cvtColor(cv2.resize(image_np, (self.width, self.height)), cv2.COLOR_BGR2RGB)
        else:
            image_np = np.zeros((self.height, self.width, 3))
        return image_np
    

    def get_mask(self, cam_id, fidx=0):
        mask_path = str(self.mask_root_dir/('%05d'%(fidx))/('%s.png'%(cam_id)))
        if os.path.exists(mask_path):
            mask = cv2.resize(cv2.imread(mask_path), (self.width, self.height))
        else:
            mask = np.zeros((self.height, self.width, 3))
        return mask


    def get_images(self, fidx):
        assert fidx<self.ttl_frame_length, f'{fidx} not in the range'
        image_dictionary = {}
        for cam_id in self.cam_params:
            image_dictionary[cam_id] = self.get_image(cam_id, fidx)
        return image_dictionary
    

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
    

    def project_rendered_to_img(self, cam_id, fidx, rendered_rgb, rendered_silhouette, alpha=0.5):

        image = cv2.resize(self.get_image(cam_id, fidx), (self.width, self.height))

        rendered_rgb_arr = (rendered_rgb[0].detach().cpu().numpy()*255).astype(np.uint8)
        weighted_rgb_arr = cv2.addWeighted(image, 1 - alpha, rendered_rgb_arr, alpha, 0)

        mask = rendered_silhouette[0].detach().cpu().numpy()
        colored_mask = np.zeros_like(rendered_rgb_arr, dtype=np.uint8)
        colored_mask[mask!=0,0] = 255
        colored_mask = cv2.addWeighted(image, 1 - alpha, colored_mask, alpha, 0)
        weighted_mask_arr = image.copy()
        weighted_mask_arr[mask!=0] = colored_mask[mask!=0]

        return weighted_rgb_arr, weighted_mask_arr


    def get_projected_img(self, cam_id, fidx, mesh, alpha=0.5):
        rendered_rgb, rendered_silhouette = self.renderer_dict[cam_id].render(mesh)
        # height, width = self.cam_params[cam_id]['size']

        return self.project_rendered_to_img(cam_id, fidx, rendered_rgb, rendered_silhouette, alpha=alpha)
    

    def batch_transform2camspace(self, vertices):
        # vertices BXNX3 ndarray
        # self.stacked_extrinsic: MX3X4
        B, J, _ = vertices.shape
        ones = np.ones((B, J, 1))
        vertices_homo = np.concatenate([vertices, ones], axis=-1)  #  (M, 3, 4), (B, J, 4), -> (M, B, J, 3)
        vertices_homo_cam = np.einsum('mij, bnj -> mbni', self.stacked_extrinsic, vertices_homo)
        
        return vertices_homo_cam
