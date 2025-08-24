import cv2
import numpy as np
import sys
import glob
from pathlib import Path
import json
import os

PROJECT_DIR = Path(__file__).absolute().parent.parent
sys.path.insert(0, str(PROJECT_DIR))

from paradex.pose_utils.vis_utils import read_frame, get_frame_number
from paradex.utils.file_io import find_latest_directory
import torch

home_path = os.path.expanduser("~")
shared_dir = os.path.join(home_path, "shared_data")
cam_param_dir = os.path.join(shared_dir, "cam_param")

class Scene:
    def __init__(self, scene_path:Path, rescale_factor=1., mask_dir_nm='mask_hq', mask_module=None, use_sam=False, dino_obj_nm=None, extracted = False, device='cuda'):
        # extrinsics_dict = json.load(open(scene_path/'cam_param'/'extrinsics.json'))
        # intrinsics_dict = json.load(open(scene_path/'cam_param'/'intrinsics.json'))
        self.camera_name = find_latest_directory(cam_param_dir)
        intrinsics_dict = json.load(open(os.path.join(cam_param_dir, self.camera_name, "intrinsics.json")))
        extrinsics_dict = json.load(open(os.path.join(cam_param_dir, self.camera_name, "extrinsics.json")))
        self.device=device

        self.scene_path = scene_path
        # self.obj_nm = str(self.scene_path).split("/")[-2]
        # self.obj_nm = "pringles"
        self.extracted = extracted
        if self.extracted:
            self.video_root_dir = scene_path/'video_extracted'
        else:
            self.video_root_dir = scene_path/'videos'
        self.extracted_img_root_dir = scene_path/'rgb_extracted'
        if mask_dir_nm is not None:
            self.mask_root_dir = scene_path/mask_dir_nm
        else:
            self.mask_root_dir = None

        if os.path.exists(scene_path/'xarm'/'qpos.npy'):
            hand_action = np.load(scene_path/'allegro'/'action.npy') # FX16
            hand_state = np.load(scene_path/'allegro'/'qpos.npy') # FX16
            arm_action = np.load(scene_path/'xarm'/'action.npy') # FX6
            arm_state = np.load(scene_path/'xarm'/'qpos.npy') # FX6

            self.robot_traj = np.concatenate([arm_state, hand_state], axis=1) # FX22
        else:
            self.robot_traj = None

        if os.path.exists(scene_path/'C2R.npy'):
            self.C2R = np.load(scene_path/'C2R.npy') # FX6
            self.R2C = np.linalg.inv(self.C2R)

        if os.path.exists(scene_path/'C2M.npy'):
            self.C2M = np.load(scene_path/'C2M.npy')
        else:
            self.C2M = None

        contact_path = scene_path/'contact'/'data.npy'
        self.contact = None
        if os.path.exists(contact_path):
            self.contact = np.load(contact_path) # TX15

        self.rescale_factor = rescale_factor
        height, width = 1536, 2048
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
            intrinsic_key = 'intrinsics_undistort' if 'intrinsics_undistort' in intrinsics_dict[cam_id] else 'original_intrinsics'
            intrinsic_np = np.array(intrinsics_dict[cam_id][intrinsic_key]).reshape(3,3)
            intrinsic_np[:2]*=rescale_factor
            dist = intrinsics_dict[cam_id]['dist_params'] if 'dist_params' in intrinsics_dict[cam_id] else  intrinsics_dict[cam_id]['dist_param'] 
            self.cam_params[cam_id] = {'extrinsic':extrinsic_np, 'intrinsic':intrinsic_np.reshape(3,3), 'dist':dist}
            self.cam2extr[cam_id] = extrinsic_np
            self.cam2intr[cam_id] = intrinsic_np
            self.cam2extr_t[cam_id] = torch.tensor(extrinsic_np, device=self.device).float()
            self.cam2intr_t[cam_id] = torch.tensor(intrinsic_np, device=self.device).float()

            cam_center = -np.linalg.inv(extrinsic_np[:3,:3])@extrinsic_np[:3,3]
            self.camera_centers[cam_id] = cam_center

            self.proj_matrix[cam_id] = self.cam_params[cam_id]['intrinsic']@self.cam_params[cam_id]['extrinsic']
            if height is None and width is None:
                if self.extracted:
                    if os.path.exists(self.video_root_dir/f'{cam_id}.mp4'):
                        frame = read_frame(self.video_root_dir/f'{cam_id}.mp4',0)
                        height, width = frame.shape[:2]
                    elif os.path.exists(scene_path/'images_undistorted_prev'):
                        for img_nm in os.listdir(scene_path/'images_undistorted_prev'):
                            frame = cv2.imread(str(scene_path/'images_undistorted_prev'/img_nm))
                            height, width  = frame.shape[:2]
                            break
                else:   
                    if os.path.exists(self.video_root_dir/f'{cam_id}.avi'):
                        frame = read_frame(self.video_root_dir/f'{cam_id}.avi',0)
                        height, width = frame.shape[:2]
                    elif os.path.exists(scene_path/'images_undistorted_prev'):
                        for img_nm in os.listdir(scene_path/'images_undistorted_prev'):
                            frame = cv2.imread(str(scene_path/'images_undistorted_prev'/img_nm))
                            height, width  = frame.shape[:2]
                            break
        self.cam_ids = [cam_id for cam_id in self.cam_params]
        self.ttl_frame_length = 0

        # first images
        self.height, self.width = int(height*rescale_factor), int(width*rescale_factor)

        for cam_id in self.cam_ids:
            if self.extracted:
                if os.path.exists(self.video_root_dir/f'{cam_id}.mp4'):
                    images_numb = get_frame_number(self.video_root_dir/f'{cam_id}.mp4')
                    if self.ttl_frame_length is None or self.ttl_frame_length<images_numb:
                        self.ttl_frame_length = images_numb
                else:
                    self.cam_ids.remove(cam_id) # remove if image not exist
            else:
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

        # make mask if the mask does not exist 
        if mask_module == 'yolo':
            from paradex.model.yolo_world_module import YOLO_MODULE
            self.mask_detector = YOLO_MODULE(categories=self.obj_nm if dino_obj_nm is None else dino_obj_nm)
            # NOTE: inference time
            # detections = self.mask_detector.process_img(rgb_img)
            # output_image = cv2.cvtColor(rgb_img, cv2.COLOR_RGB2BGR)
            # res_vis = yolo_module.annotate_image(output_image, detections, categories=yolo_module.categories, with_confidence=True)
            # imgs.append(res_vis)
            # detection_results[cam_id] = detections
            # canvas = yolo_module.annotate_image(rgb_img, detections, categories=yolo_module.categories, with_confidence=True)
            # canvas = putText(canvas, cam_id, color=(0,0, 255))
            # results_img.append(canvas)
        # elif mask_module == 'cnos': # mask_module == 'cnos'
        #     from paradex.pose_utils.cnos_module import CNOS_Detector, visualize
        #     self.mask_detector  = CNOS_Detector(self.obj_nm)
        #     # detections, mask_dict = cnos_detector.inference(img_arr) 
        #     # if len(detections)>0:
        #     #     rgb = Image.fromarray(img_arr) # load bgr image.
        #     #     concat_img = visualize(rgb, detections)
        #     #     if args.vis:
        #     #         concat_img.show()
        #     #     cv2.imwrite(os.path.join(mask_dir,f'{cam_id}.png'), mask_dict[1].astype(np.uint8)*255)
        #     #     ed_time = time.time()
        #     #     print(f'Detection for one frame takes {ed_time-st_time}')
        else:
            print("Mask module not imported to Scene")
            self.mask_detector = None


    def get_renderer(self, renderer_type='nvdiffrast'):
        if renderer_type == 'pytorch3d':
            from paradex.pose_utils.renderer_utils import Batched_RGB_Silhouette_Renderer

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
            from paradex.pose_utils.vis_utils_nvdiff import BatchRenderer
            # intrinsic should be same.
            renderer_dict = {}
            for cam_id in self.cam_ids:
                renderer_dict[cam_id] = BatchRenderer(opengl=False, cam_intrinsics = [self.cam2intr[cam_id]], cam_extrinsics=[self.cam2extr[cam_id]], \
                                                  width=self.width, height=self.height, device=self.device)
                
            self.renderer_dict = renderer_dict
            return renderer_dict
        

    def get_batched_renderer(self, tg_cam_list, renderer_type='nvdiffrast'):
        
        if renderer_type == 'pytorch3d':
            from paradex.pose_utils.renderer_utils import Batched_RGB_Silhouette_Renderer

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
            from paradex.pose_utils.vis_utils_nvdiff import BatchRenderer
            # intrinsic should be same.
            intrinsics = []
            extrinsics = []
            for cam_id in tg_cam_list:
                intrinsics.append(self.cam2intr[cam_id])
                extrinsics.append(self.cam2extr[cam_id])

            self.batched_renderer = BatchRenderer(opengl=False, cam_intrinsics = intrinsics, cam_extrinsics=extrinsics, width=self.width, height=self.height, device=self.device)

        return self.batched_renderer
    
    def get_image(self, cam_id, fidx, image_dir=None, rgb=False):
        '''
            It return images in bgr
        '''
        if image_dir is not None:
            if os.path.exists(os.path.join(image_dir,f'{cam_id}.png')):
                image_np = cv2.imread(os.path.join(image_dir, f'{cam_id}.png')) # bgr
                image_np = (image_np *255).astype(np.uint8) if image_np.dtype == np.float64  else image_np.astype(np.uint8)
                image_np = cv2.resize(image_np, (self.width, self.height))
                if rgb:
                    image_np = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)
                return image_np
            else:
                return np.zeros((self.height, self.width, 3))
        # print(self.ttl_frame_length)
        assert fidx<self.ttl_frame_length, f'{fidx} not in the range'
        if self.extracted:
            video_path = self.video_root_dir/f'{cam_id}.mp4'
        else:
            video_path = self.video_root_dir/f'{cam_id}.avi'
        if os.path.exists(video_path):
            image_np = read_frame(video_path, fidx) # bgr
            image_np = cv2.resize(image_np, (self.width, self.height))
            if rgb:
                image_np = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)
        else:
            image_np = np.zeros((self.height, self.width, 3))
        return image_np
    
    def get_extracted_image(self, cam_id, fidx, image_dir=None, empty=False, rgb=False):
        '''
            It return images in bgr
        '''
        if image_dir is not None:
            if os.path.exists(image_dir/f'{cam_id}.jpg'):
                image_np = cv2.imread(str(image_dir/f'{cam_id}.jpg')) # bgr
                image_np = cv2.resize(image_np, (self.width, self.height))
                if rgb:
                    image_np = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)
                return image_np
            else:
                return np.zeros((self.height, self.width, 3))
        # print(self.ttl_frame_length)
        assert fidx<self.ttl_frame_length, f'{fidx} not in the range'
        image_path = self.video_root_dir/ str(cam_id) / f"{fidx + 1:05d}.jpg"
        if os.path.exists(image_path): #and (empty is False):
            image_np = cv2.imread(image_path)
            image_np = cv2.resize(image_np, (self.width, self.height))
            image_np = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)

        else:
            image_np = np.ones((self.height, self.width, 3)) * 255
        return image_np
    

    def get_mask(self, cam_id, fidx=0, image_dir = None, mask_module = "yolo"):
        mask_path = str(self.mask_root_dir/('%05d'%(fidx))/('%s.png'%(cam_id)))
        print(mask_path)
        if os.path.exists(mask_path):
            mask = cv2.resize(cv2.imread(mask_path), (self.width, self.height))
        else:
            if self.mask_detector is None:
                print("Mask Detector is not initialized")
                return np.zeros((self.height, self.width, 3))
            if mask_module == "yolo":
                print(image_dir, cam_id)
                bgr_img = self.get_image(cam_id, fidx, image_dir)
                # rgb_img = cv2.cvtColor(rgb_img, cv2.COLOR_BGR2RGB) # YULO-World need BGR image. 
                detections = self.mask_detector.process_img(bgr_img)
                if len(detections.confidence) > 0:
                    detections.confidence.item()
                    mask_org = detections.mask[0].astype(np.uint8)
                    mask = np.tile(mask_org[...,np.newaxis],(1,1,3))
                    os.makedirs(Path(mask_path).parent, exist_ok=True)
                    cv2.imwrite(mask_path, mask*255)
                else:
                    mask = np.zeros((self.height, self.width, 3))
            elif mask_module == "cnos":
                bgr_img = self.get_image(cam_id, fidx)
                detections, mask_dict = self.mask_detector.inference(bgr_img) 

                if len(detections)>0:
                    mask_org = mask_dict[1].astype(np.uint8)
                    mask = np.tile(mask_org[...,np.newaxis],(1,1,3))
                    os.makedirs(Path(mask_path).parent, exist_ok=True)
                    cv2.imwrite(mask_path, mask*255)
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
