import cv2
import numpy as np
import sys
from pathlib import Path
import json
import os
from dex_robot.renderer.robot_module import Robot_Module

import torch
device = torch.device("cuda:0")


def make_grid(img_list, vis_cam_list, org_height, org_width):
    if torch.is_tensor(img_list[0]):
        images_np = torch.stack(img_list).squeeze().detach().cpu().numpy()
    else:
        images_np = np.stack(img_list)
    c_start = int(np.sqrt(len(vis_cam_list)))
    for closest_factor in range(c_start,0,-1):
        if len(vis_cam_list)%closest_factor==0:
            break
    grid_h, grid_w = closest_factor, int(len(vis_cam_list)/closest_factor)
    grid = images_np.reshape(grid_h, grid_w, org_height, org_width, 3).swapaxes(1, 2).reshape(grid_h * org_height, grid_w * org_width, 3)
        
    return grid


class Scene:
    def __init__(self, root_path:Path, rescale_factor=1):
        extrinsics_dict = json.load(open(root_path/'cam_param'/'extrinsics.json'))
        intrinsics_dict = json.load(open(root_path/'cam_param'/'intrinsics.json'))


        hand_action = np.load(root_path/'hand'/'action.npy') # FX16
        hand_state = np.load(root_path/'hand'/'state.npy') # FX16
        arm_action = np.load(root_path/'arm'/'action.npy') # FX6
        arm_state = np.load(root_path/'arm'/'state.npy') # FX6

        self.robot_traj = np.concatenate([arm_state, hand_state], axis=1) # FX22

        self.C2R = np.load(root_path/'C2R.npy') # FX6
        self.R2C = np.linalg.inv(self.C2R)

        self.proj_matrix = {}
        self.cam_params = {}
        for cam_id in extrinsics_dict:
            extrinsic_np = np.array(extrinsics_dict[cam_id]) # 3X4
            intrinsic_np = np.array(intrinsics_dict[cam_id]['Intrinsics'])*rescale_factor
            height, width = int(intrinsics_dict[cam_id]['height']*rescale_factor), int(intrinsics_dict[cam_id]['width']*rescale_factor)
            dist = intrinsics_dict[cam_id]['dist_param']

            self.cam_params[cam_id] = {'extrinsic':extrinsic_np, 'intrinsic':intrinsic_np.reshape(3,3), 'size':(height, width)}
            self.proj_matrix[cam_id] = self.cam_params[cam_id]['intrinsic']@self.cam_params[cam_id]['extrinsic']

        self.cam_ids = [cam_id for cam_id in self.cam_params]

        self.video_root_dir = root_path/'video'
        self.height, self.width = 1536, 2048
        
        self.renderer_dict = None
        self.robot_module = Robot_Module(state=self.robot_traj)


    def get_renderer(intrinsic, extrinsic, n_splits=8):
        from renderer_utils import Batched_RGB_Silhouette_Renderer

        cam_id_list = []
        
        extrinsic_list = []
        intrinsic_list = []

        for cam_id in intrinsic.keys():
            intrinsic_list.append(np.array(intrinsic[cam_id]['Intrinsics']).reshape((3,3)))
            extrinsic_list.append(extrinsic[cam_id])
            cam_id_list.append(cam_id)

        intrinsic_list = np.array(intrinsic_list)
        extrinsic_list = np.array(extrinsic_list)

        intrinsic_list = torch.tensor(intrinsic_list, dtype=torch.float32, device=device)
        extrinsic_list = torch.tensor(extrinsic_list, dtype=torch.float32, device=device)
        img_sizes = torch.tensor([[1536, 2048]], device=device)

        total = len(cam_id_list)
        split_size = (total + n_splits - 1) // n_splits  # 나머지가 있어도 분할되도록

        renderers = []
        cam_id_chunks = []

        for i in range(n_splits):
            start = i * split_size
            end = min((i + 1) * split_size, total)

            renderer = Batched_RGB_Silhouette_Renderer(
                extrinsic_list[start:end],
                intrinsic_list[start:end],
                img_sizes,
                device
            )
            renderers.append(renderer)
            cam_id_chunks.append(cam_id_list[start:end])

        return renderers, cam_id_chunks

    def get_video(self, cam_id):
        video_path = self.video_root_dir/f"{cam_id}.mp4"
        if os.path.exists(video_path):
            video = cv2.VideoCapture(str(video_path))
        else:
            video = None
        return video
    
    # def get_image(self, cam_id, fidx):
    #     assert fidx<self.ttl_frame_length, f'{fidx} not in the range'
    #     img_path = self.image_root_dir/cam_id/('%05d.jpeg'%(fidx))
    #     if os.path.exists(img_path):
    #         image = cv2.imread(img_path)
    #     else:
    #         image = np.zeros((self.height, self.width, 3))
    #     return image


    # def get_images(self, fidx):
    #     assert fidx<self.ttl_frame_length, f'{fidx} not in the range'
    #     image_dictionary = {}
    #     for cam_id in self.cam_params:
    #         image_dictionary[cam_id] = self.get_image(cam_id, fidx)
    #     return image_dictionary
    
    def overlay(self):
        fidx = 0
        video_dict = {cam_id:self.get_video(cam_id) for cam_id in self.cam_ids}
        overlay_video_dict = {cam_id:cv2.VideoWriter(str(self.video_root_dir/f"{cam_id}_overlay.mp4"), cv2.VideoWriter_fourcc(*'mp4v'), 30, (self.width, self.height)) for cam_id in self.cam_ids}

        while True:
            video_end = True
            frame_dict = {}
            for cam_id in self.cam_ids:
                if video_dict[cam_id] is not None:
                    ret, frame = video_dict[cam_id].read()
                    if not ret:
                        video_dict[cam_id] = None
                    else:
                        frame_dict[cam_id] = frame
                        video_end = False
            if video_end:
                break

    def render(self, cam_id, mesh):
        rendered_rgb, rendered_silhouette = self.renderer_dict[cam_id].render(mesh)

        return rendered_rgb, rendered_silhouette