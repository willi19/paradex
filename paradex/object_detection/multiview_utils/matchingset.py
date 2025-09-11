import numpy as np
import pickle
import cv2
import time
from copy import deepcopy
import torch

import sys, os
from pathlib import Path
PROJECT_PATH = Path(__file__).parent.parent

from paradex.object_detection.obj_utils.geometry import project_3d_to_2d_tensor, project_3d_to_2d, rotation_6d_to_matrix, rotation_6d_to_matrix_np
from paradex.object_detection.multiview_utils.optimizer import optimize, optimize_ceres

class MatchItem:
    def __init__(self, cam_id, midx, matching_item, mask, initial_T, transformed_verts, proj_matrix):
        # {'count':pair_count, 'src_3d':src_3d_dict, 'tg_2d':tg_2d_dict, 'src_2d':org_2d_dict}
        self.cam_id = cam_id
        self.midx = midx
        self.count = matching_item['count'] # total point count (target image - DB)
        # self.src_3d = matching_item['src_3d']
        # self.tg_2d = matching_item['tg_2d']
        # self.src_2d = matching_item['src_2d']
        
        self.combined_src_3d = matching_item['combined_src_3d']
        self.combined_tg_2d = matching_item['combined_tg_2d']
        self.src_cam_ids = matching_item['src_arr_cam_ids']
        self.inliers = matching_item['inliers']==1
        self.inlier_count = np.sum(self.inliers)
        
        self.mask = mask
        # self.confidence = detection.confidence
        self.initial_T = initial_T.detach().cpu().numpy()
        self.transformed_verts = transformed_verts
        self.proj_matrix = proj_matrix

    def check_output_wdistance(self, output_T:np.ndarray):
        # Check the optimization output to current new item
        transformed = self.combined_src_3d[self.inliers] @ output_T[:3,:3].T + output_T[:3,3]  # (N,3)
        projected = project_3d_to_2d(transformed, self.proj_matrix[None]).squeeze()
        distance = np.mean(np.linalg.norm((projected - self.combined_tg_2d[self.inliers]),axis=1))

        return distance

def check_potision(item1:MatchItem, item2:MatchItem, thres=0.05): 
    dist = np.linalg.norm(item1.initial_T[:3,3]-item2.initial_T[:3,3]) # center distance
    return dist<thres   


def group_optimization(item_list, prev_optim_T, tg_scene=None, img_bucket=None, obj_dict=None, \
                       loop_numb=50, lr=1e-4, sampling_numb=None, stepsize=5, vis=False, use_ceres=False):
    src_3d_list = []
    tg_2d_list = []
    proj_matrixes = []

    for matching_item in item_list:
        filtered_combined_src_3d = matching_item.combined_src_3d[matching_item.inliers]
        src_3d_list.append(filtered_combined_src_3d)
        filtered_tg_2d = matching_item.combined_tg_2d[matching_item.inliers]
        tg_2d_list.append(filtered_tg_2d)
        L = filtered_combined_src_3d.shape[0]
        proj_matrixes.append(np.tile(matching_item.proj_matrix,(L,1,1)))

    stacked_src_3d = np.vstack(src_3d_list)
    stacked_tg_2d = np.vstack(tg_2d_list)
    stacked_proj_matrixes = np.vstack(proj_matrixes)      
    
    if prev_optim_T is not None:
        initial_T = prev_optim_T
    else:
        initial_T = matching_item.initial_T

    assert stacked_src_3d.shape[0]  == stacked_tg_2d.shape[0] and stacked_src_3d.shape[0] == stacked_proj_matrixes.shape[0], 'N should be same'
    if not use_ceres:
        min_loss, optim_output = optimize(stacked_src_3d, stacked_tg_2d, stacked_proj_matrixes, initial_T, \
                                      loop_numb=loop_numb, lr=lr, sampling_numb=sampling_numb,
                                      vis=vis, vis_tgs=(tg_scene, img_bucket, obj_dict, item_list, stepsize))
    else:
        min_loss, optim_output = optimize_ceres(stacked_src_3d, stacked_tg_2d, stacked_proj_matrixes, initial_T, \
                                      loop_numb=loop_numb, lr=lr, sampling_numb=sampling_numb,
                                      vis=vis, vis_tgs=(tg_scene, img_bucket, obj_dict, item_list, stepsize))
    return min_loss, optim_output



# detection bucket, matching bucket, initial3d_bucket
class MatchingSet:
    def __init__(self, idx, item:MatchItem, tg_scene=None, img_bucket=None):

        self.idx = idx
        self.set = set()
        self.set.add(item)
        self.optim_T = item.initial_T
        self.tg_scene = tg_scene
        self.img_bucket = img_bucket
        

    def validate_compatibility(self, new_item:MatchItem, obj_dict=None, translation_thres=0.05, loss_thres=10, \
                                loop_numb=100, vis=False, ceres=False): # 8 pixels
        # points3d : NX3
        # projections: MX3X4
        # Check with Initial Translation
        if (np.linalg.norm(self.optim_T[:3,3]-new_item.initial_T[:3,3]) < translation_thres):
            # min_loss
            min_loss, optim_output = group_optimization(list(self.set)+[new_item], self.optim_T, \
                                    self.tg_scene, self.img_bucket, obj_dict, loop_numb=loop_numb, \
                                    lr=1e-4, stepsize=5, vis=vis, use_ceres=ceres)
            
            # Check the optimization output to current new item
            T_opt = np.eye(4)
            T_opt[:3,:3] = rotation_6d_to_matrix_np(optim_output['min_6d'])
            T_opt[:3,3] = optim_output['min_t']
            distance = new_item.check_output_wdistance(T_opt)

            if min_loss < loss_thres and distance < loss_thres:
                print("Valid")
                return min_loss, distance, optim_output
            else:
                print(f"Unvalid: NOT Converged issue min_loss {min_loss} target view loss {distance}")
                return min_loss, distance, None
        else:
            print("Translation Threshold")
            return None, None, None
    
    # def validate_compatibility(self, new_item:MatchItem, dist_thres=0.05, in_mask_thres=0.3):
    #     # points3d : NX3
    #     # projections: MX3X4
    #     in_flag = 0
    #     for item in self.set: # JX3, 1X3X4
    #         projected_2d = project_3d_to_2d(new_item.transformed_verts, item.proj_matrix[None]).squeeze().astype(np.int64) # JX2
    #         h,w = item.mask.shape[:2]
    #         in_img_flag = np.logical_and(np.logical_and(projected_2d[:,0]>=0 ,projected_2d[:,0]<w), np.logical_and(projected_2d[:,1]>=0 ,projected_2d[:,1]<h))
    #         if sum(in_img_flag)<=0:
    #             continue
    #         mask_flag1 = item.mask[projected_2d[in_img_flag][:,1],projected_2d[in_img_flag][:,0],0]>0
    #         mask_flag1 = (sum(mask_flag1)/len(mask_flag1)) > in_mask_thres

    #         projected_2d = project_3d_to_2d(item.transformed_verts, new_item.proj_matrix[None]).squeeze().astype(np.int64) # JX2
    #         h,w = new_item.mask.shape[:2]
    #         in_img_flag = np.logical_and(np.logical_and(projected_2d[:,0]>=0 ,projected_2d[:,0]<w), np.logical_and(projected_2d[:,1]>=0 ,projected_2d[:,1]<h))
    #         if sum(in_img_flag)<=0:
    #             continue
    #         mask_flag2 = new_item.mask[projected_2d[in_img_flag][:,1],projected_2d[in_img_flag][:,0],0]>0
    #         mask_flag2 = (sum(mask_flag2)/len(mask_flag2)) > in_mask_thres

    #         dist = np.linalg.norm(new_item.initial_T[:3,3]-item.initial_T[:3,3]) # center distance
    #         dist_flag = dist<dist_thres        
    #         print(f'{new_item.cam_id}_{new_item.midx} with {item.cam_id}_{item.midx} mask_flag1: {mask_flag1}, mask_flag2: {mask_flag2}, dist_flag: {dist_flag}')
    #         in_flag += 1 if mask_flag1&mask_flag2&dist_flag else 0

    #     return in_flag/len(self.set)
    
    def add(self, new_item:MatchItem):
        self.set.add(new_item)
    
    def remove(self, new_item):
        self.set.remove(new_item)

    def update_T(self, optim_output):
        if torch.is_tensor(optim_output['min_6d']):
            update_T = torch.eye(4)
            update_T[:3,:3] =  rotation_6d_to_matrix(optim_output['min_6d'])
            update_T[:3,3] =   optim_output['min_t']
            
            self.optim_T = update_T.detach().cpu().numpy()
        else:
            update_T = np.eye(4)
            update_T[:3,:3] = rotation_6d_to_matrix_np(optim_output['min_6d'])
            update_T[:3,3] = optim_output['min_t']

            self.optim_T = update_T