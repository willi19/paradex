'''
    Codes for multiview parsing
        multiview combined > split code
'''
import numpy as np 
import torch
import cv2
from copy import deepcopy

# Update matching bucket
def combine_multiview_matching(matching_bucket, tg_cams=None):
    for cidx, tg_cam_id in enumerate(matching_bucket): 
        # parsing matching bucket.
        if tg_cams is None or tg_cam_id in tg_cams:
            for midx in matching_bucket[tg_cam_id]:
                # For Each Mask
                if matching_bucket[tg_cam_id][midx]['count'] > 0:
                    src_3d_dict = matching_bucket[tg_cam_id][midx]['src_3d']
                    tg_2d_dict = matching_bucket[tg_cam_id][midx]['tg_2d']
                    org_2d_dict = matching_bucket[tg_cam_id][midx]['src_2d']
                    
                    # concat all src_3d and tg_2d
                    src_3d_points = []
                    tg_2d_points = []
                    src_cam_ids = []
                    for src_cam_id in src_3d_dict:
                        if len(src_3d_dict[src_cam_id])>0:
                            src_3d_points.append(src_3d_dict[src_cam_id])
                            tg_2d_points.append(tg_2d_dict[src_cam_id])
                            src_cam_ids.append([src_cam_id]*len(src_3d_dict[src_cam_id]))
                            
                    if len(src_3d_points)>0:
                        src_3d_points = np.vstack(src_3d_points).astype(np.float64)
                        tg_2d_points = np.vstack(tg_2d_points).astype(np.float64)
                        matching_bucket[tg_cam_id][midx]['combined_src_3d'] = src_3d_points # combined array
                        matching_bucket[tg_cam_id][midx]['combined_tg_2d'] = tg_2d_points # combined_array
                        matching_bucket[tg_cam_id][midx]['src_arr_cam_ids'] = np.hstack(src_cam_ids)
                    else:
                        print("Bug")
                
    return matching_bucket
                

# multiview inliers to each ininliers
def parsing_inlier2flag(inliers, src_arr_cam_ids):

    inliers_idxes = np.hstack(inliers)
    ttl_inlier_flag = np.zeros(len(src_arr_cam_ids))
    ttl_inlier_flag[inliers_idxes] = 1
    
    inlier_flags = {}
    for cam_id in np.unique(src_arr_cam_ids):
        inlier_flags[cam_id] = ttl_inlier_flag[src_arr_cam_ids==cam_id].astype(np.bool_)
        
    return inlier_flags

