import threading
import numpy as np
import torch
import cv2
import json
import time
import os
import shutil
from pathlib import Path
import pickle
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
from threading import Event, Thread

from paradex.utils.env import get_pcinfo, get_serial_list

from paradex.io.capture_pc.camera_main import RemoteCameraController
from paradex.io.capture_pc.util import get_client_socket
from paradex.io.capture_pc.connect import git_pull, run_script
from paradex.io.signal_generator.UTGE900 import UTGE900
from paradex.utils.keyboard_listener import listen_keyboard
from paradex.object_detection.obj_utils.vis_utils import parse_objectmesh_objdict
from paradex.object_detection.obj_utils.io import read_camerainfo, get_ttl_framenumb
from paradex.object_detection.obj_utils.multicam_system import MultiCamScene, NAS_IMG_SAVEDIR
from paradex.object_detection.obj_utils.geometry import project_3d_to_2d
from paradex.object_detection.multiview_utils.img_processing import draw_inliers, \
                            rendersil_obj2allview, SRC_COLOR, draw_text
from paradex.object_detection.multiview_utils.optimizer import combined_visualizer
from paradex.object_detection.multiview_utils.matchingset import MatchItem, MatchingSet, group_optimization

def process_object(process_args):
    cur_state, obj_name, scene, obj_dict_bucket, img_bucket, args, \
        oneview_dir, optimization_dir, inliers_threshold, loss_thres, sampled_obj_verts  = process_args
    
    initial_3d_bucket = {}
    # matching_bucket = {}
    matchingitem_dict = {}
    for serial_num in cur_state:
        initial_3d_bucket[serial_num] = {}
        if obj_name in cur_state[serial_num]:
            for midx in cur_state[serial_num][obj_name]:
                if cur_state[serial_num][obj_name][midx]['count']>0:
                    tmp_matching = cur_state[serial_num][obj_name][midx]
                    for key in ['combined_src_3d', 'combined_tg_2d', 'src_arr_cam_ids']:
                        tmp_matching[key] = np.array(tmp_matching[key])
                    
                    combined_src_3d, combined_tg_2d = tmp_matching['combined_src_3d'], tmp_matching['combined_tg_2d']
                    src_cam_ids = tmp_matching['src_arr_cam_ids']
                    
                    ret, rvec, tvec, inliers = cv2.solvePnPRansac(
                        objectPoints=combined_src_3d,
                        imagePoints=combined_tg_2d,
                        cameraMatrix=scene.cam2intr[serial_num], distCoeffs=None,
                        reprojectionError=8,
                        flags=cv2.SOLVEPNP_ITERATIVE)
                    
                    if ret:
                        obj2img_matrix = np.eye(4)
                        obj2img_matrix[:3, :3] = cv2.Rodrigues(rvec)[0]
                        obj2img_matrix[:3, 3]  = tvec[:, 0]
                        obj_tg_T = torch.tensor(np.linalg.inv(scene.cam2extr_4X4[serial_num])@obj2img_matrix, device=DEVICE).float()
                        initial_3d_bucket[serial_num][midx] = (obj_tg_T, inliers)

                        inliers_mask = np.zeros((combined_src_3d.shape[0]))
                        inliers_mask[inliers]=1
                        tmp_matching['inliers'] = inliers_mask
                        tmp_matching['inliers_count'] = len(inliers)
                        
                        if args.debug:
                            #  Render To All View
                            transformed_verts = np.einsum('mn, jn -> jm', obj_tg_T[:3,:3].detach().cpu().numpy(), combined_src_3d)+ obj_tg_T[:3,3].detach().cpu().numpy()
                            projected_2d = project_3d_to_2d(transformed_verts, scene.proj_matrix[serial_num][None]).squeeze().astype(np.int64)
                            mean_distance_inlier = np.sum(inliers_mask*np.linalg.norm((projected_2d-combined_tg_2d),axis=1))/np.sum(inliers_mask)
                            rendered_sil, _ = rendersil_obj2allview(scene, obj_dict_bucket[obj_name], obj_tg_T, img_bucket, \
                                                            highlight={serial_num:SRC_COLOR})
                            cv2.imwrite(str(oneview_dir/f"{serial_num}_{midx}_using_combined_{inliers.shape[0]}_{mean_distance_inlier}_{tmp_matching['inliers_count']}.jpeg") ,\
                                rendered_sil)
                            
                        if tmp_matching['inliers_count'] > inliers_threshold:
                            transformed_verts = torch.einsum('mn, jn -> jm', initial_3d_bucket[serial_num][midx][0][:3,:3], \
                                        sampled_obj_verts)+ initial_3d_bucket[serial_num][midx][0][:3,3]

                            new_item = MatchItem(cam_id=serial_num, midx=midx, matching_item=tmp_matching,\
                                mask=None, \
                                initial_T=initial_3d_bucket[serial_num][midx][0], \
                                transformed_verts=transformed_verts.detach().cpu().numpy(),
                                proj_matrix=scene.proj_matrix[serial_num])

                    matchingitem_dict[f'{serial_num}_{midx}'] = new_item
                # matching_bucket[serial_num][midx] = tmp_matching       
    

    # Make Set using Optimization
    matchingset_list = []
    keys_sorted = sorted(matchingitem_dict.keys(), key=lambda k: matchingitem_dict[k].inlier_count, reverse=True)

    for key in keys_sorted:
        print(f"** {key} Optimization, Finding set from {len(matchingset_list)} set")
        serial_num, midx = key.split("_")
        new_item = matchingitem_dict[key]
        
        min_validset_idx = None
        min_valid_distance = None
        min_optim_output = None

        valid = False
        for sidx, exising_set in enumerate(matchingset_list):
            # check with optim
            # NOTE: translation thres: I increased translation threshold because few pnp output result in bad translation (no depth input)
            # validate compatibility: check center distance + run group optimization and return and filtering result
            loss, distance, optim_output = exising_set.validate_compatibility(new_item, obj_dict=obj_dict_bucket[obj_name], \
                                                                    translation_thres=0.3, loss_thres=loss_thres, \
                                                                    loop_numb=30, vis=(args.debug), ceres=True)
            print(f'loss:{loss} distance:{distance}')
            if loss is not None and optim_output is not None:
                    valid = True 
                    if (args.debug):
                        shutil.move('./tmp/optim/rendered_pairs_optim.mp4', \
                            str(optimization_dir/f'rendered_pairs_optim_set{exising_set.idx}_{serial_num}_{midx}.mp4'))
                        target_path = str(optimization_dir/f'set{exising_set.idx}_{new_item.cam_id}_{new_item.midx}_{valid}_loss_{loss}_match.jpeg')
                        #shutil.copy('tmp_pairs_after_optim.jpeg', target_path)
                        shutil.copy('tmp_pairs.jpeg', target_path)
                    if min_validset_idx is None or min_valid_distance>distance:
                        min_validset_idx = sidx
                        min_valid_distance = distance
                        min_optim_output = optim_output

        if len(matchingset_list)>0 and valid:
            matchingset_list[min_validset_idx].add(new_item)
            matchingset_list[min_validset_idx].update_T(min_optim_output)
        else:
            print("Make New matching set")
            matchingset_list.append(MatchingSet(len(matchingset_list), new_item, tg_scene=scene, img_bucket=img_bucket))

    # ed_time = time.time()
    return obj_name, matchingset_list