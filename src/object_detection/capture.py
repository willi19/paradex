import threading
import numpy as np
import torch
import cv2
import json
import time
import os
from pathlib import Path
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

from paradex.utils.file_io import shared_dir, load_current_camparam
from paradex.utils.env import get_pcinfo, get_serial_list

from paradex.io.capture_pc.camera_main import RemoteCameraController
from paradex.io.capture_pc.util import get_client_socket
from paradex.io.capture_pc.connect import git_pull, run_script

from paradex.object_detection.multiview_utils.template import Template
from paradex.object_detection.obj_utils.vis_utils import parse_objectmesh_objdict
from paradex.object_detection.object_optim_config import template_path
from paradex.object_detection.object_optim_config import obj_list
from paradex.object_detection.obj_utils.io import read_camerainfo
from paradex.object_detection.obj_utils.multicam_system import MultiCamScene, NAS_IMG_SAVEDIR
from paradex.object_detection.obj_utils.geometry import project_3d_to_2d
from paradex.object_detection.multiview_utils.img_processing import draw_inliers, \
                            rendersil_obj2allview, SRC_COLOR, draw_text
from paradex.object_detection.multiview_utils.matchingset import MatchItem, MatchingSet, group_optimization
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--obj_name', type=str, required=True)
parser.add_argument('--camerainfo_dir', required=True, type=str)
parser.add_argument('--debug', action='store_true')
parser.add_argument('--default_rescale', type=float, default=0.5)
args = parser.parse_args()
inliers_threshold = 30

assert args.obj_name in obj_list, 'Check the object name or object is already registered'
obj_dict = parse_objectmesh_objdict(args.obj_name, min_vertex_num=1000, \
                                            remove_uv=True, renderer_type='nvdiffrast', device=DEVICE)
# template = Template(template_path[args.obj_name], obj_name=args.obj_name)
org_scaled_verts = obj_dict['verts'][0].clone().detach()
sampled_indexes = torch.randperm(org_scaled_verts.shape[0])[:100]
sampled_obj_verts = org_scaled_verts[sampled_indexes]

BOARD_COLORS = [
    (0, 0, 255), 
    (0, 255, 0)
]

pc_info = get_pcinfo()
serial_list = get_serial_list()

saved_corner_img = {serial_num:np.ones((1536, 2048, 3), dtype=np.uint8)*255 for serial_num in serial_list}
cur_state = {serial_num:{} for serial_num in serial_list}

capture_idx = 0
filename = time.strftime("%Y%m%d_%H%M%S", time.localtime())

scene = MultiCamScene(rescale_factor=args.default_rescale, device=DEVICE)
scene.get_batched_renderer(tg_cam_list=scene.cam_ids)
DEBUG_VIS = Path(NAS_IMG_SAVEDIR)/'debug'
os.makedirs(DEBUG_VIS, exist_ok=True)

def listen_socket(pc_name, socket):
    while True:
        msg = socket.recv_string()
        try:
            data = json.loads(msg)
        except json.JSONDecodeError:
            print(f"[{pc_name}] Non-JSON message: {msg}")
            continue
        
        if data.get("type") == "2D_matching":
            serial_num = data["serial_num"]
            matching_output = data["detect_result"]
            print(matching_output)
            frame = data["frame"]
            cur_state[serial_num] = matching_output                

        else:
            print(f"[{pc_name}] Unknown JSON type: {data.get('type')}")

pc_list = list(pc_info.keys())
git_pull("merging", pc_list)
# run_script(f"python paradex/object_detection/client.py --obj_name {args.obj_name}", pc_list, log=True)

camera_controller = RemoteCameraController("stream", None, sync=True, debug=args.debug)
camera_controller.start()

try:
    socket_dict = {name:get_client_socket(pc_info["ip"], 5564) for name, pc_info in pc_info.items()}

    for pc_name, sock in socket_dict.items():
        threading.Thread(target=listen_socket, args=(pc_name, sock), daemon=True).start()
        
    while True:
        img_dict = {}
        matchingitem_dict = {}
        # TODO change here.
        for serial_num in serial_list:
            if serial_num not in scene.cam2intr:
                continue

            img_dict[serial_num] = scene.get_image(serial_num, rgb=True)
            # img = saved_corner_img[serial_num].copy()
            matching_output = cur_state[serial_num]
            proj_matrix = scene.proj_matrix[serial_num]
            for midx in matching_output:
                # {'count':pair_count,'combined_src_3d':combined_src_3d, \
                # 'combined_tg_2d':combined_tg_2d, 'src_arr_cam_ids':src_arr_cam_ids}
                if 'count' in matching_output[midx] and matching_output[midx]['count']>0:
                    for key in ['combined_src_3d', 'combined_tg_2d', 'src_arr_cam_ids']:
                        matching_output[midx][key] = np.array(matching_output[midx][key])
                    combined_src_3d, combined_tg_2d = matching_output[midx]['combined_src_3d'], matching_output[midx]['combined_tg_2d']
                    src_cam_ids = matching_output[midx]['src_arr_cam_ids']
                    
                    tg_cam_extr_4X4 = np.eye(4)
                    tg_cam_extr_4X4[:3] = scene.cam2extr[serial_num]
                
                    ret, rvec, tvec, inliers = cv2.solvePnPRansac(
                        objectPoints = combined_src_3d,
                        imagePoints = combined_tg_2d,
                        cameraMatrix = scene.cam2intr[serial_num], distCoeffs=None,
                        reprojectionError=8,
                        flags=cv2.SOLVEPNP_ITERATIVE)
                    
                    if ret:
                        obj2img_matrix = np.eye(4)
                        obj2img_matrix[:3, :3] = cv2.Rodrigues(rvec)[0]
                        obj2img_matrix[:3, 3]  = tvec[:, 0]
                        obj_tg_T = torch.tensor(np.linalg.inv(tg_cam_extr_4X4)@obj2img_matrix, device=DEVICE).float()
                        # initial_3d_bucket[serial_num][midx] = (obj_tg_T, inliers)

                        inliers_mask = np.zeros((combined_src_3d.shape[0]))
                        inliers_mask[inliers]=1
                        matching_output[midx]['inliers'] = inliers_mask
                        matching_output[midx]['inliers_count'] = len(inliers)
                        
                        if matching_output[midx]['inliers_count'] > inliers_threshold:
                            transformed_verts = torch.einsum('mn, jn -> jm', obj_tg_T[:3,:3], \
                            sampled_obj_verts)+ obj_tg_T[:3,3]

                            new_item = MatchItem(cam_id=serial_num, midx=midx, matching_item=matching_output[midx],\
                                mask=None, \
                                initial_T=obj_tg_T, \
                                transformed_verts=transformed_verts.detach().cpu().numpy(),
                                proj_matrix=proj_matrix)

                            matchingitem_dict[f'{serial_num}_{midx}'] = new_item
                            # translated_T = np.copy(new_item.initial_T)
                            # Render output
                            
                            if args.debug:
                                #  Render To All View
                                transformed_verts = np.einsum('mn, jn -> jm', obj_tg_T[:3,:3].detach().cpu().numpy(), combined_src_3d)+ obj_tg_T[:3,3].detach().cpu().numpy()
                                projected_2d = project_3d_to_2d(transformed_verts, proj_matrix[None]).squeeze().astype(np.int64)
                                mean_distance_inlier = np.sum(inliers_mask*np.linalg.norm((projected_2d-combined_tg_2d),axis=1))/np.sum(inliers_mask)
                                rendered_sil, _ = rendersil_obj2allview(scene, obj_dict, obj_tg_T, img_dict, \
                                                                highlight={serial_num:SRC_COLOR})
                                inlier_count = matching_output[midx]['inliers_count']
                                img_name = f'{serial_num}_{midx}_using_combined_{inliers.shape[0]}_inliernumb{inlier_count}_loss{mean_distance_inlier}.jpeg'
                                cv2.imwrite(str(DEBUG_VIS/img_name), rendered_sil)
                            
                        

            cur_state[serial_num] = matching_output
        matchingset_list = []
        keys_sorted = sorted(matchingitem_dict.keys(), key=lambda k: matchingitem_dict[k].inlier_count, reverse=True)

        for key in keys_sorted:
            print(f"** {key} Optimization")
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
                loss, distance, optim_output = exising_set.validate_compatibility(new_item, obj_dict=obj_dict, \
                                                                        translation_thres=0.3, loss_thres=10, \
                                                                        loop_numb=30, vis=False, ceres=True)
                print(f'loss:{loss} distance:{distance} optim_ouptput:{optim_output}')

                # move rendered image target information: loss, set number
                if loss is not None: # optimization run
                    if optim_output is not None: # Optimization Run
                        valid = True 
                        if min_validset_idx is None or min_valid_distance>distance:
                            min_validset_idx = sidx
                            min_valid_distance = distance
                            min_optim_output = optim_output

            if len(matchingset_list)>0 and valid:
                matchingset_list[min_validset_idx].add(new_item)
                matchingset_list[min_validset_idx].update_T(min_optim_output)
            else:
                matchingset_list.append(MatchingSet(len(matchingset_list), new_item, tg_scene=None, img_bucket=img_dict))
                
        print(f"Total {len(matchingset_list)} sets")
        # Rendering! 
        
        
finally:
    camera_controller.end()
    camera_controller.quit()        