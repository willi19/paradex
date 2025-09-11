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

from paradex.utils.file_io import shared_dir, load_current_camparam
from paradex.utils.env import get_pcinfo, get_serial_list

from paradex.io.capture_pc.camera_main import RemoteCameraController
from paradex.io.capture_pc.util import get_client_socket
from paradex.io.capture_pc.connect import git_pull, run_script
from paradex.io.signal_generator.UTGE900 import UTGE900

from paradex.object_detection.multiview_utils.template import Template
from paradex.object_detection.obj_utils.vis_utils import parse_objectmesh_objdict
from paradex.object_detection.object_optim_config import template_path
from paradex.object_detection.object_optim_config import obj_list
from paradex.object_detection.obj_utils.io import read_camerainfo,get_ttl_framenumb
from paradex.object_detection.obj_utils.multicam_system import MultiCamScene, NAS_IMG_SAVEDIR
from paradex.object_detection.obj_utils.geometry import project_3d_to_2d
from paradex.object_detection.multiview_utils.img_processing import draw_inliers, \
                            rendersil_obj2allview, SRC_COLOR, draw_text
from paradex.object_detection.multiview_utils.optimizer import combined_visualizer
from paradex.object_detection.multiview_utils.matchingset import MatchItem, MatchingSet, group_optimization
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--obj_name', type=str, required=True)
parser.add_argument('--camerainfo_dir', required=True, type=str)
parser.add_argument('--debug', action='store_true')
parser.add_argument('--default_rescale', type=float, default=0.5)
parser.add_argument('--loss_thres', type=float, default=12)
parser.add_argument('--toggle', action='store_true')
args = parser.parse_args()
inliers_threshold = 30

assert args.obj_name in obj_list, 'Check the object name or object is already registered'
obj_dict = parse_objectmesh_objdict(args.obj_name, min_vertex_num=1000, \
                                            remove_uv=True, renderer_type='nvdiffrast', device=DEVICE)
# template = Template(template_path[args.obj_name], obj_name=args.obj_name)
org_scaled_verts = obj_dict['verts'][0].clone().detach()
sampled_indexes = torch.randperm(org_scaled_verts.shape[0])[:100]
sampled_obj_verts = org_scaled_verts[sampled_indexes]
min_L = min(sampled_obj_verts.max(axis=0)[0]-sampled_obj_verts.min(axis=0)[0]).item()

BOARD_COLORS = [
    (0, 0, 255), 
    (0, 255, 0)
]

pc_info = get_pcinfo()
serial_list = get_serial_list()

saved_corner_img = {serial_num:np.ones((1536, 2048, 3), dtype=np.uint8)*255 for serial_num in serial_list}
cur_state = {}
cur_numinput = {}
for serial_num in serial_list:
    cur_state[serial_num] = {}

capture_idx = 0
filename = time.strftime("%Y%m%d_%H%M%S", time.localtime())

scene = MultiCamScene(rescale_factor=args.default_rescale, device=DEVICE, height=1536, width=2048)
scene.get_batched_renderer(tg_cam_list=scene.cam_ids)
DEBUG_VIS = Path(NAS_IMG_SAVEDIR)/'debug'
os.makedirs(DEBUG_VIS, exist_ok=True)
OUTPUTDIR = './objoutput'
os.makedirs(OUTPUTDIR, exist_ok=True)

signal_generator = UTGE900()
signal_generator.generate(freq=2000) # 100 frequency > 10Hz 1000 > 1Hz , 2000 > 0.5Hz

cur_tg_frame = -1

def listen_socket(pc_name, socket):
    global cur_tg_frame
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
            frame = data["frame"]
            if frame not in cur_state[serial_num]:
                cur_state[serial_num][frame] = matching_output   
                if frame not in cur_numinput:
                    cur_numinput[frame]=1
                else:
                    cur_numinput[frame]+=1
                    print(f"Number of inputs {frame}: {cur_numinput[frame]}")
            else:
                cur_state[serial_num][frame] = matching_output   
            if len(matching_output)>0:
                if cur_tg_frame==-1:
                    cur_tg_frame = frame+5
        else:
            print(f"[{pc_name}] Unknown JSON type: {data.get('type')}")

pc_list = list(pc_info.keys())
git_pull("merging", pc_list)
# if args.debug:
#     run_script(f"python paradex/object_detection/client.py --obj_name {args.obj_name} --saveimg", pc_list, log=True)
# else:
#     run_script(f"python paradex/object_detection/client.py --obj_name {args.obj_name}", pc_list, log=True)


camera_controller = RemoteCameraController("stream", None, sync=True, debug=args.debug)
camera_controller.start()
if args.toggle:
    signal_generator.off(1)
else:
    signal_generator.on(1)

try:
    socket_dict = {name:get_client_socket(pc_info["ip"], 5564) for name, pc_info in pc_info.items()}

    for pc_name, sock in socket_dict.items():
        threading.Thread(target=listen_socket, args=(pc_name, sock), daemon=True).start()
        
    if args.debug:
        img_dict = {}
        for serial_num in serial_list:
            img_dict[serial_num] = scene.get_image(serial_num, rgb=True)
    else:
        img_dict = {}
    
    processing_frame = 1
    while True:
        if cur_tg_frame==-1: # Not intial matching output is given
            continue
        # print(f'Frame: {cur_tg_frame} number of input image: {get_ttl_framenumb(cur_state, cur_tg_frame)}')

        if cur_tg_frame in cur_numinput and cur_numinput[cur_tg_frame]>=20:   
            print(f"Processing start with frame {cur_tg_frame}")

            if serial_num in serial_list:
                img_path = os.path.join(NAS_IMG_SAVEDIR,f'frame_{cur_tg_frame%10}_{serial_num}.jpeg')
                if os.path.exists(img_path):
                    img_dict[serial_num] = cv2.imread(img_path)
                    
            matchingitem_dict = {}
            # TODO change here.
            st_time = time.time()
            for serial_num in serial_list:
                if serial_num not in scene.cam2intr and serial_num in cur_state:
                    continue
                if cur_tg_frame not in  cur_state[serial_num]:
                    continue

                # img = saved_corner_img[serial_num].copy()
                matching_output = cur_state[serial_num][cur_tg_frame]
                
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
                            # print(f"inliercount : {matching_output[midx]['inliers_count'] }")
                            
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
                                
                                if args.debug and False:
                                    #  Render To All View
                                    transformed_verts = np.einsum('mn, jn -> jm', obj_tg_T[:3,:3].detach().cpu().numpy(), combined_src_3d)+ obj_tg_T[:3,3].detach().cpu().numpy()
                                    projected_2d = project_3d_to_2d(transformed_verts, proj_matrix[None]).squeeze().astype(np.int64)
                                    mean_distance_inlier = np.sum(inliers_mask*np.linalg.norm((projected_2d-combined_tg_2d),axis=1))/np.sum(inliers_mask)
                                    rendered_sil, _ = rendersil_obj2allview(scene, obj_dict, obj_tg_T, img_dict, \
                                                                    highlight={serial_num:SRC_COLOR})
                                    inlier_count = matching_output[midx]['inliers_count']
                                    img_name = f'{serial_num}_{midx}_using_combined_{inliers.shape[0]}_inliernumb{inlier_count}_loss{mean_distance_inlier}.jpeg'
                                    cv2.imwrite(str(DEBUG_VIS/img_name), rendered_sil)
                                
            for serial_num in serial_list:
                if cur_tg_frame in cur_state[serial_num]:
                    cur_state[serial_num].pop(cur_tg_frame)
                
            # cur_state.pop(cur_tg_frame)
            print(f"matching list number : {len(matchingitem_dict)}")
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
                                                                            translation_thres=0.3, loss_thres=args.loss_thres, \
                                                                            loop_numb=50, vis=False, ceres=True)
                    print(f'{new_item.cam_id} loss:{loss} distance:{distance}')

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
                    matchingset_list.append(MatchingSet(len(matchingset_list), new_item, tg_scene=scene, img_bucket=img_dict))
                    
            print(f"Total {len(matchingset_list)} sets")

            # output visualization if needed
            output_dict = {}
            output_idx = 0
            reoptim = False

            # combine matchingset using the translation

            if len(matchingset_list)>=2:
                print("converge")
                for midx in range(len(matchingset_list)):
                    for mmidx in range(midx+1, len(matchingset_list)):
                        set1, set2 = matchingset_list[midx], matchingset_list[mmidx]
                        if set1 and set2 is not None:
                            center_dist = np.linalg.norm(set1.optim_T[:3,3]-set2.optim_T[:3,3])
                            if center_dist<min_L:
                                # Check combineable
                                min_loss, optim_output = group_optimization(list(set1.set)+list(set2.set), set1.optim_T, scene, img_dict, obj_dict, loop_numb=50)
                                if min_loss < args.loss_thres:
                                    set1.set = set1.set.union(set2.set)
                                    set1.update_T(optim_output)
                                    matchingset_list[mmidx] = None
                                else:
                                    print(f"Not merging {midx} with {mmidx} because of the loss {min_loss}")
                                    if len(set1.set)>= len(set2.set):
                                        matchingset_list[mmidx] = None
                                    else:
                                        matchingset_list[midx] = None
                    
                                    
                                    



            for matchingset in matchingset_list: 
                if len(matchingset.set) >= 3: # TODO: check thres
                        if args.debug:
                            if reoptim:
                                firstitem = list(matchingset.set)[0]

                                min_loss, optim_output = group_optimization(list(matchingset.set), matchingset.optim_T, \
                                                                    scene, img_dict, obj_dict, \
                                                                    loop_numb=30, stepsize=2, vis= args.debug, use_ceres=True)
                                print(f'loss:{loss} optim_ouptput:{optim_output}')
                                
                                if min_loss is not None and optim_output is not None:
                                    shutil.move('./tmp/optim/rendered_pairs_optim.mp4', str(DEBUG_VIS/f'rendered_pairs_optim_set{matchingset.idx}_{serial_num}_{midx}.mp4'))
                                    valid = True if optim_output is not None else False
                                    target_path = str(DEBUG_VIS/f'set{matchingset.idx}_{valid}_loss_{min_loss}_match.jpeg')
                                    shutil.copy('tmp_pairs.jpeg', target_path)
                            else:
                                target_path = str(DEBUG_VIS/f'set{matchingset.idx}_match.jpeg')
                                highlights = {}
                                for matchingitem in list(matchingset.set):
                                    highlights[matchingitem.cam_id] = SRC_COLOR

                                rendered_on_overlaid = combined_visualizer(matchingset.optim_T, scene, obj_dict, list(matchingset.set), \
                                                    img_dict, highlights, DEVICE)
                                cv2.imwrite(target_path, rendered_on_overlaid)                        
                        output_dict[output_idx] = matchingset.optim_T
                        output_idx+=1
                        
            if len(output_dict)>0:
                pickle.dump(output_dict,open(os.path.join(OUTPUTDIR,'obj_T.pkl'),'wb'))

            ed_time = time.time()
            print(f"One round time {ed_time-st_time:.2f} sec")
            
            cur_tg_frame+=1
        else:
            time.sleep(0.1)


        
        
finally:
    camera_controller.end()
    camera_controller.quit()        
    if args.toggle:
        signal_generator.on(1)
    else:
        signal_generator.off(1)
    signal_generator.quit()