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
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--obj_names', type=str, required=True, nargs='+')
parser.add_argument('--debug', action='store_true')
parser.add_argument('--vis_final', action='store_true')
parser.add_argument('--default_rescale', type=float, default=0.5)
parser.add_argument('--loss_thres', type=float, default=12)
parser.add_argument('--toggle', action='store_true')
args = parser.parse_args()
inliers_threshold = 20
cam_numb_thres = 18
loss_thres = 10

obj_dict_bucket = {}
for obj_name in args.obj_names:
    obj_dict = parse_objectmesh_objdict(obj_name, min_vertex_num=1000, \
                                            remove_uv=True, renderer_type='nvdiffrast', device=DEVICE)
    org_scaled_verts = obj_dict['verts'][0].clone().detach()
    sampled_indexes = torch.randperm(org_scaled_verts.shape[0])[:100]
    sampled_obj_verts = org_scaled_verts[sampled_indexes]
    min_L = min(sampled_obj_verts.max(axis=0)[0]-sampled_obj_verts.min(axis=0)[0]).item()
    obj_dict['sampled_obj_verts'] = sampled_obj_verts
    obj_dict['min_L']=min_L
    obj_dict_bucket[obj_name] = obj_dict


BOARD_COLORS = [
    (0, 0, 255), 
    (0, 255, 0)
]

pc_info = get_pcinfo()
serial_list = get_serial_list()
capture_event = Event()
listen_keyboard({"c":capture_event})
saved_corner_img = {serial_num:np.ones((1536, 2048, 3), dtype=np.uint8)*255 for serial_num in serial_list}

cur_state = {}
cur_numinput = None
    
capture_idx = 1
filename = time.strftime("%Y%m%d_%H%M%S", time.localtime())

scene = MultiCamScene(rescale_factor=args.default_rescale, device=DEVICE, height=1536, width=2048)
scene.get_batched_renderer(tg_cam_list=scene.cam_ids)
DEBUG_VIS = Path('./debug')
os.makedirs(DEBUG_VIS, exist_ok=True)

# timestamp = time.strftime("%Y%m%d-%H%M%S")
objoutput_path = './obj_output'
os.makedirs(objoutput_path, exist_ok=True)
oneview_dir = Path(objoutput_path+'/one_view')
os.makedirs(oneview_dir, exist_ok=True)
optimization_dir = Path(objoutput_path+'/optim')
os.makedirs(optimization_dir, exist_ok=True)
set_root_dir = Path(objoutput_path+'/set')
os.makedirs(set_root_dir, exist_ok=True)
image_root_dir = Path(objoutput_path+'/image')
os.makedirs(image_root_dir, exist_ok=True)    
save_first_image = True

    
def get_frameinfo(cur_state, serial_num):
    return cur_state[serial_num] if serial_num in cur_state else None

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
            frame = data["frame"]
            if len(matching_output)>0:
                cur_state[serial_num] = matching_output
            print(f"Frame {frame} got total {len(cur_state)} inputs")
        else:
            print(f"[{pc_name}] Unknown JSON type: {data.get('type')}")

pc_list = list(pc_info.keys())
git_pull("merging", pc_list)

if args.debug:
    run_script(f"python paradex/object_detection/client.py --obj_names {' '.join(args.obj_names)} --saveimg", pc_list, log=False)
else:
    run_script(f"python paradex/object_detection/client.py --obj_names {' '.join(args.obj_names)}", pc_list, log=False)

save_path = './shared_data/tmp_images'
if os.path.exists(Path.home()/save_path):
    shutil.rmtree(Path.home()/save_path)
os.makedirs(save_path, exist_ok=True)

camera_controller = RemoteCameraController("image", None, debug=args.debug)



try:
    socket_dict = {name:get_client_socket(pc_info["ip"], 5564) for name, pc_info in pc_info.items()}

    for pc_name, sock in socket_dict.items():
        threading.Thread(target=listen_socket, args=(pc_name, sock), daemon=True).start()
    
    while True:
        if not capture_event.is_set():
            time.sleep(0.01)
            continue
        print("Start Capture")
        
        cur_save_path = os.path.join(save_path, '%05d'%capture_idx)
        os.makedirs(cur_save_path, exist_ok=True)
        cur_save_abspath = str(Path.home()/cur_save_path)
        camera_controller.start(cur_save_path)
        camera_controller.end()
        time.sleep(0.5)
        # cur_state = {}
        while not os.path.exists(cur_save_abspath) or len(os.listdir(cur_save_abspath)) > len(cur_state) or len(cur_state)<10:
            time.sleep(0.5)
            
        if args.debug or args.vis_final:
            img_bucket = {}
            for i, serial_num in enumerate(serial_list):
                img_bucket[serial_num]=cv2.resize(cv2.imread(os.path.join(cur_save_abspath, f'{serial_num}.png')), dsize=(1024,768))
                if save_first_image:
                    cv2.imwrite(str(image_root_dir/f'{serial_num}.png'), img_bucket[serial_num])
            # if save_first_image:
            #     save_first_image = False

        start_time = time.time()
        ttl_output_dict = {}
        ttl_matchingset_list = {}
        
        for obj_name in args.obj_names:
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
                                shutil.move('./tmp/optim/rendered_pairs_optim.mp4', str(optimization_dir/f'rendered_pairs_optim_set{exising_set.idx}_{serial_num}_{midx}.mp4'))
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
            
            for matchingset in matchingset_list: 
                if len(matchingset.set) >= 3: # TODO: check thres
                    ttl_matchingset_list[f'{obj_name}_set{matchingset.idx}'] = matchingset




        ttl_output_dict = {}
        reoptim = False

        ttl_matchingset_list = dict(
            sorted(ttl_matchingset_list.items(), key=lambda x: len(x[1].set), reverse=True)
        )
        
        for matchingset_key, matchingset in ttl_matchingset_list.items(): 
            obj_name, set_idx = matchingset_key.split("_set")
            duplicate = False
            for prev_key in ttl_output_dict:
                if np.linalg.norm(ttl_output_dict[prev_key][:3,3]-matchingset.optim_T[:3,3]) < obj_dict_bucket[obj_name]['min_L']:
                    # Duplicate
                    duplicate = True
            if duplicate:
                continue
            
            if reoptim:
                firstitem = list(matchingset.set)[0]
                # Run Optimization onemore
                min_loss, optim_output = group_optimization(list(matchingset.set), matchingset.optim_T, \
                                                        scene, img_bucket, obj_dict_bucket[obj_name], loop_numb=30, stepsize=2, vis=True, use_ceres=True)
                print(f'loss:{loss} optim_ouptput:{optim_output}')
                
                if args.debug:
                    if min_loss is not None and optim_output is not None:
                        shutil.move('./tmp/optim/rendered_pairs_optim.mp4', str(set_root_dir/f'rendered_pairs_optim_set{matchingset.idx}_{set_idx}.mp4'))
                        valid = True if optim_output is not None else False
                        target_path = os.path.join(set_root_dir,f'set{matchingset.idx}_{valid}_loss_{min_loss}_match.jpeg')
                        shutil.copy('tmp_pairs.jpeg', target_path)
            if args.vis_final:
                target_path = os.path.join(set_root_dir,f'{obj_name}_set{matchingset.idx}_match.jpeg')
                highlights = {}
                for matchingitem in list(matchingset.set):
                    highlights[matchingitem.cam_id] = SRC_COLOR

                rendered_on_overlaid = combined_visualizer(matchingset.optim_T, scene, obj_dict_bucket[obj_name], list(matchingset.set), \
                                    img_bucket, highlights, DEVICE)
                cv2.imwrite(target_path, cv2.cvtColor(rendered_on_overlaid, cv2.COLOR_BGR2RGB))                        
            ttl_output_dict[f'{obj_name}_idx{len(ttl_output_dict)}'] = matchingset.optim_T
            
        print(f"Found object : { ','.join([key for key in ttl_output_dict])}")
        
        parsed_ttl_output_dict = {obj_name:{} for obj_name in args.obj_names}
        for key in ttl_output_dict:
            obj_name, sidx = key.split("_idx")
            parsed_ttl_output_dict[obj_name][len(parsed_ttl_output_dict[obj_name])] = ttl_output_dict[key]
        pickle.dump(parsed_ttl_output_dict, open(os.path.join(objoutput_path,'obj_T.pkl'),'wb'))
        end_time = time.time()
        
        print(f'Total Time for Frame {capture_idx}  : {end_time-start_time} sec')        
        capture_idx+=1
        cur_state = {}

        capture_event.clear()

finally:
    camera_controller.quit()        