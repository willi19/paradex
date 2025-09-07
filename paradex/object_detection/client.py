import json
import os
import time
import torch
import numpy as np
from pathlib import Path
import cv2
HOME_PATH = Path.home()

from paradex.utils.file_io import config_dir
from paradex.image.aruco import detect_charuco, merge_charuco_detection
from paradex.io.capture_pc.util import get_server_socket
from paradex.io.capture_pc.camera_local import CameraCommandReceiver
from paradex.utils.file_io import shared_dir

# Insert object name
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--obj_name', type=str, required=True)
parser.add_argument('--debug', action='store_true')
args = parser.parse_args()
obj_name = args.obj_name

from paradex.object_detection.multiview_utils.template import Template
from paradex.object_detection.object_optim_config import template_path
from paradex.object_detection.multiview_utils.matcher import MatcherTo3D

# TODO: check cuda device work
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
template = Template(template_path[obj_name], obj_name=obj_name)
matcherto3d = MatcherTo3D(DEVICE, img_L=256)
paircount_threshold=40

from paradex.model.yolo_world_module import YOLO_MODULE
mask_detector = YOLO_MODULE(categories=obj_name, device=DEVICE)

if not args.debug:
    camera_loader = CameraCommandReceiver()
    ident = camera_loader.ident
    serial_list = camera_loader.camera.serial_list
    socket = get_server_socket(5564)
else:
    from paradex.io.camera.camera_loader import CameraManager
    camera = CameraManager("image")
    num_cam = camera.num_cameras
    serial_list = camera.serial_list

    NAS_IMG_SAVEDIR = Path(shared_dir)/'current_img'
    os.makedirs(NAS_IMG_SAVEDIR, exist_ok=True)

board_info = json.load(open(os.path.join(config_dir, "environment", "charuco_info.json"), "r"))
num_cam = len(serial_list)
last_frame_ind = [-1 for _ in range(num_cam)]

frame_id = -1
while (not args.debug and not camera_loader.exit) or (args.debug):
    if args.debug:
        save_path = './tmp'
        os.makedirs(save_path, exist_ok=True)
        camera.start(save_path)
        camera.wait_for_capture_end()
    
    if args.debug:
        st_time = time.time()
    
    for i, serial_num in enumerate(serial_list):
        # print(f"Processing camera {serial_num}...")
        if not args.debug:
            frame_id = camera_loader.camera.get_frameid(i)
            data = camera_loader.camera.get_data(i)
            last_frame_ind[i] = data["frameid"]
            last_image = data["image"]
            if frame_id == last_frame_ind[i]:
                print(f"[{serial_num}] Frame {frame_id} already processed.")
                continue
        else:
            last_image = cv2.cvtColor(cv2.imread(os.path.join(save_path, f'{serial_num}.png')), cv2.COLOR_BGR2RGB)
            cv2.imwrite(NAS_IMG_SAVEDIR/f'{serial_num}.jpeg', last_image)
        print(f"Before Mask Detection: Camera {serial_num} capturing frame {last_frame_ind[i]}")
        last_image = cv2.resize(last_image, dsize=template.img_template[serial_num].shape[:2][::-1])
        detections = mask_detector.process_img(last_image, top_1=False)
        result_dict = {}  
        print("after detection", len(detections.box))

        ttl_pair_count = 0
        for midx, tg_mask in enumerate(detections.mask):
            tg_mask = np.repeat(tg_mask[..., None], 3, axis=2).astype(np.int64)*255.0
    
            src_3d_dict, tg_2d_dict, org_2d_dict = \
                matcherto3d.match_img2template(last_image, tg_mask, \
                                            template, paircount_threshold, batch_size=24, \
                                            draw=False, use_crop=True)
            
            pair_count = 0
            src_3d_points = []
            tg_2d_points = []
            src_cam_ids = []
            
            for src_cam_id in src_3d_dict:
                if len(src_3d_dict[src_cam_id])>0:
                    pair_count+=len(src_3d_dict[src_cam_id])
                    src_3d_points.append(src_3d_dict[src_cam_id])
                    tg_2d_points.append(tg_2d_dict[src_cam_id])
                    src_cam_ids.append([src_cam_id]*len(src_3d_dict[src_cam_id]))
                        
            if pair_count > 0:
                ttl_pair_count+=pair_count
                src_3d_points = np.vstack(src_3d_points).astype(np.float64)
                tg_2d_points = np.vstack(tg_2d_points).astype(np.float64)
                
                combined_src_3d = src_3d_points # combined array
                combined_tg_2d = tg_2d_points # combined_array
                src_arr_cam_ids = np.hstack(src_cam_ids)
                result = {'count':pair_count,'combined_src_3d':combined_src_3d.tolist(), \
                    'combined_tg_2d':combined_tg_2d.tolist(),\
                    'src_arr_cam_ids':src_arr_cam_ids.tolist()}
                # TODO send image and mask
            else:
                result = {'count':0}
                
            result_dict[midx] = result

        msg_dict = {
            "frame": int(last_frame_ind[i]),
            "detect_result": result_dict,
            "type": "2D_matching",
            "serial_num": serial_num}
        
        print(f"[{serial_num}] Frame {last_frame_ind[i]}: {ttl_pair_count} detected.")
    
        msg_json = json.dumps(msg_dict)
        if not args.debug:
            socket.send_multipart([ident, msg_json.encode()])

    # print(f"End of frame id {frame_id}")

    if args.debug:
        ed_time = time.time()
        print(f"Time for matchng in one desktop {ed_time-st_time}")
        
    time.sleep(0.01)