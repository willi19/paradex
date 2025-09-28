import json
import os
import time
import torch
import numpy as np
from pathlib import Path
import cv2
HOME_PATH = Path.home()

from paradex.utils.file_io import config_dir
from paradex.io.capture_pc.util import get_server_socket
from paradex.io.capture_pc.camera_local import CameraCommandReceiver
from paradex.utils.file_io import shared_dir

# Insert object name
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--obj_names', type=str, nargs='+')
parser.add_argument('--debug', action='store_true')
parser.add_argument('--saveimg', action='store_true')
parser.add_argument('--paircount', type=int, default=30)
args = parser.parse_args()
obj_names = args.obj_names

from paradex.object_detection.multiview_utils.template import Template
from paradex.object_detection.object_optim_config import template_path
from paradex.object_detection.multiview_utils.matcher import MatcherTo3D
from paradex.object_detection.default_config import default_template, name2prompt, prompt2name
from paradex.model.yolo_world_module import YOLO_MODULE

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Template and Yolo Model
template_dict = {}
combined_yoloprompts = []
for obj_name in args.obj_names:
    tmp_template = Template(str(default_template[obj_name]), obj_name=obj_name) # template has mask and rgb image. 
    template_dict[obj_name] = tmp_template
    combined_yoloprompts.append(name2prompt[obj_name])
combined_yoloprompts = ','.join(combined_yoloprompts)
mask_detector = YOLO_MODULE(categories=combined_yoloprompts, device=DEVICE, use_pretrained=('ramen' in combined_yoloprompts))
# Matcher
matcherto3d = MatcherTo3D(device=DEVICE, img_L=256)
paircount_threshold=args.paircount

# Camera
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
# Save Img Dir
NAS_IMG_SAVEDIR = Path(shared_dir)/'current_img'
os.makedirs(NAS_IMG_SAVEDIR, exist_ok=True)

print(f"Listening to cameras: {serial_list}")
num_cam = len(serial_list)
last_frame_ind = [-1 for _ in range(num_cam)]

save_path = './tmp_images'

next_frame_idx = 1
while (not args.debug and not camera_loader.exit) or (args.debug):
    # Check new image
    if os.path.exists(os.path.join(save_path, '%05d'%next_frame_idx)):
        for file in os.listdir(os.path.join(save_path, '%05d'%next_frame_idx)):
            print(file)
            
        # last_image = cv2.imread(os.path.join(save_path, f'{next_frame_idx}', 'image.jpeg'))
        # next_frame_idx += 1

    # st_time = time.time()
    
    # ttl_pair_count_perpc = 0
    # for i, serial_num in enumerate(serial_list):
    #     print(f"Processing camera {serial_num}...")

    #     last_image = cv2.resize(last_image, dsize=tmp_template.img_template[serial_num].shape[:2][::-1])
    #     # cv2.imwrite(str(NAS_IMG_SAVEDIR/f'{serial_num}.jpeg'), last_image)
    #     # cv2.imwrite(str(NAS_IMG_SAVEDIR/f'frame_{serial_num}_{int(last_frame_ind[i]%10)}.jpeg'), last_image)
    #     detections, output_image = mask_detector.process_img(last_image, top_1=False, draw_mask=args.debug)
    #     print(f"[{serial_num}] Frame {last_frame_ind[i]}: {len(detections)} detected before filtering.")
    #     result_dict = {}  

    #     ttl_pair_count = 0
    #     for obj_name in args.obj_names:
    #         tmp_result_dict = {}
    #         if len(detections) > 0:
    #             tg_detections = detections[detections['class_name']==name2prompt[obj_name]]
    #             for midx, tg_mask in enumerate(tg_detections.mask):
    #                 tg_mask = np.repeat(tg_mask[..., None], 3, axis=2).astype(np.int64)*255.0
    
    #                 src_3d_dict, tg_2d_dict, org_2d_dict = \
    #                     matcherto3d.match_img2template(last_image, tg_mask, \
    #                                                 template_dict[obj_name], paircount_threshold, batch_size=24, \
    #                                                 draw=args.saveimg, use_crop=True, image_name=str(NAS_IMG_SAVEDIR/f'matching_{serial_num}_{midx}.png'))
    #                 pair_count = 0
    #                 src_3d_points = []
    #                 tg_2d_points = []
    #                 src_cam_ids = []
    #                 for cam_id in src_3d_dict:
    #                     if len(src_3d_dict[cam_id])>0:
    #                         pair_count+=len(src_3d_dict[cam_id])
    #                         src_3d_points.append(src_3d_dict[cam_id])
    #                         tg_2d_points.append(tg_2d_dict[cam_id])
    #                         src_cam_ids.append([cam_id]*len(src_3d_dict[cam_id]))
                            
    #                 if pair_count>0:
    #                     src_3d_points = np.vstack(src_3d_points).astype(np.float64)
    #                     tg_2d_points = np.vstack(tg_2d_points).astype(np.float64)
    #                     combined_src_3d = src_3d_points # combined array
    #                     combined_tg_2d = tg_2d_points # combined_array
    #                     src_arr_cam_ids = np.hstack(src_cam_ids)
            
    #                     result = {'count':pair_count,'combined_src_3d':combined_src_3d.tolist(), \
    #                         'combined_tg_2d':combined_tg_2d.tolist(),\
    #                         'src_arr_cam_ids':src_arr_cam_ids.tolist()}
    #                 else:
    #                     result = {'count':0}
                        
    #                 tmp_result_dict[midx] = result
    #         result_dict[obj_name] = tmp_result_dict
    #         ttl_pair_count = sum([tmp_result_dict[k]['count'] for k in tmp_result_dict])

    #     msg_dict = {
    #         "frame": int(last_frame_ind[i]),
    #         "detect_result": result_dict,
    #         "type": "2D_matching",
    #         "serial_num": serial_num}
        
    #     if args.debug:
    #         print(f"[{serial_num}] Frame {last_frame_ind[i]}: {ttl_pair_count} detected.")
    #     ttl_pair_count_perpc += ttl_pair_count
    
    #     msg_json = json.dumps(msg_dict)
    #     if not args.debug:
    #         socket.send_multipart([ident, msg_json.encode()])

    # ed_time = time.time()
    # print(f"Time for matchng in one desktop {ed_time-st_time} total pair count {ttl_pair_count_perpc}")
        
    # time.sleep(0.01)