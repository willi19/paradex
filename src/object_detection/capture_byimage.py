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


from paradex.utils.env import get_pcinfo, get_serial_list

from paradex.io.capture_pc.camera_main import RemoteCameraController
from paradex.io.capture_pc.util import get_client_socket
from paradex.io.capture_pc.connect import git_pull, run_script
from paradex.io.signal_generator.UTGE900 import UTGE900

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
parser.add_argument('--default_rescale', type=float, default=0.5)
parser.add_argument('--loss_thres', type=float, default=12)
parser.add_argument('--toggle', action='store_true')
args = parser.parse_args()
inliers_threshold = 20
cam_numb_thres = 18

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

saved_corner_img = {serial_num:np.ones((1536, 2048, 3), dtype=np.uint8)*255 for serial_num in serial_list}

cur_state = {}
cur_numinput = None
    
capture_idx = 0
filename = time.strftime("%Y%m%d_%H%M%S", time.localtime())

scene = MultiCamScene(rescale_factor=args.default_rescale, device=DEVICE, height=1536, width=2048)
scene.get_batched_renderer(tg_cam_list=scene.cam_ids)
DEBUG_VIS = Path('./debug')
os.makedirs(DEBUG_VIS, exist_ok=True)
OUTPUTDIR = './objoutput'
os.makedirs(OUTPUTDIR, exist_ok=True)

signal_generator = UTGE900()
signal_generator.generate(freq=4000) # 100 frequency > 10Hz 1000 > 1Hz , 2000 > 0.5Hz

    
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
                if cur_tg_frame==-1:
                    cur_tg_frame = 0
                cur_state[serial_num] = matching_output
            print(f"Frame {frame} got total {len(cur_state)} inputs")
        else:
            print(f"[{pc_name}] Unknown JSON type: {data.get('type')}")

pc_list = list(pc_info.keys())
git_pull("merging", pc_list)

# if args.debug:
#     run_script(f"python paradex/object_detection/client.py --obj_name {args.obj_name} --saveimg", pc_list, log=True)
# else:
#     run_script(f"python paradex/object_detection/client.py --obj_names {' '.join(args.obj_names)}", pc_list, log=True)

camera_controller = RemoteCameraController("image", None, debug=args.debug)

save_path = './tmp_images'
if os.path.exists(save_path):
    shutil.rmtree(save_path)
os.makedirs(save_path, exist_ok=True)

if args.toggle:
    signal_generator.off(1)
else:
    signal_generator.on(1)

try:
    socket_dict = {name:get_client_socket(pc_info["ip"], 5564) for name, pc_info in pc_info.items()}

    for pc_name, sock in socket_dict.items():
        threading.Thread(target=listen_socket, args=(pc_name, sock), daemon=True).start()
    
    processing_frame = 1
    while True:
        camera_controller.start(os.path.join(save_path, '%05d'%capture_idx))
        camera_controller.end()
        time.sleep(0.5)  # wait for file to be fully written
        cur_state = {}

        
        
finally:
    camera_controller.quit()        
    if args.toggle:
        signal_generator.on(1)
    else:
        signal_generator.off(1)
    signal_generator.quit()