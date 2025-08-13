import os
import json

from paradex.utils.file_io import config_dir, shared_dir
from paradex.utils.env import get_pcinfo

ind_list = os.listdir(os.path.join(shared_dir, "capture_/lookup/pringles/lay_allegro"))

for ind in ind_list:
    vid_list = os.listdir(os.path.join(shared_dir, f"capture_/lookup/pringles/lay_allegro/{ind}/videos"))

    pc_list = get_pcinfo()
    for pc_name, info in pc_list.items():
        for cam_name in info['cam_list']:
            cam_vid_name = str(cam_name) + ".avi"
            if cam_vid_name not in vid_list:
                print(pc_name, cam_name)
