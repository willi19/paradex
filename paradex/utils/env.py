import json
from paradex.utils.file_io import config_dir, shared_dir
import os

def get_serial_list():
    pc_info = json.load(open(os.path.join(config_dir, "environment", "pc.json"), "r"))
    serial_list = []
    for pc in pc_info.keys():
        serial_list.extend(pc_info[pc]['cam_list'])