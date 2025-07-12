import json
from paradex.utils.file_io import config_dir, shared_dir
import os

def get_pcinfo():
    pc_info = json.load(open(os.path.join(config_dir, "environment", "pc.json"), "r"))
    return pc_info

def get_serial_list():
    pc_info = get_pcinfo()
    serial_list = []
    for pc in pc_info.keys():
        serial_list.extend(pc_info[pc]['cam_list'])
        
def get_network_info():
    network_info = json.load(open(os.path.join(config_dir, "environment", "network.json"), "r"))
    return network_info