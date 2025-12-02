import json
import os

config_dir = os.path.join(os.path.dirname(__file__), "..", "..", "system", "current")
pc_info = json.load(open(os.path.join(config_dir, "pc.json"), "r"))
network_info = json.load(open(os.path.join(config_dir, "network.json"), "r"))
pc_name = os.path.basename(os.path.expanduser("~"))

def get_pc_ip(pc_name: str) -> str:
    return pc_info[pc_name]["ip"]

def get_camera_list(pc_name: str = pc_name) -> list:
    return pc_info[pc_name]["cam_list"]

def get_pc_list() -> list:
    return list(pc_info.keys())

def get_camera_config() -> dict:
    cam_config_path = os.path.join(config_dir, "camera.json")
    if os.path.exists(cam_config_path):
        cam_info = json.load(open(cam_config_path, "r"))
    else:
        cam_info = {}
    return cam_info