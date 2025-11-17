import json
import os

config_dir = os.path.join(os.path.dirname(__file__), "..", "..", "system", "current")
pc_info = json.load(open(os.path.join(config_dir, "pc.json"), "r"))

def get_pc_ip(pc_name: str) -> str:
    return pc_info[pc_name]["ip"]

def get_camera_list(pc_name: str):
    return pc_info[pc_name]["camera_list"]

def get_pc_list():
    return list(pc_info.keys())

def get_camera_config():    
    cam_config_path = os.path.join(config_dir, "camera.json")
    if os.path.exists(cam_config_path):
        cam_info = json.load(open(cam_config_path, "r"))
    else:
        cam_info = {}
    return cam_info

def load_device(device_type, device_name):
    pass

def load_devices(config_name: str):
    capture_config_path = os.path.join(config_dir, "dataset_acquisition", f"{config_name}.json")
    if os.path.exists(capture_config_path):
        capture_info = json.load(open(capture_config_path, "r"))
    else:
        capture_info = {}
    return capture_info