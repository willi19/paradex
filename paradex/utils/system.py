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