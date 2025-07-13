import os
from paradex.utils.file_io import shared_dir
import shutil

cam_param_dir = os.path.join(shared_dir, "extrinsic")
dir = "config"

intrinsic_dict = {}
extrinsic_dict = {}

if __name__ == "__main__":
    name_list = ["20250713_192523", "20250713_195450", "20250713_200124", "20250713_205250"]
    dest_dir = os.path.join(cam_param_dir, "20250713_205300")
    
    offset = 0
    
    for name in name_list:
        root_dir = os.path.join(cam_param_dir, name)
        for idx in os.listdir(root_dir):
            shutil.copytree(os.path.join(root_dir, str(idx)), os.path.join(dest_dir, str(int(idx)+offset)))
        offset += len(os.listdir(root_dir))