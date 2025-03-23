import os
from paradex.utils.io import home_dir, capture_path_list    
import numpy as np
import shutil

for capture_path in capture_path_list:
    name_list = os.listdir(os.path.join(capture_path, "capture"))
    for name in name_list:
        index_list = os.listdir(os.path.join(capture_path, "capture", name))
        for index in index_list:
            if int(index) >= 3:
                shutil.copytree(os.path.join(capture_path, "capture", name, index), os.path.join(capture_path, "capture", name, str(int(index)-3)))
                shutil.rmtree(os.path.join(capture_path, "capture", name, index))
    
            