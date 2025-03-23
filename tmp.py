import os
from paradex.utils.io import home_dir, capture_path_list    
import shutil

for capture_path in capture_path_list:
    capture_dir = os.path.join(capture_path, "capture")
    name_list = os.listdir(capture_dir)

    for name in name_list:
        target_path = os.path.join(capture_dir, name)
        index_list = os.listdir(target_path)

        # Filter to ensure only directories with numeric names are considered
        index_list = [idx for idx in index_list if idx.isdigit()]
        
        # Sort the index list by integer value (or use creation time if needed)
        index_list.sort(key=lambda x: int(x))

        # Temporary renaming to avoid conflicts
        temp_names = {}
        for i, old_index in enumerate(index_list):
            src = os.path.join(target_path, old_index)
            temp = os.path.join(target_path, f"temp_{i}")
            os.rename(src, temp)
            temp_names[temp] = os.path.join(target_path, str(i))

        # Rename temp folders to final names
        for temp, final in temp_names.items():
            os.rename(temp, final)
