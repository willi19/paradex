import os
from paradex.utils.io import shared_dir

processed_dir = os.path.join(shared_dir, "processed")
obj_list = os.listdir(processed_dir)

for obj_name in obj_list:
    index_list = os.listdir(os.path.join(processed_dir, obj_name))
    index_list.sort()
    for index in index_list:
        vid_list = os.listdir(os.path.join(processed_dir, obj_name, index, 'video'))
        if len(vid_list) != 24:
            print(len(vid_list), obj_name, index)