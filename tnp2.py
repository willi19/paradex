import os
import json

from paradex.utils.file_io import config_dir, shared_dir
from paradex.utils.env import get_pcinfo
from paradex.video.raw_video import *

vid_list = get_videopath_list(os.path.join(shared_dir, "tmp/tmp/0"))

for video_path in vid_list:
    
    timestamp_path = get_timestamp_path(video_path)
    if not os.path.exists(timestamp_path):
        print(f"{video_path} stream terminated unexpectedly")
        continue
    try:
        timestamp = json.load(open(timestamp_path))
    except:
        print(f"{video_path} : sync or lan cable connection unstable")
        continue
    
    if not check_valid(timestamp):
        print(f"{video_path} : sync or lan cable connection unstable")
        continue
    
    print(video_path, "success")