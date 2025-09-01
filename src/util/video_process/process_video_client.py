import os
from multiprocessing import Pool
import time
import numpy as np

from paradex.video.raw_video_processor import RawVideoProcessor
from paradex.video.raw_video import get_savepath, get_serialnum

from paradex.utils.file_io import shared_dir, load_camparam
from paradex.image.undistort import undistort_img
from paradex.image.aruco import detect_aruco
from paradex.io.capture_pc.raw_video_processor import RawVideoProcessorWithProgress

def process_frame(frame, info, fid):
    intrinsic = info
    frame = undistort_img(frame.copy(), intrinsic)
    data = {}
    
    return frame, data

def process_result(video_path, data_list, frame_ids):
    return
 
def load_info(video_path):
    save_path = get_savepath(video_path)
    serial_num = get_serialnum(video_path)
    
    cam_param_path = os.path.join(shared_dir, os.path.dirname(os.path.dirname(save_path)))
    cam_param = load_camparam(cam_param_path)[0][serial_num]
    
    return cam_param

start_time = time.time()
rvp = RawVideoProcessorWithProgress(process_frame=process_frame, load_info=load_info, process_result=process_result)
rvp.start()