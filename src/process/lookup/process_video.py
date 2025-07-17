import os
from multiprocessing import Pool
import time
import numpy as np

from paradex.video.raw_video_processor import RawVideoProcessor
from paradex.video.raw_video import get_savepath, get_serialnum

from paradex.utils.file_io import shared_dir, load_camparam
from paradex.image.undistort import undistort_img
from paradex.image.aruco import detect_aruco

def process_frame(frame, info, fid):
    intrinsic = info
    frame = undistort_img(frame.copy(), intrinsic)
    
    undist_kypt, ids = detect_aruco(frame)
    data = {}
    
    for id, corner in zip(ids, undist_kypt):
        data[int(id)] = corner.squeeze()
            
    return frame, data

def process_result(video_path, data_list, frame_ids):
    save_path = os.path.join(os.path.dirname(os.path.dirname(video_path)), "marker2D")
    file_name = get_serialnum(video_path) + ".npy"
    
    result = {fid:data_list[i] for i, fid in enumerate(frame_ids)}
    np.save(os.path.join(save_path, file_name), result)

def load_info(video_path):
    save_path = get_savepath(video_path)
    serial_num = get_serialnum(video_path)
    
    cam_param_path = os.path.join(shared_dir, os.path.dirname(os.path.dirname(save_path)))
    cam_param = load_camparam(cam_param_path)[0][serial_num]
    
    return cam_param

start_time = time.time()
rvp = RawVideoProcessor("capture_/lookup", process_frame=process_frame, load_info=load_info, process_result=process_result, overwrite=True)
while not rvp.finished():
    time.sleep(0.01)
    
print("consumed time : ", time.time()-start_time)
for msg in rvp.log:
    print(msg)