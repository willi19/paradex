from paradex.video.raw_video_processor import RawVideoProcessor
from paradex.video.raw_video import get_savepath, get_serialnum

from paradex.utils.file_io import shared_dir, load_camparam
from paradex.image.undistort import undistort_img
import os
from multiprocessing import Pool
import time

def process_frame(frame, info, fid):
    intrinsic = info
    frame = undistort_img(frame.copy(), intrinsic)
    return frame, None

def process_result():
    pass

def load_info(video_path):
    save_path = get_savepath(video_path)
    serial_num = get_serialnum(video_path)
    
    cam_param_path = os.path.join(shared_dir, os.path.dirname(os.path.dirname(save_path)))
    cam_param = load_camparam(cam_param_path)[0][serial_num]
    
    return cam_param

start_time = time.time()
rvp = RawVideoProcessor("capture_/hri_teleop", process_frame=process_frame, load_info=load_info, overwrite=False, preserve=True)
while not rvp.finished():
    time.sleep(0.01)
    
print("consumed time : ", time.time()-start_time)
for msg in rvp.log:
    print(msg)