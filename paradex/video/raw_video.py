import cv2
import numpy as np
import os
import json
from multiprocessing import Pool, shared_memory, Manager, Value

from paradex.utils.upload_file import rsync_copy
from paradex.utils.path import shared_dir, home_path
from paradex.calibration.util import load_camparam
from paradex.image.undistort import precomute_undistort_map, apply_undistort_map

def get_videopath_list(video_dir):
    avi_path_list = []
    for root, _, files in os.walk(video_dir):
        for f in files:
            if f.endswith('.avi'):
                avi_full_path = os.path.join(root, f)
                avi_path_list.append(avi_full_path)
    
    return avi_path_list

def get_savepath(path):
    path = os.path.expanduser(path)
    home = os.path.expanduser(home_path)
    
    for cap_dir in ["captures1", "captures2"]:
        prefix = os.path.join(home, cap_dir)
        if path.startswith(prefix):
            relative = os.path.relpath(path, prefix)
            return os.path.join(relative)

def get_serialnum(video_path):
    return os.path.basename(video_path).split("-")[0]

def process_raw_video(video_path, frame_counter=None): # process_frame=None, preserve = True):
    serial_num = get_serialnum(video_path) # root_path / raw / video / *.avi
    
    root_path = os.path.dirname(os.path.dirname(video_path))
    nas_root_path = os.path.join(shared_dir, get_savepath(root_path))
    os.makedirs(os.path.join(nas_root_path, "videos"), exist_ok=True)
    
    out_path = os.path.join(os.path.dirname(video_path), f"{serial_num}_undistort.avi")
    nas_path = os.path.join(nas_root_path, "videos", f"{serial_num}.avi")

    if os.path.exists(nas_path) and not os.path.exists(out_path):
        print(f"✅ Undistorted video already exists: {nas_path}")
        return f"{video_path}:exists"
    
    if os.path.exists(nas_path) and os.path.exists(out_path):
        rsync_copy(out_path, nas_path, move=True)
        return f"{video_path}:success"

    try:
        intrinsic = load_camparam(nas_root_path)[0][serial_num]
        _, mapx, mapy = precomute_undistort_map(intrinsic)
            
    except:
        return f"{video_path}:no_camparam"
    
    if os.path.exists(out_path): # end in the middle of previous run
        os.remove(out_path)
        
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    fourcc = cv2.VideoWriter_fourcc(*'MJPG')  # <-- 프레임 단위 압축
    
    out = cv2.VideoWriter(out_path, fourcc, fps, (w, h))
    last_frame = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        last_frame += 1
        undistorted = apply_undistort_map(frame, mapx, mapy)
        
        if frame_counter is not None:      
            frame_counter.value = last_frame

        out.write(undistorted)
        
    cap.release()
    out.release()

    rsync_copy(out_path, nas_path, move=True) 
       
    os.remove(video_path)  # 원본 파일 삭제
    
    return f"{video_path}:success"


class RawVideoProcessor():
    def __init__(self, save_path):
        self.save_path = save_path
        self.manager = Manager()
        
        self.log = []
        self.video_path_list = get_videopath_list(f"{home_path}/captures1/{save_path}") + \
                                get_videopath_list(f"{home_path}/captures2/{save_path}")
        
        self.total_frames = {}
        
        for vid_path in self.video_path_list:
            cap = cv2.VideoCapture(vid_path)
            if not cap.isOpened() or int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) == 0:
                print(f"Invalid video file: {vid_path}")
            self.total_frames[vid_path] = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        self.frame_counter = {vid_path:self.manager.Value('i', 0) for vid_path in self.video_path_list}
        self.pool = Pool()
        self.process()
            
    def async_callback(self, result):
        self.log.append(result)
        
    def error_callback(self, e):
        self.log.append("ERROR in process:", e)
        
    def process(self):
        self.process_list = [self.pool.apply_async(process_raw_video, args=(vid_path, self.frame_counter[vid_path]), callback=self.async_callback, error_callback=self.error_callback)
                        for vid_path in self.video_path_list]    
       
    def finished(self):
        return all(r.ready() for r in self.process_list)

    