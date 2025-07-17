import cv2
import numpy as np
import os
import json

from paradex.utils.upload_file import copy_file
from paradex.utils.file_io import shared_dir, home_path

magic_number = 5

def check_valid(timestamp):
    assert "frameID" in timestamp and "timestamps" in timestamp
    
    fid_array = np.array(timestamp["frameID"])
    fid_diff = fid_array[1:] - fid_array[:-1]
    
    ts_array = np.array(timestamp["timestamps"])
    ts_diff = ts_array[1:] - ts_array[:-1]
    
    interval = ts_diff / fid_diff
    if np.size(interval) == 0:
        print("no timestamp")
        return False
    
    if np.max(interval[magic_number:]) > np.min(interval[magic_number:]) * 1.5:
        print(interval)
        print(np.min(interval), np.max(interval))
        return False
    
    return True

def get_timestamp_path(video_path):
    video_name = os.path.basename(video_path).split("-")[0]
    video_dir = os.path.dirname(video_path)
    
    timestamp_file_name = video_name + "_timestamps.json"
    timestamp_path = os.path.join(video_dir, timestamp_file_name)
    return timestamp_path

def load_timestamp(video_path):
    timestamp_path = get_timestamp_path(video_path)
    timestamp = json.load(open(timestamp_path))
    return timestamp

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

def fill_dropped_frames(video_path, load_info, process_frame, process_result, preserve, overwrite, frame_counter=None): # process_frame=None, preserve = True):
    timestamp_path = get_timestamp_path(video_path)
    serial_num = get_serialnum(video_path)
    out_path = os.path.join(os.path.dirname(video_path), f"{serial_num}.avi")
    nas_path = os.path.join(shared_dir, get_savepath(out_path))
    
    data_list = []
    
    if os.path.exists(nas_path) and not overwrite:
        return f"{video_path}:already exist"
    try:
        info = load_info(video_path)
    except Exception as e:
        return f"{video_path}: {e} error during loading info"            
    
    timestamp_dict = json.load(open(timestamp_path))
    frame_ids = np.array(timestamp_dict["frameID"])
    
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    fourcc = cv2.VideoWriter_fourcc(*'MJPG')  # <-- 프레임 단위 압축
    
    out = cv2.VideoWriter(out_path, fourcc, fps, (w, h))
    
    last_frame = magic_number

    for fid in frame_ids:
        if last_frame >= fid:
            continue
        
        while last_frame + 1 < fid:
            black_frame = np.zeros((h, w, 3), dtype=np.uint8)
            out.write(black_frame)
            last_frame += 1
            
        ret, frame = cap.read()
        last_frame += 1
        if process_frame is not None:
            try:
                frame, data = process_frame(frame, info, fid)
            except Exception as e:
                return f"{video_path}:{str(e)} during processing frame {last_frame}"
            data_list.append(data)
        
        if frame_counter is not None:      
            frame_counter = last_frame
        out.write(frame)
    cap.release()
    out.release()

    if process_result is not None:
        try:
            process_result(video_path, data_list)
        except Exception as e:
            return f"{video_path}:{str(e)} during processing result"
            
    if not preserve:
        os.remove(video_path)
        os.remove(timestamp_path)
        
    # upload video and remove previous one if preserve is False
    copy_file(out_path, nas_path)
    os.remove(out_path)
    return f"{video_path}:success"
    