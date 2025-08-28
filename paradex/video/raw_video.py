import cv2
import numpy as np
import os
import json

from paradex.utils.upload_file import copy_file
from paradex.utils.file_io import shared_dir, home_path

import bisect

magic_number = 5
td = 2 / 30

def fill_framedrop(cam_timestamp):
    frameID = cam_timestamp["frameID"]
    real_start = -1
    for i, fi in enumerate(frameID):
        if fi == 5:
            real_start = i
    
    frameID = frameID[real_start:]
    pc_time = np.array(cam_timestamp["pc_time"])[real_start:]
    timestamp = np.array(cam_timestamp["timestamps"])

    time_delta = (pc_time[-1] - pc_time[0]) / (frameID[-1] - frameID[0])
    offset = np.mean(pc_time - (np.array(frameID)-1)*time_delta)
    pc_time_nodrop = []
    frameID_nodrop = []

    time_delta_new = 1 / 30
    
    if time_delta / time_delta_new > 1.01:
        return None, None
    
    for i in range(1, frameID[-1] + 10):
        frameID_nodrop.append(i)
        pc_time_nodrop.append((i-1)*time_delta_new+offset - td)
    
    return pc_time_nodrop, frameID_nodrop

def get_synced_data(pc_times, data, data_times):
    """
    2-pointer 방식으로 pc_times와 가장 가까운 data_times의 데이터를 매칭
    """
    synced_data = []
    n = len(pc_times)
    m = len(data_times)

    i = 0  # pc_times pointer
    j = 0  # data_times pointer

    while i < n:
        # data_times[j]가 pc_time[i]보다 작으면 j를 앞으로
        while j + 1 < m and abs(data_times[j + 1] - pc_times[i]) <= abs(data_times[j] - pc_times[i]):
            j += 1
        synced_data.append(data[j])
        i += 1

    return np.array(synced_data)


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
        # if not preserve:
        #     os.remove(video_path)
        #     os.remove(timestamp_path)
        return f"{video_path}: {e} error during loading info"            
    
    timestamp_dict = json.load(open(timestamp_path))
    frame_ids = np.array(timestamp_dict["frameID"])
    
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    fourcc = cv2.VideoWriter_fourcc(*'MJPG')  # <-- 프레임 단위 압축
    
    out = cv2.VideoWriter(out_path, fourcc, fps, (w, h))
    
    last_frame = 0

    for fid in frame_ids:
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
            frame_counter.value = last_frame
            
        out.write(frame)
    cap.release()
    out.release()

    if process_result is not None:
        try:
            process_result(video_path, data_list, frame_ids)
        except Exception as e:
            return f"{video_path}:{str(e)} during processing result"
            
    if not preserve:
        os.remove(video_path)
        os.remove(timestamp_path)
        
    # upload video and remove previous one if preserve is False
    copy_file(out_path, nas_path)
    os.remove(out_path)
    return f"{video_path}:success"
    