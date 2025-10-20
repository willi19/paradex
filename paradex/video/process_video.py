import cv2
import os
from paradex.video.convert_codec import change_to_h264
import numpy as np

def process_video(video_path, out_path, load_info, process_frame):
    print(video_path)
    data = load_info(video_path)
    
    cap = cv2.VideoCapture(video_path)
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    tmp_path = os.path.join(os.path.dirname(out_path), os.path.basename(out_path).split(".")[0]+"_tmp.avi")
    out = cv2.VideoWriter(tmp_path, fourcc, fps, (w, h))
    
    fid = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        processed_frame = process_frame(frame, video_path, fid, data)
        
        out.write(processed_frame)
        fid += 1
    cap.release()
    out.release()
    
    change_to_h264(tmp_path, out_path)

def process_video_list(video_dir, out_path, data, process_frame):
    video_name_list = os.listdir(video_dir)    
    cap_dict = {}
    finished = {}
    
    fourcc = cv2.VideoWriter_fourcc(*'MJPG')  # <-- 프레임 단위 압축
    tmp_path = os.path.join(os.path.dirname(out_path), os.path.basename(out_path).split(".")[0]+"_tmp.avi")
    
    for video_name in video_name_list:
        name = video_name.split(".")[0]
        cap_dict[name] = cv2.VideoCapture(os.path.join(video_dir, video_name))
        finished[name] = False
        fps = cap_dict[name].get(cv2.CAP_PROP_FPS)
        w = int(cap_dict[name].get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap_dict[name].get(cv2.CAP_PROP_FRAME_HEIGHT))
        
    out = cv2.VideoWriter(tmp_path, fourcc, fps, (w, h))
    fid = 0
    
    while True:
        img_dict = {}
        cnt = 0
        for name, cap in cap_dict.items():
            if finished[name]:
                img_dict[name] = np.zeros((h, w, 3))
                continue
            
            ret, frame = cap.read()
            if not ret:
                finished[name] = True
                img_dict[name] = np.zeros((h, w, 3))
                continue
                
            img_dict[name] = frame.copy()
            cnt += 1
        
        if cnt == 0:
            break
        
        frame = process_frame(img_dict, video_dir, fid, data)
        frame = cv2.resize(frame, (w, h))
    
        out.write(frame)
        fid += 1
        print(fid)
    for _, cap in cap_dict.items():
        cap.release()
        
    out.release()