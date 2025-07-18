import cv2
import os
from paradex.video.convert_codec import change_to_h264

def process_video(video_path, out_path, load_info, process_frame):
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
        # how to know video end
        processed_frame = process_frame(frame, video_path, fid, data)

        out.write(processed_frame)
        fid += 1
    
    cap.release()
    out.release()
    
    change_to_h264(tmp_path, out_path)

def process_video_list(video_dir, out_path, process_frame):
    video_name_list = os.listdir(video_dir)    
    cap_dict = {}
    finished = {}
    
    fourcc = cv2.VideoWriter_fourcc(*'x264')  # <-- 프레임 단위 압축
    
    
    for video_name in video_name_list:
        name = video_name.split(".")[0]
        cap_dict[name] = cv2.VideoCapture(os.path.join(video_dir, video_name))
        finished[name] = False
        
        fps = cap_dict[name].get(cv2.CAP_PROP_FPS)
        w = int(cap_dict[name].get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap_dict[name].get(cv2.CAP_PROP_FRAME_HEIGHT))
        
    out = cv2.VideoWriter(out_path, fourcc, fps, (w, h))
    fid = 0
    while True:
        img_dict = {}
        for name, cap in cap_dict.items():
            if finished[name]:
                continue
            
            ret, frame = cap.read()
            if not ret:
                finished[name] = True
                continue
                
            img_dict[name] = frame.copy()
        
        if len(img_dict) == 0:
            break
        
        frame = process_frame(process_frame, video_dir, fid)
        out.write(frame)
        fid += 1
    
    for _, cap in cap_dict.items():
        cap.release()
        
    out.release()