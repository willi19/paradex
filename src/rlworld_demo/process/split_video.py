import os
import cv2
import numpy as np
import tqdm

from paradex.utils.file_io import shared_dir

def process_video(video_path, save_path):
    cap = cv2.VideoCapture(video_path)
    num_frame = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    os.makedirs(f"{save_path}", exist_ok=True)
    
    for i in tqdm.tqdm(range(num_frame), desc=f"Processing {os.path.basename(video_path)}"):
        ret, frame = cap.read()
        if i < 18000:
            continue
        if not ret:
            error = True
            break
        cv2.imwrite(f"{save_path}/{i:06d}.png", frame)
    cap.release()

for index in range(1):
    # multiview
    video_path = os.path.join(shared_dir, "capture", f"final_final_simulate",f"{index}", "25305466.avi")
    save_dir = os.path.join("image", str(index), "multiview")
    process_video(video_path, save_dir)

    # cam1
    vid_name = None
    for name in os.listdir(os.path.join("RLWRLD_DEMO", str(index))):
        if name != "multiview" and name != "iphone1.mov":
            vid_name = name
            video_path = os.path.join("RLWRLD_DEMO", str(index), vid_name)
            save_dir = os.path.join("image", str(index), vid_name)
            process_video(video_path, save_dir)