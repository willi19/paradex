import os
import cv2
import numpy as np
import tqdm

from paradex.utils.file_io import shared_dir

def process_video(video_path, save_path):
    cap = cv2.VideoCapture(video_path)
    out = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), 10, (2048, 1536))

    prev_frame = None
    error = False
    num_frame = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    for _ in tqdm.tqdm(range(num_frame), desc=f"Processing {os.path.basename(video_path)}"):
        ret, frame = cap.read()
        if not ret:
            error = True
            break

        if np.sum(frame) < 10 and prev_frame is not None:
            out.write(prev_frame)
        else:
            out.write(frame)
            prev_frame = frame

    out.release()
    cap.release()
    if error:
        os.remove(save_path)

nas_dir = [(0, 1), (1, 0), (1, 1), (1, 2), (1, 3), (1, 6), (1, 8), (1, 9), (1, 11), (1, 12),(2, 0), (2, 1), (4, 0), (4, 1), (4, 2), (4, 3), (4, 4), (4, 5), (4, 6), (4, 7)]
for index, (capture_idx, capture_num) in enumerate(nas_dir):

    video_dir = os.path.join(shared_dir, "capture", f"resimulate_{capture_idx}",f"{capture_num}")
    save_dir = os.path.join("RLWRLD_DEMO", str(index), "multiview")
    os.makedirs(save_dir, exist_ok=True)

    for serial_num in os.listdir(video_dir):
        if serial_num == "cam_param":
            continue

        video_path = os.path.join(video_dir, serial_num)
        save_path = os.path.join(save_dir, f"{serial_num[:-4]}.mp4")
        process_video(video_path, save_path)

