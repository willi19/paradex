import cv2
import os
import json
from tqdm import tqdm
from paradex.utils.image_util import undistort_img
import numpy as np

def split_video(video_path_tuple, intrinsic, selected_frame):
    """
    Extract frames from a video using OpenCV with FFmpeg backend.

    Parameters:
    - video_path_tuple: Tuple (video_path, json_path) containing video file and corresponding JSON.
    - image_dir: Directory to save extracted frames.
    - frame_interval: Extract every 'n' frames (default is 1, meaning extract all frames).
    """

    video_path, json_path = video_path_tuple  # Unpack tuple
    serial_num = os.path.basename(video_path).split("_")[0]  # Extract serial number from video filename

    # Open the video file using OpenCV
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Cannot open video file {video_path}")
        return
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Read timestamp JSON file safely
    try:
        with open(json_path, "r") as json_file:
            timestamp = json.load(json_file).get("frameID", {})
    except (json.JSONDecodeError, FileNotFoundError) as e:
        print(f"Error reading JSON file {json_path}: {e}")
        cap.release()
        return

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))  # Get total frame count
    if total_frames != len(timestamp):
        print(f"Frame count mismatch between video ({total_frames}) and JSON ({len(timestamp)})")
        return 

    video_name = os.path.basename(video_path).split(".")[0]
    output_video_name = f"{video_name}_selected_frame.avi"
    output_video_path = os.path.join(os.path.dirname(video_path), output_video_name)
    
    fourcc = cv2.VideoWriter_fourcc(*"XVID")  # AVI format
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))

    frame_count = 0    
    si = 0

    # Initialize tqdm progress bar based on total frames
    with tqdm(total=total_frames, desc=f"Processing {video_path}", unit="frame", leave=False) as inner_bar:
        while True:
            ret, frame = cap.read()
            if not ret:
                break  # End of video
            
            if timestamp[frame_count] < selected_frame[si]:
                frame_count += 1
                continue
            # Save frame at specified intervals
            if timestamp[frame_count] == selected_frame[si]:
                undist_frame = undistort_img(frame, intrinsic)
                si += 1
                frame_count += 1

            else:
                undist_frame = np.zeros_like(frame)
                si += 1
            out.write(undist_frame)
            inner_bar.update(1)  # Update tqdm progress bar
            if si == len(selected_frame):
                break
    cap.release()
    out.release()
