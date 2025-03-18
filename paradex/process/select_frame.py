import cv2
import os
import json
from tqdm import tqdm
from paradex.utils.image_util import undistort_img
import numpy as np

def split_video(video_path_tuple, intrinsic, selected_frame, index_offset):
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

    fourcc = cv2.VideoWriter_fourcc(*"XVID")  # AVI format
    frame_count = 0    
    
    # Initialize tqdm progress bar based on total frames
    with tqdm(total=total_frames, desc=f"Processing {video_path}", unit="frame", leave=False) as inner_bar:
        for idx, range_list in selected_frame.items():
            output_video_name = f"{serial_num}.avi"
            output_video_dir = os.path.dirname(os.path.dirname(video_path)).replace("capture/", "processed/")
            os.makedirs(os.path.join(output_video_dir,str(int(idx)+int(index_offset)), "video"), exist_ok=True)

            output_video_path = os.path.join(output_video_dir,str(int(idx)+int(index_offset)), "video", output_video_name)
        
            out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))

            for(start_frame, end_frame) in range_list:
                while timestamp[frame_count] < start_frame:
                    ret, frame = cap.read()
                    frame_count += 1
                    inner_bar.update(1)
                    if not ret or frame_count >= total_frames:
                        break
                for i in range(start_frame, end_frame+1):
                    if frame_count == len(timestamp) or i < timestamp[frame_count]:
                        undistorted_frame = np.zeros((frame_width, frame_height, 3), dtype=np.uint8)
                    else:
                        ret, frame = cap.read()
                        frame_count += 1
                        inner_bar.update(1)
                        if not ret:
                            break
                        undistorted_frame = undistort_img(frame, intrinsic)                       
                    out.write(undistorted_frame)
            out.release()
    cap.release()
