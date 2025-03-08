import cv2
import os
import json
from tqdm import tqdm

def split_video(video_path_tuple, image_dir, frame_interval=1):
    """
    Extract frames from a video using OpenCV with FFmpeg backend.

    Parameters:
    - video_path_tuple: Tuple (video_path, json_path) containing video file and corresponding JSON.
    - image_dir: Directory to save extracted frames.
    - frame_interval: Extract every 'n' frames (default is 1, meaning extract all frames).
    """

    video_path, json_path = video_path_tuple  # Unpack tuple
    serial_num = os.path.basename(video_path).split("_")[0]  # Extract serial number from video filename

    # Ensure output directory exists
    os.makedirs(image_dir, exist_ok=True)

    # Open the video file using OpenCV
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Cannot open video file {video_path}")
        return

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

    finished = True
    for ts in timestamp:
        if not os.path.exists(os.path.join(image_dir, f"{ts:05d}", f"{serial_num}.jpg")):
            finished = False
            break
    if finished:
        return 
    
    frame_count = 0    

    # Initialize tqdm progress bar based on total frames
    with tqdm(total=total_frames, desc=f"Processing {video_path}", unit="frame", leave=False) as inner_bar:
        while True:
            ret, frame = cap.read()
            if not ret:
                break  # End of video

            # Save frame at specified intervals
            if timestamp[frame_count] % frame_interval == 0:
                os.makedirs(os.path.join(image_dir, f"{timestamp[frame_count]:05d}"), exist_ok=True)
                frame_filename = os.path.join(image_dir, f"{timestamp[frame_count]:05d}", f"{serial_num}.jpg")
                if os.path.exists(frame_filename):
                    inner_bar.update(1)  # Update tqdm progress bar
                    continue
                cv2.imwrite(frame_filename, frame)

            frame_count += 1
            inner_bar.update(1)  # Update tqdm progress bar

    cap.release()
