import cv2
import os
import json
from tqdm import tqdm
import subprocess

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

def convert_avi_to_mp4(input_path, output_path):
    # Build the ffmpeg command
    command = [
        "ffmpeg",
        "-i", input_path,        # input file
        "-c:v", "libx264",       # video codec
        "-preset", "fast",       # encoding speed (options: ultrafast, superfast, fast, medium, slow, etc.)
        "-crf", "23",            # quality (lower = better, 18–28 is common)
        "-c:a", "aac",           # audio codec
        "-b:a", "192k",          # audio bitrate
        "-y",                    # overwrite output file if it exists
        output_path
    ]

    try:
        subprocess.run(command, check=True)
        print("Conversion successful!")
    except subprocess.CalledProcessError as e:
        print("An error occurred during conversion:")
        print(e)

def convert_avi_to_compressed_avi(input_path, output_path):
    # FFmpeg 명령어로 h.264 압축된 avi 생성
    command = [
        "ffmpeg",
        "-i", input_path,
        "-c:v", "libx264",
        "-preset", "fast",
        "-crf", "23",
        "-an",  # 오디오 제거. 필요시 제거
        "-y",   # 기존 파일 덮어쓰기
        output_path
    ]

    try:
        subprocess.run(command, check=True)
        print(f"Compressed AVI saved to {output_path}")
    except subprocess.CalledProcessError as e:
        print(f"Compression failed for {input_path}: {e}")