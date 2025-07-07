from paradex.video.video import convert_avi_to_compressed_avi
import os
import shutil

def compress_video(video_path_tuple):
    """
    Extract frames from a video using OpenCV with FFmpeg backend.

    Parameters:
    - video_path_tuple: Tuple (video_path, json_path) containing video file and corresponding JSON.
    - image_dir: Directory to save extracted frames.
    - frame_interval: Extract every 'n' frames (default is 1, meaning extract all frames).
    """

    video_path, json_path = video_path_tuple  # Unpack tuple
    serial_num = os.path.basename(video_path).split("-")[0]  # Extract serial number from video filename
    # Open the video file using OpenCV
    
    # Read timestamp JSON file safely
    if json_path == None:
        return
    
        
    index = os.path.basename(os.path.dirname(os.path.dirname(json_path)))
    name = os.path.basename(os.path.dirname(os.path.dirname(os.path.dirname(json_path))))
    output_video_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(video_path)))).replace("capture/", "capture_compressed/")
    
    os.makedirs(os.path.join(output_video_dir, name, index, "videos"), exist_ok=True)
    convert_avi_to_compressed_avi(video_path, os.path.join(output_video_dir, name, index, "videos", serial_num + ".avi"))

    shutil.copy(json_path, os.path.join(output_video_dir, name, index, "videos", os.path.basename(json_path)))
