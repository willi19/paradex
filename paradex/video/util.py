
def get_video_list(video_dir):
    """
    Get a list of video files in the specified directory.

    Parameters:
    - video_dir: Directory containing video files.

    Returns:
    - video_list: List of video files in the directory.
    """
    video_list = []
    for f in os.listdir(video_dir):
        if f.endswith(".avi") or f.endswith(".mp4"):
            video_name = f.split("-")[0] # {serial_num}_{date}
            timestamp_path = os.path.join(video_dir, video_name+"_timestamp.json")
            if not os.path.exists(timestamp_path):
                video_list.append((os.path.join(video_dir, f), None))
                continue
            video_list.append((os.path.join(video_dir, f), timestamp_path))

    return video_list
