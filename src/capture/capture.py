from paradex.camera.camera_loader import CameraManager
import argparse
from paradex.utils.io import capture_path_list, shared_dir
import os

def main(save_path):
    manager = CameraManager(num_cameras=4, name=save_path, is_streaming=False)
    manager.start()

def get_last_directory(name):
    capture_path = os.path.join(shared_dir, "capture", name)
    if not os.path.exists(capture_path):
        return 0
    dirs = [int(d) for d in dirs if d.isdigit()]
    max_dir = max(dirs)

    capture_path_local = os.path.join(capture_path_list[0], "capture", name)
    has_local = False
    for ind in os.listdir(capture_path_local):
        if int(ind) > max_dir:
            max_dir = int(ind)
            has_local = True
    if has_local:
        return max_dir+1    
    if os.path.exists(os.path.join(capture_path, max_dir, "videos")):
        return max_dir + 1
    else:
        return max_dir

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Capture video from cameras.")
    parser.add_argument("--name", type=str, default="trash", help="Name of the capture.")
    args = parser.parse_args()
    name = args.name

    for capture_path in capture_path_list:
        os.makedirs(os.path.join(capture_path, "capture", name), exist_ok=True)

    index = get_last_directory(name)

    save_path = f"capture/{name}/{index}/videos"
    main(save_path)
