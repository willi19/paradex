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
        next_dir = 0
    else:      
        dirs = [int(d) for d in os.listdir(capture_path) if d.isdigit()]
        next_dir = max(dirs) if len(dirs) > 0 else 0
        if os.path.exists(os.path.join(capture_path, str(next_dir), "videos")):
            next_dir += 1
    
    capture_path_local = os.path.join(capture_path_list[0], "capture", name)
    for ind in os.listdir(capture_path_local):
        if int(ind)+1 > next_dir:
            next_dir = int(ind)+1
    return next_dir

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

    last_capture_index = max(os.listdir(os.path.join(shared_dir, "capture", name)), key=int)
    
    capture_path_local = os.path.join(capture_path_list[0], "capture", name)

    f = open(os.path.join(capture_path_local, str(index), "index.txt"), "w")
    f.write(f"{last_capture_index}")()
    f.close()