from paradex.camera.camera_loader import CameraManager
import argparse
from paradex.utils.io import capture_path_list
import os

def main(name):
    manager = CameraManager(num_cameras=4, name=name, is_streaming=False)
    manager.start()

def get_last_directory(name):
    capture_path = os.path.join(capture_path_list[0], "capture", name)
    dirs = [d for d in os.listdir(capture_path)]
    return len(dirs)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Capture video from cameras.")
    parser.add_argument("--name", type=str, default="trash", help="Name of the capture.")
    args = parser.parse_args()
    name = args.name

    index = get_last_directory(name)
    save_path = f"capture/{name}/{index}"
    os.makedirs(save_path, exist_ok=True)
    main(save_path)
