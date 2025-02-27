from ..camera.camera_loader import CameraManager
import os
import argparse

def main(name):
    manager = CameraManager(num_cameras=4, duration=30, name=name, is_streaming=False)
    manager.start()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Capture video from cameras.")
    parser.add_argument("--name", type=str, default="trash", help="Name of the capture.")
    args = parser.parse_args()
    name = args.name
    main(name)
