from ..camera.camera_loader import CameraManager
import os


def main():
    homedir = os.path.expanduser("~")
    manager = CameraManager(num_cameras=4, duration=30, save_dir=f"{homedir}/captures1", is_streaming=False)
    manager.start()

if __name__ == "__main__":
    main()