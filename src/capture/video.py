from ..camera.camera_loader import CameraManager
import os


def main():
    manager = CameraManager(num_cameras=4, duration=30, name="test", is_streaming=False)
    manager.start()

if __name__ == "__main__":
    main()
