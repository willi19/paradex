from ..camera.camera_loader import CameraManager

def main():
    manager = CameraManager(num_cameras=4, duration=30, save_dir="/home/capture16/captures1", is_streaming=False)
    manager.start()

if __name__ == "__main__":
    main()