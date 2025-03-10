import argparse
from paradex.capture.image import capture_images_from_all_cameras
from pathlib import Path

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Capture a single image from multiple cameras.")
    parser.add_argument(
        "--save_folder", type=str, required=True, help="Folder to save the captured images."
    )
    camera_config = "config/lens.json"
    lens_info = "config/camera.json"

    args = parser.parse_args()

    save_path = Path(args.save_folder)
    save_path.mkdir(parents=True, exist_ok=True)

    capture_images_from_all_cameras(save_path, 1, lens_info, camera_config)
