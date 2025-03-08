import sys
import json
import argparse
import cv2
import PySpin as ps
from pathlib import Path
from paradex.camera.camera import Camera
from ..utils.image_util import spin2cv
import threading
import numpy as np
import os

def capture_images_from_all_cameras(save_path, num_image, lens_info_path, camera_config_path):
    """
    Captures a single image from all connected cameras and saves them.
    """
    # Iterate through interfaces and cameras
    camera_config = json.load(open(camera_config_path, "r"))
    lens_info = json.load(open(lens_info_path, "r"))

    # Initialize the Spinnaker system
    system = ps.System.GetInstance()

    threads = []
    def capture_image(cam, num_image, save_path):
        """
        Captures a single image from a single camera and saves it.
        """
        for frame_num in range(num_image):
            pImg, retcode = cam.get_capture()
            if retcode:
                cvImg = spin2cv(pImg, 1536, 2048)  # Adjust resolution as needed
                image_save_path = save_path / f"{cam.serialnum}.png" if num_image == 1 else save_path / str(frame_num) / f"{cam.serialnum}.png"
                cv2.imwrite(str(image_save_path), cvImg)
                print(f"Image saved at: {image_save_path}")

            else:
                print(f"Failed to capture image from camera {cam.serialnum}")

        cam.stop_camera()

    if num_image > 1:
        for frame_num in range(num_image):
            os.makedirs(save_path / str(frame_num),exist_ok=True)
    try:
        camera_list = system.GetCameras()
        for pCam in camera_list:
            # Load camera configuration and lens information
            # Initialize the camera
            cam = Camera(pCam, camera_config, lens_info, save_path, False)
            t = threading.Thread(target=capture_image, args=(cam, num_image, save_path))
            threads.append(t)
            t.start()
        for t in threads:
            t.join()


    finally:
        import gc

        # Enable garbage collection debugging
        # gc.set_debug(gc.DEBUG_LEAK)

        # Force garbage collection
        for pCam in camera_list:
            del pCam

        camera_list.Clear()

        system.ReleaseInstance()

    print("Image capture completed for all cameras.")\

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Capture a single image from multiple cameras.")
    parser.add_argument(
        "--save_folder", type=str, required=True, help="Folder to save the captured images."
    )
    camera_config = "config/lens.json"
    lens_info = "config/camera.json"
    parser.add_argument(
        "--num_image", type=int, required=False, default=1
    )
    args = parser.parse_args()

    save_path = Path(args.save_folder)
    save_path.mkdir(parents=True, exist_ok=True)

    capture_images_from_all_cameras(save_path, args.num_image, lens_info, camera_config)
