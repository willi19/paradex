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

def capture_images_from_all_cameras(save_path, lens_info_path, camera_config_path):
    """
    Captures a single image from all connected cameras and saves them.
    """
    # Iterate through interfaces and cameras
    camera_config = json.load(open(camera_config_path, "r"))
    lens_info = json.load(open(lens_info_path, "r"))

    # Initialize the Spinnaker system
    system = ps.System.GetInstance()

    ret = {}
    camera_list = system.GetCameras()
    for pCam in camera_list:
        cam = Camera(pCam, camera_config, lens_info, save_path, False)
        for frame_num in range(1):
            for _ in range(10):
                pImg, retcode = cam.get_capture(0) # 200 ms timeout
                if retcode:
                    cvImg = spin2cv(pImg, 1536, 2048)  # Adjust resolution as needed
                    image_save_path = os.path.join(save_path , f"{cam.serialnum}.png") 
                    cv2.imwrite(str(image_save_path), cvImg)
                    print(f"Image saved at: {image_save_path}")
                    cam.stop_camera()
                    ret[cam.serialnum] = cvImg
                    break

                else:
                    print(f"Failed to capture image from camera {cam.serialnum}")

    for pCam in camera_list:
        del pCam

    camera_list.Clear()

    system.ReleaseInstance()

    print("Image capture completed for all cameras.")
    return ret

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Capture a single image from multiple cameras.")
    parser.add_argument(
        "--save_folder", type=str, required=True, help="Folder to save the captured images."
    )
    camera_config = "config/lens_info.json"
    lens_info = "config/camera.json"
    parser.add_argument(
        "--num_image", type=int, required=False, default=1
    )
    args = parser.parse_args()

    save_path = Path(args.save_folder)
    save_path.mkdir(parents=True, exist_ok=True)

    capture_images_from_all_cameras(save_path, args.num_image, lens_info, camera_config)
