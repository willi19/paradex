import sys
import json
import argparse
import cv2
import PySpin as ps
from pathlib import Path
from flir_python.camera.camera import Camera
from flir_python.utils.utils import spin2cv
import threading
import numpy as np


def capture_images_from_all_cameras(save_path, camera_config_path, lens_info_path):
    """
    Captures a single image from all connected cameras and saves them.
    """
    # Initialize the Spinnaker system
    system = ps.System.GetInstance()
    interface_list = system.GetInterfaces()
    threads = []
    results = {}

    def capture_image(cam, save_path, results):
        """
        Captures a single image from a single camera and saves it.
        """
        cam.start_camera()
        pImg, retcode = cam.get_capture()
        if retcode:
            cvImg = spin2cv(pImg, 1536, 2048)  # Adjust resolution as needed
            image_save_path = save_path / f"{cam.serialnum}.png"
            cv2.imwrite(str(image_save_path), cvImg)
            print(f"Image saved at: {image_save_path}")
            results[cam.serialnum] = True
        else:
            print(f"Failed to capture image from camera {cam.serialnum}")
            results[cam.serialnum] = False
        cam.stop_camera()

    try:
        # Iterate through interfaces and cameras
        for pInterface in interface_list:
            camera_list = pInterface.GetCameras()
            if camera_list.GetSize() > 0:
                for pCam in camera_list:
                    # Load camera configuration and lens information
                    camera_config = json.load(open(camera_config_path, "r"))
                    lens_info = json.load(open(lens_info_path, "r"))

                    # Initialize the camera
                    cam = Camera(pCam, None, camera_config, lens_info, save_path, 0, False, False)

                    # Start a thread for each camera
                    t = threading.Thread(target=capture_image, args=(cam, save_path, results))
                    threads.append(t)
                    t.start()

                    del pCam
            camera_list.Clear()

        # Wait for all threads to complete
        for t in threads:
            t.join()

    finally:
        # Cleanup
        interface_list.Clear()
        system.ReleaseInstance()

    print("Image capture completed for all cameras.")
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Capture a single image from multiple cameras.")
    parser.add_argument(
        "--save_folder", type=str, required=True, help="Folder to save the captured images."
    )
    parser.add_argument(
        "--camera_config", type=str, required=True, help="Path to the camera configuration JSON file."
    )
    parser.add_argument(
        "--lens_info", type=str, required=True, help="Path to the lens information JSON file."
    )
    args = parser.parse_args()

    save_path = Path(args.save_folder)
    save_path.mkdir(parents=True, exist_ok=True)

    results = capture_images_from_all_cameras(save_path, args.camera_config, args.lens_info)
    print("Capture Results:", results)
