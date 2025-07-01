from paradex.io.camera.camera_loader import CameraManager
import cv2
import time
from paradex.utils.file_io import find_latest_directory, home_path, download_dir, shared_dir

camera = CameraManager("image")
num_cam = camera.num_cameras

for i in range(1):
    save_path = f"{shared_dir}/demo_250618/pringles/{i}/images"
    camera.set_save_dir(save_path)

    camera.start()
    start_time = time.time()
    camera.wait_for_capture_end()
    end_time = time.time()
    print(f"Capture time: {end_time - start_time:.2f} seconds")
    time.sleep(1)

camera.quit()