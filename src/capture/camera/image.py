from paradex.io.camera.camera_loader import CameraManager
import cv2
import time

camera = CameraManager("image")
num_cam = camera.num_cameras

for i in range(1):
    save_path = f"/shared_data/demo_250618/pringles/{i}"
    camera.set_save_dir(save_path)

    camera.start()
    start_time = time.time()
    camera.wait_for_capture_end()
    end_time = time.time()
    print(f"Capture time: {end_time - start_time:.2f} seconds")
    time.sleep(1)

camera.quit()