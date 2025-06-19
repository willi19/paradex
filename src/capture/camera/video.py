from paradex.io.camera.camera_loader import CameraManager
import time
from paradex.utils.file_io import find_latest_directory, home_path, download_dir, shared_dir

camera = CameraManager("video", syncMode=True)
num_cam = camera.num_cameras

save_path = f"{shared_dir}/demo_250618/pringles/0/video"
camera.set_save_dir(save_path)
camera.start()

start_time = time.time()

while True:
    if time.time() - start_time > 5:
        break
    time.sleep(0.01)
camera.end()
camera.quit()