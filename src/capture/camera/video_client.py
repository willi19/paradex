from paradex.io.camera.camera_loader import CameraManager
import time

camera = CameraManager("video")
num_cam = camera.num_cameras

save_path = f"erasethis"
camera.set_save_dir(save_path)
camera.start()

start_time = time.time()

while True:
    if time.time() - start_time > 5:
        break
    time.sleep(0.01)
camera.end()
camera.quit()