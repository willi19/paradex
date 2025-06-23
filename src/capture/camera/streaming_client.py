from paradex.io.camera.camera_loader import CameraManager
import cv2
import time
from paradex.utils.file_io import shared_dir
import os

camera = CameraManager("stream")
num_cam = camera.num_cameras

camera.start()
start_time = time.time()

last_frame = 0
last_image = None

idx = 0 
start_time = time.time()
asdf = time.time()
while time.time() - start_time < 20:
    if camera.frame_num[0] != last_frame:
        print(last_frame, time.time() - asdf, camera.frame_num[0], last_frame)
        asdf = time.time()
        last_frame = camera.frame_num[0]
        
        with camera.locks[0]:
            last_image = camera.image_array[0].copy()

        # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        serial_num = camera.serial_list[0]
        # os.makedirs(f"{shared_dir}/tmp/{last_frame}", exist_ok=True)
        # last_image = cv2.resize(last_image, (640, 480))
        # cv2.imwrite(f"{shared_dir}/tmp/{last_frame}/{serial_num}.jpg", last_image)
camera.end()
camera.quit()