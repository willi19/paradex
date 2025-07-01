from paradex.io.camera.camera_loader import CameraManager
import cv2
import time

camera = CameraManager("stream")
num_cam = camera.num_cameras

camera.start()
start_time = time.time()

last_frame = -1
last_image = None

while time.time() - start_time < 10:
    frame_id = camera.get_frameid(0)
    if frame_id == last_frame:
        cv2.imshow("Camera Stream", last_image)
        cv2.waitKey(30)
    
    else:
        data = camera.get_data(0)
        last_frame = data["frameid"]
        last_image = data["image"]
        last_image = cv2.resize(last_image, (640, 480))
        cv2.imshow("Camera Stream", last_image)
        cv2.waitKey(30)

camera.end()
camera.quit()