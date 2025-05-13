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
    if camera.frame_num[0] != last_frame:
        last_frame = camera.frame_num[0]
        print(last_frame)
        with camera.locks[0]:
            last_image = camera.image_array[0].copy()

        # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        last_image = cv2.resize(last_image, (640, 480))
        cv2.imshow("Camera Stream", last_image)
        cv2.waitKey(30)

    else:
        cv2.imshow("Camera Stream", last_image)
        cv2.waitKey(30)
camera.end()
camera.quit()