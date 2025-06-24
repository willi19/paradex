from paradex.io.camera.camera_loader import CameraManager
from paradex.io.capture_pc.client import register, get_socket
from paradex.io.capture_pc.util import serialize

import time
import threading
import numpy as np
import sys
import cv2

from paradex.model.yolo_world_module import YOLO_MODULE
from queue import SimpleQueue
import json

socket = get_socket(5556) 
client_ident = register(socket) # client identity given by server

msg_queue = SimpleQueue()
end_event = threading.Event()

capture_ready = [False] * 4
try:
    camera = CameraManager("stream", path=None, serial_list=None, syncMode=True)
except:
    sys.exit(1)

num_cam = camera.num_cameras
camera.start()
_ = YOLO_MODULE(categories="pringles")

def camera_thread_func(cam_ind):
    last_frame_ind = 0
    # print(f"Before Setting up YOLO module {serial_num}")
    serial_num = camera.serial_list[cam_ind]
    yolo_module = YOLO_MODULE(categories="pringles")
    capture_ready[cam_ind] = True
    # print(f"Camera {serial_num} is ready.")
    while not end_event.is_set():

        if camera.frame_num[cam_ind] == last_frame_ind:
            time.sleep(0.005)
            continue
        
        loop_start_time = time.time()
        last_frame_ind = camera.frame_num[cam_ind]

        with camera.locks[cam_ind]:
            last_image = camera.image_array[cam_ind].copy()
        
        detections = yolo_module.process_img(last_image, with_segmentation=False)
        
        # if detections.xyxy.size > 0:
        #     bbox = detections.xyxy[0]
        #     detections.mask = np.zeros((1, last_image.shape[0], last_image.shape[1]), dtype=bool)
        #     detections.mask[0, int(bbox[1]):int(bbox[3]), int(bbox[0]):int(bbox[2])] = True

        # if detections.mask is not None: detections.mask = detections.mask.tolist()
        # if detections.mask is None: detections.mask = []
        if detections.xyxy is not None: detections.xyxy = detections.xyxy.tolist()
        if detections.confidence is not None: detections.confidence = detections.confidence.tolist()

        if int(last_frame_ind) % 10 == 0:
            width, height =  int(camera.width/16), int(camera.height/16)
            resized_rgb = cv2.resize(last_image, (width, height))
        else:
            resized_rgb = 'None'
            
        msg_dict = serialize({
            "frame": int(last_frame_ind),
            # "detections.mask": detections.mask,
            "detections.xyxy": detections.xyxy,
            "detections.confidence": detections.confidence,
            "type": "demo",
            "serial_num": serial_num,
            "time": time.time()- loop_start_time,
            "resized_rgb": resized_rgb
        })
        
        msg_queue.put(msg_dict)


def wait_for_cameras_ready():
    while not all(capture_ready):
        time.sleep(0.01)
        
threads = []

for i in range(4):
    t = threading.Thread(target=camera_thread_func, args=(i,))
    t.start()
    threads.append(t)

wait_for_cameras_ready()
# print("All cameras are ready.")
socket.send_multipart([client_ident, b"camera_ready"])

start_time = time.time()
while time.time() - start_time < 100:
    if not msg_queue.empty():
        msg_dict = msg_queue.get()
        msg = json.dumps(msg_dict).encode()
        socket.send_multipart([client_ident, msg])
        print(msg_dict)
    else:
        time.sleep(0.001)

end_event.set()

for t in threads:
    t.join()
camera.end()
camera.quit()
