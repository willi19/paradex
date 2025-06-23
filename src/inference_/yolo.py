from paradex.io.camera.camera_loader import CameraManager
from paradex.io.capture_pc.client import register, get_socket
from paradex.io.capture_pc.util import serialize

import time
import threading
import numpy as np
import sys

from paradex.model.yolo_world_module import YOLO_MODULE
from queue import SimpleQueue

socket = get_socket(5556)
client_ident = register(socket)

msg_queue = SimpleQueue()
end_event = threading.Event()

capture_ready = [False] * 4
try:
    camera = CameraManager("stream", path=None, serial_list=None, syncMode=True)
except:
    sys.exit(1)


num_cam = camera.num_cameras
camera.start()

def camera_thread_func(cam_ind):
    last_frame_ind = 0
    yolo_module = YOLO_MODULE(categories="pringles")
    serial_num = camera.serial_list[cam_ind]
    capture_ready[cam_ind] = True
    
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

        msg_dict = serialize({
            "frame": int(last_frame_ind[cam_ind]),
            # "detections.mask": detections.mask,
            "detections.xyxy": detections.xyxy,
            "detections.confidence": detections.confidence,
            "type": "demo",
            "serial_num": serial_num,
        })
        
        msg_queue.put(msg_dict)

def wait_for_cameras_ready():
    while not all(capture_ready):
        time.sleep(0.1)
        
threads = []

for i in range(4):
    t = threading.Thread(target=camera_thread_func, args=(i,))
    t.start()
    threads.append(t)

for t in threads:
    t.join()

wait_for_cameras_ready()

start_time = time.time()
while time.time() - start_time < 10:
    if not msg_queue.empty():
        msg = msg_queue.get()
        socket.send_multipart([client_ident, msg])
    else:
        time.sleep(0.001)

end_event.set()
camera.end()
camera.quit()
