import zmq
from paradex.utils.file_io import config_dir, shared_dir
import json
import os
from paradex.io.camera.camera_loader import CameraManager
import time
from paradex.image.aruco import detect_charuco, merge_charuco_detection
import threading
import numpy as np
import sys

from yolo_world_module import YOLO_MODULE

from threading import Lock

lock = Lock()

def camera_thread_func(i):
    global last_frame_ind, save_flag

    while time.time() - start_time < 30:
        if camera.frame_num[i] == last_frame_ind[i]:
            time.sleep(0.005)
            continue
        
        loop_start_time = time.time()
        last_frame_ind[i] = camera.frame_num[i]

        with camera.locks[i]:
            last_image = camera.image_array[i].copy()
        
        print(f"[CAM {i}] Frame: {last_frame_ind[i]}, Time: {time.time() - start_time:.2f} Cost time: {time.time() - loop_start_time:.4f}s")
        with lock:
            detections = yolo_module.process_img(last_image, with_segmentation=False)
        print(f"[CAM {i}] Detections time: {time.time() - loop_start_time:.4f}s {camera.serial_list[i]}")
        
        if detections.xyxy.size > 0:
            bbox = detections.xyxy[0]
            detections.mask = np.zeros((1, last_image.shape[0], last_image.shape[1]), dtype=bool)
            detections.mask[0, int(bbox[1]):int(bbox[3]), int(bbox[0]):int(bbox[2])] = True

        if detections.mask is not None: detections.mask = detections.mask.tolist()
        if detections.mask is None: detections.mask = []
        if detections.xyxy is not None: detections.xyxy = detections.xyxy.tolist()
        if detections.confidence is not None: detections.confidence = detections.confidence.tolist()

        msg_dict = {
            "frame": int(last_frame_ind[i]),
            "detections.mask": detections.mask,
            "detections.xyxy": detections.xyxy,
            "detections.confidence": detections.confidence,
            "type": "demo",
            "serial_num": camera.serial_list[i],
        }

        # 여기서 메시지를 전송하거나 저장 등의 후처리
        # 예: zmq_pub.send_json(msg_dict)
        print(f"[CAM {i}] Sending data for frame {time.time() - start_time:.2f}, Serial: {camera.serial_list[i]}")
        save_flag[i] = False



yolo_module = YOLO_MODULE(categories="pringles")
print("yolo module initialized")
try:
    camera = CameraManager("stream", path=None, serial_list=None, syncMode=True)
except:
    sys.exit(1)

num_cam = camera.num_cameras

camera.start()
last_frame_ind = [0 for _ in range(num_cam)]
save_flag = [False for _ in range(num_cam)]
save_finish = True

start_time = time.time()
# while time.time() - start_time < 30:
#     for i in range(num_cam):
#         if camera.frame_num[i] == last_frame_ind[i]:
#             continue

#         last_frame_ind[i] = camera.frame_num[i]
#         with camera.locks[i]:
#             last_image = camera.image_array[i].copy()
#         print(last_frame_ind[i], time.time()-start_time)

        
#         detections = yolo_module.process_img(last_image, with_segmentation=False)
        
#         if detections.xyxy.size > 0:
#             bbox = detections.xyxy[0]
#             # detections.bbox_center = bbox[:2] + (bbox[2:] - bbox[:2]) / 2
#             detections.mask = np.zeros((1, last_image.shape[0], last_image.shape[1]), dtype=bool)
#             detections.mask[0, int(bbox[1]):int(bbox[3]), int(bbox[0]):int(bbox[2])] = True
        
                
#         if detections.mask is not None: detections.mask = detections.mask.tolist()
#         if detections.mask is None: detections.mask = []
#         if detections.xyxy is not None: detections.xyxy = detections.xyxy.tolist()
#         if detections.confidence is not None: detections.confidence = detections.confidence.tolist()

#         serial_num = camera.serial_list[i]

#         save_flag[i] = False

#         msg_dict = {
#             "frame": int(last_frame_ind[i]),
#             "detections.mask": detections.mask,
#             "detections.xyxy": detections.xyxy,
#             "detections.confidence": detections.confidence,
#             "type": "demo",
#             "serial_num": serial_num,
#         }
#     time.sleep(0.01)
threads = []
start_time = time.time()

for i in range(4):
    t = threading.Thread(target=camera_thread_func, args=(i,))
    t.start()
    threads.append(t)

for t in threads:
    t.join()

camera.end()
camera.quit()
