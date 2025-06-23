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
print("yolo module loaded")

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
while time.time() - start_time < 10:
    for i in range(num_cam):
        if camera.frame_num[i] == last_frame_ind[i]:
            continue

        last_frame_ind[i] = camera.frame_num[i]
        with camera.locks[i]:
            last_image = camera.image_array[i].copy()
        print(last_frame_ind[i], time.time()-start_time)

        
        detections = yolo_module.process_img(last_image, with_segmentation=False)
        
        if detections.xyxy.size > 0:
            bbox = detections.xyxy[0]
            # detections.bbox_center = bbox[:2] + (bbox[2:] - bbox[:2]) / 2
            detections.mask = np.zeros((1, last_image.shape[0], last_image.shape[1]), dtype=bool)
            detections.mask[0, int(bbox[1]):int(bbox[3]), int(bbox[0]):int(bbox[2])] = True
        
                
        if detections.mask is not None: detections.mask = detections.mask.tolist()
        if detections.mask is None: detections.mask = []
        if detections.xyxy is not None: detections.xyxy = detections.xyxy.tolist()
        if detections.confidence is not None: detections.confidence = detections.confidence.tolist()

        serial_num = camera.serial_list[i]

        save_flag[i] = False

        msg_dict = {
            "frame": int(last_frame_ind[i]),
            "detections.mask": detections.mask,
            "detections.xyxy": detections.xyxy,
            "detections.confidence": detections.confidence,
            "type": "demo",
            "serial_num": serial_num,
        }
        print(int(last_frame_ind[i]), type(detections.mask), type(detections.xyxy), type(detections.confidence), serial_num)
        
    time.sleep(0.01)

camera.end()
camera.quit()
