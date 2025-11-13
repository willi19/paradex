#!/usr/bin/env python3
"""
Camera Reader Quick Start Guide
================================

가장 빠르게 시작하는 방법
"""

from paradex.io.camera_system.camera_reader import discover_cameras, CameraReader, MultiCameraReader
from paradex.io.camera_system.camera_loader import CameraLoader
from paradex.image.merge import merge_image

import time
import cv2

cl = CameraLoader()
cl.load_pyspin_camera()  

cl.start("stream", False, fps=10)
try:
    cr = MultiCameraReader(cl.camera_names)
    
except:
    cl.end()

start_time = time.time()
while time.time() - start_time < 20:
    data = cr.get_images(copy=True)
    img_dict = {name: img for name, (img, frame_id) in data.items()}
    
    cv2.namedWindow("Multi Camera Stream", cv2.WINDOW_NORMAL)
    cv2.imshow("Multi Camera Stream", data[list(data.keys())[0]][0])
    print(merge_image(img_dict).shape)
    time.sleep(0.02)
    
cl.end()

# import cv2

# from paradex.io.camera_system.camera_reader import CameraReader, MultiCameraReader
# from paradex.utils.env import get_serial_list

# serial_list = get_serial_list()
# for serial in serial_list:
#     cr = CameraReader(serial)
#     frame = cr.read_frame()
#     cr.release()

# mcr = MultiCameraReader(serial_list)
# frames = mcr.read_frames()
# mcr.release()    
# >>>>>>> 0aa6552674ff1c61e79cdff03ab96088a3147033
