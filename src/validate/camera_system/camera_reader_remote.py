#!/usr/bin/env python3
"""
Camera Reader Quick Start Guide
================================

가장 빠르게 시작하는 방법
"""

from paradex.io.camera_system.camera_reader import MultiCameraReader
from paradex.io.camera_system.camera_loader import CameraLoader
from paradex.image.merge import merge_image

import time
import cv2

cr = MultiCameraReader()
cr.close()
cr = None
print("CameraReader initialized.")
while True:
    time.sleep(1)
# start_time = time.time()
# while time.time() - start_time < 3:
#     data = cr.get_images(copy=True)
#     img_dict = {name: img for name, (img, frame_id) in data.items()}
#     print(img_dict.keys())
#     for name, (img, frame_id) in data.items():
#         print(f"{name}: frame_id={frame_id}, shape={img.shape}")
#     cv2.imshow("Multi Camera Stream", merge_image(img_dict))
#     cv2.waitKey(1)
# cr.close()
# cr = MultiCameraReader()

# while time.time() - start_time < 3:
#     data = cr.get_images(copy=True)
#     img_dict = {name: img for name, (img, frame_id) in data.items()}
#     print(img_dict.keys())
#     for name, (img, frame_id) in data.items():
#         print(f"{name}: frame_id={frame_id}, shape={img.shape}")
#     cv2.imshow("Multi Camera Stream", merge_image(img_dict))
#     cv2.waitKey(1)
    
# cr.close()
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
