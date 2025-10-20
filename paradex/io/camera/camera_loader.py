import json
from threading import Event, Lock, Thread
import time
import numpy as np
import cv2
import os

from paradex.io.camera.camera import Camera
from paradex.utils.file_io import home_path, config_dir

class CameraManager:
    def __init__(self):
        pass
    
    def wait_camera_event(self, event_name):
        for camera in self.cameras:
            event = camera.get_event(event_name)
            event.wait()
         
    def load_pyspin_camera(self, serial_list=None, mode="stream", syncMode=False):
        from paradex.io.camera.pyspin import get_serial_list
        if serial_list is None:
            serial_list = get_serial_list()

        self.cameralist = [Camera("pyspin", serial) for serial in serial_list]
        self.wait_camera_event("connection")
    
    def start(self, mode, syncMode):
        for camera in self.cameralist:
            camera.start(mode, syncMode)
        self.wait_camera_event("acquisition")
    
    def stop(self):
        for camera in self.cameralist:
            camera.stop()
        self.wait_camera_event("stop")
    
    def end(self):
        for camera in self.cameralist:
            camera.end()
        self.wait_camera_event("release")
        
    def get_frameid(self, index):
        if self.mode != "stream":
            return
        
        with self.locks[index]:
            return self.frame_num[index]
        
    def get_data(self, index):
        if self.mode != "stream":
            return 
        
        with self.locks[index]:
            return {"image":self.image_array[index].copy(), "frameid":self.frame_num[index]}