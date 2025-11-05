from threading import Thread
import os

from paradex.io.camera_system.camera import Camera
from paradex.utils.file_io import home_path

class CameraLoader:
    def __init__(self):
        self.cameralist = []
    
    def load_pyspin_camera(self, serial_list=None):
        from paradex.io.camera_system.pyspin import get_serial_list, autoforce_ip
        
        autoforce_ip()
        
        if serial_list is None:
            serial_list = get_serial_list()
        
        self.cameralist = [Camera("pyspin", serial) for serial in serial_list]
    
    def start(self, mode, syncMode, save_path=None, fps=30):
        if mode == "image":
            save_paths = [os.path.join(home_path, save_path, "images") for _ in self.cameralist]
            print(save_paths)   
            for path in save_paths:
                os.makedirs(path, exist_ok=True)

        elif mode in ["video", "full"]:
            save_paths = [os.path.join(home_path, f"captures{ind % 2 + 1}", save_path, "videos") for ind, _ in enumerate(self.cameralist)]
            for path in save_paths:
                os.makedirs(path, exist_ok=True)
            
        else:
            save_paths = [None for _ in self.cameralist]

        threads = []
        for camera, path in zip(self.cameralist, save_paths):
            t = Thread(target=camera.start, args=(mode, syncMode, path, fps))
            threads.append(t)
            
        for t in threads:
            t.start()
        
        for t in threads:
            t.join()
    
    def stop(self):
        threads = []
        for camera in self.cameralist:
            t = Thread(target=camera.stop)
            threads.append(t)
            
        for t in threads:
            t.start()
        
        for t in threads:
            t.join()
    
    def end(self):
        threads = []
        for camera in self.cameralist:
            t = Thread(target=camera.end)
            threads.append(t)
            
        for t in threads:
            t.start()
        
        for t in threads:
            t.join()

    def get_status_list(self):
        status_list = []
        for camera in self.cameralist:
            status_list.append(camera.get_status())
        return status_list