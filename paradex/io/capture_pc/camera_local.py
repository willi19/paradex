import json
import threading
import time
import os

from paradex.utils.env import get_network_info
from paradex.io.capture_pc.util import get_server_socket
from paradex.io.camera.camera_loader import CameraManager
from paradex.utils.file_io import home_path

class CameraCommandReceiver():
    def __init__(self):  
        self.ident = None
        self.camera = None
        self.exit = False
        self.file_name = None
        self.init = False
        
        self.get_thread = threading.Thread(target=self.get_message)
        self.get_thread.start()
        while not self.init:
            time.sleep(0.01)
            
    def get_message(self):
        port = get_network_info()["remote_camera"]
        self.socket = get_server_socket(port)
        
        self.register()
        self.initialize_camera()
        self.init = True
        
        while not self.exit:
            _, message = self.socket.recv_multipart()
            message = message.decode()
            print(message)
            if message == "quit":
                self.exit = True
                self.camera.end()
                self.camera.quit()
                self.send_message("terminated")
            
            if message[:6] == "start:":
                self.file_name = message.split(":")[1]
                if self.mode == "image":
                    self.camera.set_save_dir(os.path.join(home_path, self.file_name))
                self.camera.start()
                self.send_message("capture_start")
                
                if self.mode == "image":
                    self.camera.wait_for_capture_end()
                    self.send_message("capture_end")
                                
            if message == "stop":
                self.camera.end()
                self.send_message("capture_end")
            time.sleep(0.01)
        
    def send_message(self, message):
        self.socket.send_multipart([self.ident, message.encode('utf-8')])
    
    def register(self):
        ident, msg = self.socket.recv_multipart()
        msg = msg.decode()
        if msg == "register":
            self.ident = ident
        self.send_message("registered")   
         
    def initialize_camera(self):
        ident, message = self.socket.recv_multipart()
        cam_info = json.loads(message.decode())
        
        self.mode = cam_info["mode"]
        self.serial_list = cam_info["serial_list"]
        self.camera = CameraManager(mode = cam_info["mode"], serial_list=cam_info["serial_list"], syncMode=cam_info["sync"])
        self.send_message("camera_ready")