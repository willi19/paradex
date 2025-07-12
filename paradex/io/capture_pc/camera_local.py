import json
import threading

from paradex.utils.env import get_network_info
from paradex.io.capture_pc.util import get_server_socket
from paradex.io.camera.camera_loader import CameraManager

class CameraCommandReceiver():
    def __init__(self):  
        port = get_network_info()["remote_camera"]
        self.socket = get_server_socket(port)
        self.ident = None
        self.camera = None
        self.exit = False
        self.file_name = None
        
        self.register()
        self.initialize_camera()
        
        self.get_thread = threading.Thread(target=self.get_message)
        
    def get_message(self):
        while not self.exit():
            _, message = self.socket.recv_multipart()
            if message == "quit":
                self.exit = True
                self.camera.end()
                self.camera.quit()
                self.send_message("terminated")
            
            if message[:5] == "start:":
                self.file_name = message.split(":")[1]
                self.camera.set_save_dir(self.file_name)
                self.camera.start()
                self.send_message("capture_start")
                
                if self.mode == "image":
                    self.camera.wait_for_capture_end()
                    self.send_message("capture_end")
                                
            if message == "stop":
                self.camera.end()
                self.send_message("capture_end")
        
    def send_message(self, message):
        self.socket.send_multipart([self.ident, message.encode('utf-8')])
    
    def register(self):
        ident, msg = self.socket.recv_multipart()
        msg = msg.decode()
        print(msg)
        if msg == "register":
            self.ident = ident
            print(self.ident)
        self.send_message("registered")   
         
    def initialize_camera(self):
        ident, message = self.socket.recv_multipart()
        cam_info = json.loads(message.decode())
        
        self.mode = cam_info["mode"]
        self.serial_list = cam_info["serial_list"]
        self.camera = CameraManager(mode = cam_info["mode"], serial_list=cam_info["serial_list"], syncMode=cam_info["sync"])
        self.send_message("camera_ready")
        
        print(cam_info)