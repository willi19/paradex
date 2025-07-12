import json
import zmq
import time
from paradex.utils.env import get_pcinfo, get_network_info
from paradex.io.capture_pc.util import get_client_socket

class RemoteCameraController():
    def __init__(self, mode, serial_list,sync=False):  
        self.pc_info = get_pcinfo()
        self.mode = mode
        self.serial_list = serial_list
        self.sync = sync
        port = get_network_info()["remote_camera"]
        
        self.pc_list = []
        self.serial_list = serial_list
        
        for pc_name, info in self.pc_info.items():
            for serial_num in info['cam_list']:
                if serial_num in serial_list:
                    self.pc_list.append(pc_name)
                    break
                    
        self.socket_dict = {pc_name:get_client_socket(self.pc_info[pc_name]["ip"], port) for pc_name in self.pc_list}
        self.register()
        self.initiate_camera()
        
        
    def send_message(self, message):
        for pc_name, socket in self.socket_dict.items():
            socket.send_string(message)
    
    def wait_for_message(self, message, timeout=-1):
        recv_dict = {pc_name:False for pc_name in self.pc_list}
        start_time = time.time()
        while timeout == -1 or time.time()-start_time < timeout:
            success = True
            for pc_name, socket in self.socket_dict.items():
                if recv_dict[pc_name]:
                    continue
                recv_msg = socket.recv_string()
                print(recv_msg)
                if recv_msg == message:
                    recv_dict[pc_name] = True

                if not recv_dict[pc_name]:
                    success = False
            
            if success:
                return True                
            time.sleep(0.01)
            
        return False
    
    def register(self):
        self.send_message("register")   
        return self.wait_for_message("registered")
         
    def initiate_camera(self):
        message = json.dumps(
            {
                "mode":self.mode,
                "serial_list":self.serial_list,
                "sync":self.sync
            }
        )
        self.send_message(message)
        return self.wait_for_message("camera_ready")
    
    def start_capture(self, filename=''):
        message = "start:"+filename
        self.send_message(message)
        print("start")
        print(self.wait_for_message("capture_start"))
        
    def end_capture(self):
        if self.mode != "image":
            self.send_message("stop")
        self.wait_for_message("capture_end")   
        
    def quit(self):
        self.send_message("quit") 
        self.wait_for_message("terminated")

