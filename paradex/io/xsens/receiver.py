import numpy as np
import socket
import struct

import numpy as np
from scipy.spatial.transform import Rotation as R
from paradex.utils.file_io import config_dir
import json
import os
from threading import Event, Thread, Lock

class XSensReceiver:
    def __init__(self) -> None:
        network_config = json.load(open(os.path.join(config_dir, "environment/network.json"), "r"))
        port = network_config["xsens"]
        self.hand_pose = {"left":np.array((20, 4, 4)),
                          "right":np.array((20, 4, 4))}
        
        self.host_ip = "0.0.0.0"
        self.port = port
        
        self.stop_event = Event()
        self.lock = Lock()
        
        self.recv_thread = Thread(target=self.run)
        self.recv_thread.start()

    def get_data(self):
        with self.lock:
            return {
                "left": self.hand_pose["left"].copy(),
                "right": self.hand_pose["right"].copy(),
            }

    def parse(self, pose_bytearray):
        total_data_len = len(pose_bytearray)
        pose_data_per_frame = []

        for data_idx in range(0, total_data_len, 32):
            pose_T = np.eye(4)
            
            pose_seg = pose_bytearray[data_idx : data_idx + 32]
            decoded_pose_seg = struct.unpack("!I 3f 4f", pose_seg)
            
            xyz = np.array(decoded_pose_seg[1:4])
            quaternion = np.array([*decoded_pose_seg[5:], decoded_pose_seg[4]])
            rot_mat = R.from_quat(quaternion).as_matrix()
            
            pose_T[:3,:3] = rot_mat
            pose_T[:3, 3] = xyz
            pose_data_per_frame.append(pose_T)
        
        return pose_data_per_frame
    
    def run(self):
        # Define the server socket
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.socket.bind((self.host_ip, self.port))

        header_format = "!6s I B B I B B B B 2s H"
        header_struct = struct.Struct(header_format)
        
        while not self.stop_event.is_set():
            data, addr = self.socket.recvfrom(4096)
            data_header = header_struct.unpack(data[:24])  # header is 24 bytes
            
            header_id_string = data_header[0].decode("utf-8")
            assert header_id_string[:4] == "MXTP"
            message_id = int(header_id_string[4:])

            if message_id == 2:
                pose_data_parsed = self.parse(data[24:])
                    
                left_hand_pose = pose_data_parsed[23+0:23+20].copy()
                right_hand_pose = pose_data_parsed[23+20:23+40].copy()
                pelvis_pose = pose_data_parsed[0].copy() # global pose of Xsens
                
                for i in range(0, 20):
                    self.hand_pose["Right"][i] = np.linalg.inv(pelvis_pose) @ right_hand_pose[i]
                    self.hand_pose["Left"][i] = np.linalg.inv(pelvis_pose) @ left_hand_pose[i]
    
        self.socket.close()
    
    def quit(self):
        self.stop_event.set()
        self.recv_thread.join()