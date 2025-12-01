import numpy as np
import socket
import struct
from scipy.spatial.transform import Rotation as R
import time
import os
from threading import Event, Thread, Lock

xsens_joint_name = ["wrist", 
                    "thumb_metacarpal", "thumb_proximal", "thumb_distal", 
                    "index_metacarpal", "index_proximal", "index_intermediate", "index_distal", 
                    "middle_metacarpal", "middle_proximal", "middle_intermediate", "middle_distal",
                    "ring_metacarpal", "ring_proximal", "ring_intermediate", "ring_distal",
                    "pinky_metacarpal", "pinky_proximal", "pinky_intermediate", "pinky_distal"
                    ]

xsens_joint_parent_name = [None,
                      "wrist", "thumb_metacarpal", "thumb_proximal",
                      "wrist", "index_metacarpal", "index_proximal", "index_intermediate",
                      "wrist", "middle_metacarpal", "middle_proximal", "middle_intermediate",
                      "wrist", "ring_metacarpal", "ring_proximal", "ring_intermediate",
                      "wrist", "pinky_metacarpal", "pinky_proximal", "pinky_intermediate"]

XSENS2WRIST_Left = np.array([[-1, 0, 0, 0], 
                             [0, 1, 0, 0],
                             [0, 0, -1, 0],
                             [0, 0, 0, 1]])

XSENS2WRIST_Right = np.array([[1, 0, 0, 0], 
                              [0, -1, 0, 0],
                              [0, 0, -1, 0],
                              [0, 0, 0, 1]])

XSENS2GLOBAL = np.array([[1, 0, 0, 0], 
                         [0, 1, 0, 0], 
                         [0, 0, 1, 0],
                         [0, 0, 0, 1]])

class XSensReceiver:
    def __init__(self, port) -> None:
        self.hand_pose = {"Left":{},
                          "Right":{}}
        
        self.host_ip = "0.0.0.0"
        self.port = port
        
        self.exit_event = Event()
        self.save_event = Event()
        self.error_event = Event()
        
        self.lock = Lock()
        
        self.recv_thread = Thread(target=self.run)
        self.recv_thread.start()

    def get_data(self):
        with self.lock:
            left_pose = {joint_name:self.hand_pose['Left'][joint_name].copy() for joint_name in xsens_joint_name} if len(self.hand_pose['Left']) > 0 else None
            right_pose = {joint_name:self.hand_pose['Right'][joint_name].copy() for joint_name in xsens_joint_name} if len(self.hand_pose['Right']) > 0 else None
        return {
            "Left": left_pose,
            "Right": right_pose,
            "time": time.time()
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
        self.socket.settimeout(1.0)
        
        header_format = "!6s I B B I B B B B 2s H"
        header_struct = struct.Struct(header_format)

        while not self.exit_event.is_set():
            try:
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
                    
                    with self.lock:
                        for i, joint_name in enumerate(xsens_joint_name):
                            self.hand_pose["Right"][joint_name] = XSENS2GLOBAL @ np.linalg.inv(pelvis_pose) @ right_hand_pose[i] @ np.linalg.inv(XSENS2WRIST_Right)
                            self.hand_pose["Left"][joint_name] = XSENS2GLOBAL @ np.linalg.inv(pelvis_pose) @ left_hand_pose[i] @ np.linalg.inv(XSENS2WRIST_Left)
                
                    if self.save_event.is_set():
                        self.data["time"].append(data_header[3])  # timestamp
                        for joint_name in xsens_joint_name:
                            self.data["Left"][joint_name].append(self.hand_pose["Left"][joint_name].copy())
                            self.data["Right"][joint_name].append(self.hand_pose["Right"][joint_name].copy())
                            
            except socket.timeout:
                self.error_event.set()
                            
        self.socket.close()
    
    def start(self, save_path):
        self.save_path = save_path
        self.data = {
            "time":[],
            "Left": {joint_name:[] for joint_name in xsens_joint_name},
            "Right": {joint_name:[] for joint_name in xsens_joint_name}
        }
        with self.lock:
            self.save_event.set()
    
    def stop(self):
        with self.lock:
            self.save_event.clear()
        
        os.makedirs(self.save_path, exist_ok=True)
        np.save(os.path.join(self.save_path, "time.npy"), self.data["time"])
        np.save(os.path.join(self.save_path, "left.npy"), self.data["Left"])
        np.save(os.path.join(self.save_path, "right.npy"), self.data["Right"])
        
        self.save_path = None
        self.data = None

    def end(self):
        self.exit_event.set()
        self.recv_thread.join()
        
        if self.save_event.is_set():
            self.stop()
            self.save_event.clear()
            
        print("Xsens terminate")
        
    def is_error(self):
        return self.error_event.is_set()