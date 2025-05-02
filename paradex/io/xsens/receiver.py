import sys
import time
import argparse
# from IPython import embed
import numpy as np
import datetime
import time
import pickle
from copy import deepcopy
import socket
import time
import struct
import argparse

import numpy as np
from multiprocessing import shared_memory, Lock, Value, Event, Process
from paradex.utils import conversions
from . import hand_index
from scipy.spatial.transform import Rotation as R
import time
import re
from multiprocessing import Lock

host = "192.168.0.2"
port = 9763

class XSensReceiver:
    def __init__(self) -> None:
        self.state = -1 # moving to homepose
    
        self.shm = {}
        self.create_shared_memory("timestamp", 1 * np.dtype(np.float64).itemsize)
        self.create_shared_memory("timestamp_recv", 1 * np.dtype(np.float64).itemsize)
        self.create_shared_memory("pose_data_recv", 1 * np.dtype(np.float64).itemsize)
        self.create_shared_memory("hand_pose", 20 * 4 * 4 * np.dtype(np.float32).itemsize)
        self.create_shared_memory("hand_joint_angle", 20 * 3 * np.dtype(np.float32).itemsize)
        self.create_shared_memory("state", 1 * np.dtype(np.int32).itemsize)
        

        self.data_array = {
            "timestamp": np.ndarray((1,), dtype=np.float64, buffer=self.shm["timestamp"].buf),
            "timestamp_recv": np.ndarray((1,), dtype=np.float64, buffer=self.shm["timestamp_recv"].buf),
            "pose_data_recv": np.ndarray((1,), dtype=np.float64, buffer=self.shm["pose_data_recv"].buf),
            "hand_pose": np.ndarray((20, 4, 4), dtype=np.float32, buffer=self.shm["hand_pose"].buf),
            "hand_joint_angle": np.ndarray((20, 3), dtype=np.float32, buffer=self.shm["hand_joint_angle"].buf),
            "state": np.ndarray((1,), dtype=np.int32, buffer=self.shm["state"].buf),
        }
        self.data_array["state"][0] = -1
        self.stop_event = Event()
        self.timestamp_hist = []
        self.xsems_timestamp_hist = []


        self.init_server(host, port)

    def create_shared_memory(self, name, size):
        try:
            existing_shm = shared_memory.SharedMemory(name=name)
            existing_shm.close()
            existing_shm.unlink()
        except FileNotFoundError:
            pass
        self.shm[name] = shared_memory.SharedMemory(create=True, name=name, size=size)
        
    def init_server(self, host_ip=None, port=None) -> None:
        self.host_ip = host_ip
        self.port = port
        
        self.exit = False
        self.recv_process = Process(target=self.recv_data)
        self.recv_process.start()

    def pose_to_jointangle(self, pose_data, parent_id):
        euler_angle = np.zeros((len(pose_data), 3))#[0 for i in range(len(pose_data))]
        for i in range(len(pose_data)):
            if parent_id[i] == -1:
                continue
            euler_angle[i] = R.from_matrix(
                pose_data[0][:3, :3].T @ pose_data[i][:3, :3]
            ).as_euler('zyx')

        return euler_angle

    def check_straight(self, pose_data):
        ret = [True, True, True, True] # if one of the joints is not straight, return False

        for finger_id in range(4):
            for joint_num in range(3):
                joint_id = finger_id * 4 + joint_num + 5

                if joint_num == 0:
                    if pose_data[joint_id][2,1] < -0.8:
                        ret[finger_id] = False
                else:
                    rel_pose = np.linalg.inv(pose_data[hand_index.hand_index_parent[joint_id]]) @ pose_data[joint_id]
                    if rel_pose[2,1] < -0.8:
                        ret[finger_id] = False
        return ret
    
    def get_state(self, pose_data):
        straight = self.check_straight(pose_data)
        if straight[0] and straight[1] and not straight[2] and not straight[3]: # V pose
            return 2
        
        if straight[0] and straight[1] and straight[2] and straight[3]: # Fist pose
            return 1
        
        if straight[0] and not straight[1] and not straight[2] and not straight[3]: # Open pose
            return 3
        
        return 0
            
    def get_data(self):
        """Returns a dictionary with the latest shared memory data."""
        return {
            "timestamp": self.data_array["timestamp"][0],
            "timestamp_recv": self.data_array["timestamp_recv"][0],
            "pose_data_recv": self.data_array["pose_data_recv"][0],
            "hand_pose": self.data_array["hand_pose"].copy(),  # Copy to avoid memory corruption
            "hand_joint_angle": self.data_array["hand_joint_angle"].copy(),
            "state": self.data_array["state"][0],
        }

    def recv_data(self):
        # Define the server socket
        if self.host_ip is not None:
            self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            self.server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            self.server_socket.bind((self.host_ip, self.port))

        
        
        header_format = "!6s I B B I B B B B 2s H"
        header_struct = struct.Struct(header_format)
        
        while not self.stop_event.is_set():
            data, addr = self.server_socket.recvfrom(4096)
            recv_time = time.time() * 1000

            data_header = header_struct.unpack(data[:24])  # header is 24 bytes
            
            header_id_string = data_header[0].decode("utf-8")
            assert header_id_string[:4] == "MXTP"
            message_id = int(header_id_string[4:])
            
            
            if message_id == 25:
                _new_xsens_time = list(map(int, re.split(r'[:.]', data[28:].decode("utf-8"))))
                new_xsens_time = (_new_xsens_time[0] * 3600 + _new_xsens_time[1] * 60 + _new_xsens_time[2]) * 1000 + _new_xsens_time[3]

                self.data_array["timestamp"][0] = new_xsens_time
                self.data_array["timestamp_recv"][0] = recv_time

                self.xsems_timestamp_hist.append(new_xsens_time)
                self.timestamp_hist.append(recv_time)

            elif message_id == 2:
                pose_data_parsed = self.pose_data_parser(data[24:])
                if pose_data_parsed[1] == "bodyhands":
                    
                    right_hand_pose = pose_data_parsed[0][23+20:23+40].copy()
                    pelvis_pose = pose_data_parsed[0][0].copy()
                    for i in range(0, 20):
                        right_hand_pose[i] = np.linalg.inv(pelvis_pose) @ right_hand_pose[i]
                    
                    
                    if np.linalg.norm(right_hand_pose[0][:3,3] - self.data_array["hand_pose"][0][:3,3]) > 0.5 and self.data_array["state"][0] != -1:
                        continue # outlier  
                    

                    left_hand_pose = pose_data_parsed[0][23+0:23+20].copy()
                    
                    for i in range(1, 20):
                        left_hand_pose[i] = np.linalg.inv(left_hand_pose[0]) @ left_hand_pose[i]

                    left_hand_pose[0] = np.eye(4)

                    state = self.get_state(left_hand_pose)
                    
                    self.data_array["hand_pose"][:, :, :] = right_hand_pose.copy()
                
                    self.data_array["hand_joint_angle"][:, :] = self.pose_to_jointangle(
                        right_hand_pose, hand_index.hand_index_parent
                    )
                    self.data_array["pose_data_recv"][0] = recv_time
                    self.data_array["state"][0] = state
        
        self.server_socket.close()
        np.save("timestamp_hist.npy", self.timestamp_hist)
        np.save("xsems_timestamp_hist.npy", self.xsems_timestamp_hist)
        print("Server closed")


    def quit(self):
        print("Closing server")
        self.stop_event.set()
        self.recv_process.join()

        for shm_name, shm in self.shm.items():
            shm.close()
            shm.unlink()

    def pose_data_parser(self, pose_data):
        data_idx = 0
        total_data_len = len(pose_data)
        pose_data_per_frame = []

        num_seg = int(total_data_len / 32)
        pose_data_type = "unknown"

        if num_seg == 23:
            pose_data_type = "body"
        if num_seg == 63:
            pose_data_type = "bodyhands"
        if num_seg < 4:
            pose_data_type = "vive"
        while data_idx < total_data_len:
            pose_seg = pose_data[data_idx : data_idx + 32]
            decoded_pose_seg = struct.unpack("!I 3f 4f", pose_seg)
            xyz = np.array(decoded_pose_seg[1:4])
            try:
                quaternion = np.array([*decoded_pose_seg[5:], decoded_pose_seg[4]])
                R = conversions.Q2R(quaternion)
            except:
                R = np.eye(3)  # TODO fix
            pose_T = conversions.Rp2T(R, xyz)
            pose_data_per_frame.append(pose_T)
            data_idx += 32
        return pose_data_per_frame, pose_data_type

if __name__ == "__main__":
    host = "192.168.0.2"
    port = 9763

    xsens_updater = XSensReceiver()
    xsens_updater.init_server(host, port)
    start_time = time.time()
    cnt = 0
    while True:
        cnt += 1
        # print(xsens_updater.get_data(), 1 / (time.time()-start_time))
        start_time = time.time()

    xsens_updater.quit()

