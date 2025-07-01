import numpy as np
from scipy.spatial.transform import Rotation as R
from threading import Event, Thread, Lock
import zmq
import json
import os
from paradex.utils.file_io import config_dir
import time

occulus_joint_name = ["wrist", "forearmstub", 
              "thumb_trapezium", "thumb_metacarpal", "thumb_proximal", "thumb_distal", 
              "index_proximal", "index_intermediate", "index_distal",
              "middle_proximal", "middle_intermediate", "middle_distal",
              "ring_proximal", "ring_intermediate", "ring_distal",
              "pinky_metacarpel", "pinky_proximal", "pinky_intermediate", "pinky_distal",
              "thumb_tip", "index_tip", "middle_tip", "ring_tip", "pinky_tip"]

occulus_joint_parent = [-1, 0, 0, 2, 3, 4, 0, 6, 7, 0, 9, 10, 0, 12, 13, 0, 15, 16, 17, 5, 8, 11, 14, 18]

class OculusReceiver:
    def __init__(self):
        network_config = json.load(open(os.path.join(config_dir, "environment/network.json"), "r"))
        self.hand_pose = {}
        
        self.host_ip = "0.0.0.0"
        self.port = network_config["metaquest"]
        
        self.stop_event = Event()
        self.lock = Lock()
        
        self.recv_thread = Thread(target=self.run)
        self.recv_thread.start()

    def get_data(self):
        with self.lock:
            return {
                "Left": self.hand_pose["Left"].copy() if 'Left' in self.hand_pose else None,
                "Right": self.hand_pose["Right"].copy() if 'Right' in self.hand_pose else None,
            }
    
    def parse_handdata(self, hand_data):
        vector_list = hand_data.split("/")
        ret = []
        
        for i, vector_string in enumerate(vector_list):
            vector = [float(v) for v in vector_string.split(",")]
            quat = np.array(vector[3:])
            pos = np.array(vector[:3])
            pose_T = np.eye(4)
            
            pose_T[:3,3] = pos
            pose_T[:3,:3] = R.from_quat(quat).as_matrix()
            ret.append(pose_T)
        return ret
            
        
    def parse(self, token):
        ret = {}
        
        data_string = token.decode().strip()
        hand_string_list = data_string.split("|")[:-1]
        
        
        for hand_string in hand_string_list:
            hand_name = hand_string.split(":")[0]
            hand_data = hand_string.split(":")[1]
            hand_pose = self.parse_handdata(hand_data)
            
            ret[hand_name] = np.array(hand_pose)
            
        return ret
    
    def run(self):
        context = zmq.Context()
        self.socket = context.socket(zmq.PULL)
        self.socket.setsockopt(zmq.CONFLATE, 1)
        self.socket.bind('tcp://{}:{}'.format(self.host_ip, self.port))

        while not self.stop_event.is_set():
            try:
                token = self.socket.recv(flags=zmq.NOBLOCK)
                data = self.parse(token)
                with self.lock:
                    for name, pose in data.items():
                        self.hand_pose[name] = pose
            except zmq.Again:
                time.sleep(0.01)
        

    def quit(self):
        print("quit")
        self.stop_event.set()
        self.recv_thread.join()