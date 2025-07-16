import numpy as np
from scipy.spatial.transform import Rotation as R
from threading import Event, Thread, Lock
import zmq
import json
import os
from paradex.utils.file_io import config_dir
import time
import socket

occulus_hand_joint_name = ["palm", "wrist",
                      "thumb_metacarpal", "thumb_proximal", "thumb_distal", "thumb_tip",
                      "index_metacarpal", "index_proximal", "index_intermediate", "index_distal", "index_tip",
                      "middle_metacarpal", "middle_proximal", "middle_intermediate", "middle_distal", "middle_tip",
                      "ring_metacarpal", "ring_proximal", "ring_intermediate", "ring_distal", "ring_tip",
                      "pinky_metacarpal", "pinky_proximal", "pinky_intermediate", "pinky_distal", "pinky_tip"]


occulus_hand_joint_parent = [1, -1, 1, 2, 3, 4, 1, 6, 7, 8, 9, 1, 11, 12, 13, 14, 1, 16, 17, 18, 19, 1, 21, 22, 23, 24]
occulus_hand_joint_parent_name = {'palm': 'wrist', 'wrist': None,
                                  'thumb_metacarpal': 'wrist', 'thumb_proximal': 'thumb_metacarpal', 'thumb_distal': 'thumb_proximal', 'thumb_tip': 'thumb_distal',
                                  'index_metacarpal': 'wrist', 'index_proximal': 'index_metacarpal', 'index_intermediate': 'index_proximal', 'index_distal': 'index_intermediate', 'index_tip': 'index_distal',
                                  'middle_metacarpal': 'wrist', 'middle_proximal': 'middle_metacarpal', 'middle_intermediate': 'middle_proximal', 'middle_distal': 'middle_intermediate', 'middle_tip': 'middle_distal',
                                  'ring_metacarpal': 'wrist', 'ring_proximal': 'ring_metacarpal', 'ring_intermediate': 'ring_proximal', 'ring_distal': 'ring_intermediate', 'ring_tip': 'ring_distal',
                                  'pinky_metacarpal': 'wrist', 'pinky_proximal': 'pinky_metacarpal', 'pinky_intermediate': 'pinky_proximal', 'pinky_distal': 'pinky_intermediate', 'pinky_tip': 'pinky_distal'}

occulus_body_joint_name = ["Root", "Hips", "Spine Lower", "Spine Middle", "Spine Upper", "Chest", "Neck", "Head",                                                                   # 0:8
                           "Left Shoulder", "Left Scapula", "Left Arm Upper", "Left Arm Lower", "Left Hand Wrist Twist",                                                            # 8:13
                           "Right Shoulder", "Right Scapula", "Right Arm Upper", "Right Arm Lower", "Right Hand Wrist Twist",                                                       # 13:18
                           "Left Hand Palm", "Left Hand Wrist",                                                                                                                     # 18:44
                           "Left Hand Thumb Metacarpal", "Left Hand Thumb Proximal", "Left Hand Thumb Distal", "Left Hand Thumb Tip",
                           "Left Hand Index Metacarpal", "Left Hand Index Proximal", "Left Hand Index Intermediate", "Left Hand Index Distal", "Left Hand Index Tip",
                           "Left Hand Middle Metacarpal", "Left Hand Middle Proximal", "Left Hand Middle Intermediate", "Left Hand Middle Distal", "Left Hand Middle Tip",
                           "Left Hand Ring Metacarpal", "Left Hand Ring Proximal", "Left Hand Ring Intermediate", "Left Hand Ring Distal", "Left Hand Ring Tip",
                           "Left Hand Little Metacarpal", "Left Hand Little Proximal", "Left Hand Little Intermediate", "Left Hand Little Distal", "Left Hand Little Tip",
                           "Right Hand Palm", "Right Hand Wrist",                                                                                                                   # 44:70
                           "Right Hand Thumb Metacarpal", "Right Hand Thumb Proximal", "Right Hand Thumb Distal", "Right Hand Thumb Tip",
                           "Right Hand Index Metacarpal", "Right Hand Index Proximal", "Right Hand Index Intermediate", "Right Hand Index Distal", "Right Hand Index Tip",
                           "Right Hand Middle Metacarpal", "Right Hand Middle Proximal", "Right Hand Middle Intermediate", "Right Hand Middle Distal", "Right Hand Middle Tip",
                           "Right Hand Ring Metacarpal", "Right Hand Ring Proximal", "Right Hand Ring Intermediate", "Right Hand Ring Distal", "Right Hand Ring Tip",
                           "Right Hand Little Metacarpal", "Right Hand Little Proximal", "Right Hand Little Intermediate", "Right Hand Little Distal", "Right Hand Little Tip",
                           "Left Leg Upper", "Left Leg Lower", "Left Foot Ankle Twist", "Left Foot Ankle", "Left Foot Subtalar", "Left Foot Transverse", "Left Foot Ball",          # 70:77
                           "Right Leg Upper", "Right Leg Lower", "Right Foot Ankle Twist", "Right Foot Ankle", "Right Foot Subtalar", "Right Foot Transverse", "Right Foot Ball"]   # 77:84

occulus_body_joint_parent = [-1, 0, 1, 2, 3, 4, 5, 6,
                             5, 8, 9, 10, 11,
                             5, 13, 14, 15, 16,
                             19, 11, 19, 20, 21, 22, 19, 24, 25, 26, 27, 19, 29, 30, 31, 32, 19, 34, 35, 36, 37, 19, 39, 40, 41, 42,
                             45, 16, 45, 46, 47, 48, 45, 50, 51, 52, 53, 45, 55, 56, 57, 58, 45, 60, 61, 62, 63, 45, 65, 66, 67, 68,
                             1, 70, 71, 72, 73, 74, 75,
                             1, 77, 78, 79, 80, 81, 82]

occulus_upper_body_joint_parent = [-1, -1,
                                    3, -1, 3, 4, 5, 6, 3, 8, 9, 10, 11, 3, 13, 14, 15, 16, 3, 18, 19, 20, 21, 3, 23, 24, 25, 26,
                                    29, -1, 29, 30, 31, 32, 29, 34, 35, 36, 37, 29, 39, 40, 41, 42, 29, 44, 45, 46, 47, 29, 49, 50, 51, 52]

unity2Euclidian_mat = np.array([
                        [ 0,  0,  1, 0],   # new X = old Z
                        [-1,  0,  0, 0],   # new Y = –old X
                        [ 0,  1,  0, 0],   # new Z = old Y
                        [ 0,  0,  0, 1],
                    ], dtype=float)

unity2EuclidianPelvis_mat = np.array([
                                [0, 0, 1, 0],
                                [1, 0, 0, 0],
                                [0, 1, 0, 0],
                                [0, 0, 0, 1]
                            ], dtype=float) @ unity2Euclidian_mat


OCCULUS2WRIST_Left = np.array([[0, 0 ,1 ,0],
                                [-1, 0, 0, 0],
                                [0, 1, 0, 0],
                                [0, 0, 0, 1]]),
    
OCCULUS2WRIST_Right = np.array([[0, 0 ,1 ,0],
                                [1, 0, 0, 0],
                                [0, -1, 0, 0],
                                [0, 0, 0, 1]])
    
class OculusReceiver:
    def __init__(self, is_body=False):
        network_config = json.load(open(os.path.join(config_dir, "environment/network.json"), "r"))
        self.hand_pose = {}
        self.body_pose = None

        self.is_body = is_body
        
        self.host_ip = network_config["metaquest"]["ip"]
        self.port = network_config["metaquest"]["port"]
        
        self.stop_event = Event()
        self.lock = Lock()
        
        self.recv_thread = Thread(target=self.run, daemon=True)
        self.recv_thread.start()

    def get_data(self):
        with self.lock:
            if self.is_body:
                return self.body_pose.copy() if self.body_pose is not None else None
            else:
                ret = {}
                for id in ['Left','Right','Head','Root']:
                    if id not in self.hand_pose:
                        ret[id] = None
                    else:
                        if id in ['Left', 'Right']:
                            ret[id] = {joint_name: self.hand_pose[id][joint_id].copy()
                                        for joint_id, joint_name in enumerate(occulus_hand_joint_name)}
                        else:
                            ret[id] = self.hand_pose[id].copy()
                return ret
    
    def parse_posedata(self, pose_data):
        vector_list = pose_data.split("/")
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
            
        
    def parse_hand(self, token):
        # expect:
        # Left:0,0,0,0,0,0,0/1,1,1,1,1,1,1|Right:...(same as Left)
        ret = {}
        
        data_string = token.decode().strip()


        hand_string_list = data_string.split("|")[:-1]
        
        
        for hand_string in hand_string_list:
            hand_name = hand_string.split(":")[0]
            hand_data = hand_string.split(":")[1]
            hand_pose = self.parse_posedata(hand_data)
            
            ret[hand_name] = np.array(hand_pose)
            
        return ret
    
    def run(self):
        #context = zmq.Context()
        #self.socket = context.socket(zmq.PULL)
        #self.socket.setsockopt(zmq.CONFLATE, 1)
        #self.socket.bind('tcp://{}:{}'.format(self.host_ip, self.port))
        print("asdf")
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.bind((self.host_ip, self.port))
        self.sock.settimeout(0.1)
        while not self.stop_event.is_set():
            try:
                token, addr = self.sock.recvfrom(65536)  # 64KB maximally
                if self.is_body:
                    data_string = token.decode().strip()
                    if data_string == '':
                        continue
                    data = unity2Euclidian_mat @ np.array(self.parse_posedata(data_string)) @ unity2Euclidian_mat.T
                else:
                    data = self.parse_hand(token)
                    if 'Root' not in data:
                        continue
                    pelvis_pose = data['Root']

                    for name in data.keys():
                        data[name] = unity2EuclidianPelvis_mat @ (np.linalg.inv(pelvis_pose) @ data[name])
                        if name == "Right":
                            data[name] = data[name] @ np.linalg.inv(OCCULUS2WRIST_Right)
                        if name == "Left":
                            data[name] = data[name] @ np.linalg.inv(OCCULUS2WRIST_Left)

                with self.lock:
                    if self.is_body:
                        self.body_pose = data
                    else:
                        for name, pose in data.items():
                            self.hand_pose[name] = pose
            except socket.timeout:
                print("asdfasdf")
                continue

    def quit(self):
        print("quit")
        self.stop_event.set()
        self.recv_thread.join()
        self.sock.close()