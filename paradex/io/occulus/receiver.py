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
from paradex.geometry import conversions
from scipy.spatial.transform import Rotation as R
import time
import re
from multiprocessing import Lock
import zmq

host = "192.168.0.2"
port = 8088

class OculusReceiver:
    def __init__(self):
        context = zmq.Context()
        self.socket = context.socket(zmq.PULL)
        self.socket.setsockopt(zmq.CONFLATE, 1)
        self.socket.bind('tcp://{}:{}'.format(host, port))
        self.stream()

    def _process_data_token(self, data_token):
        return data_token.decode().strip()

    def _extract_data_from_token(self, token):        
        data = self._process_data_token(token)
        information = dict()
        keypoint_vals = [0] if data.startswith('absolute') else [1]
        # Data is in the format <hand>:x,y,z|x,y,z|x,y,z
        vector_strings = data.split(':')[1].strip().split('|')
        for vector_str in vector_strings:
            vector_vals = vector_str.split(',')
            for float_str in vector_vals[:3]:
                keypoint_vals.append(float(float_str))
            
        information['keypoints'] = keypoint_vals
        return information
    
    def stream(self):
        while True:
            try:
                token = self.socket.recv()
                
                information = self._extract_data_from_token(token)
                print(information)
            except zmq.Again:
                time.sleep(0.01)