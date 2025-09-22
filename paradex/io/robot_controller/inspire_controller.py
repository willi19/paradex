import time
import serial
from threading import Thread, Event, Lock
import numpy as np
import os
import json


from paradex.utils.file_io import config_dir

action_dof = 6
hand_id = 1
command = {
    'setpos':[0xEB, 0x90, hand_id, 0x0F, 0x12, 0xC2, 0x05],
    'setangle':[0xEB, 0x90, hand_id, 0x0F, 0x12, 0xCE, 0x05],
    'setpower':[0xEB, 0x90, hand_id, 0x0F, 0x12, 0xDA, 0x05],
    'setspeed':[0xEB, 0x90, hand_id, 0x0F, 0x12, 0xF2, 0x05],
    'getsetspeed':[0xEB, 0x90, hand_id, 0x04, 0x11, 0xC2, 0x05, 0x0C],
    'getsetangle':[0xEB, 0x90, hand_id, 0x04, 0x11, 0xCE, 0x05, 0x0C],
    'getsetpower':[0xEB, 0x90, hand_id, 0x04, 0x11, 0xDA, 0x05, 0x0C],
    'getactpos':[0xEB, 0x90, hand_id, 0x04, 0x11, 0xFE, 0x05, 0x0C],
    'getactangle':[0xEB, 0x90, hand_id, 0x04, 0x11, 0x0A, 0x06, 0x0C],
    'getactforce':[0xEB, 0x90, hand_id, 0x04, 0x11, 0x2E, 0x06, 0x0C],
}

def data2bytes(data):
    data = int(data)
    rdata = [0xff]*2
    if data == -1:
        rdata[0] = 0xff
        rdata[1] = 0xff
    else:
        rdata[0] = data&0xff
        rdata[1] = (data>>8)&(0xff)
    return rdata

def num2str(num):
    str = hex(num)
    str = str[2:4]
    if(len(str) == 1):
        str = '0'+ str
    str = bytes.fromhex(str)    
    return str

def checknum(data):
    result = 0
    for v in data[2:-1]:
        result += v
    result = result&0xff
    return result

def data2str(data):
    ret = b''
    for v in data:
        ret = ret + num2str(v)
    return ret

class InspireController:
    def __init__(self, ):
        self.capture_path = None
        
        self.home_pose = np.zeros(action_dof)+800
        
        self.exit = Event()
        self.lock = Lock()
        self.target_action = np.zeros(action_dof)+800
        
        self.thread = Thread(target=self.move_hand)
        self.thread.daemon = True
        self.thread.start()

    def start(self, save_path=None):
        with self.lock:
            self.capture_path=save_path
            self.data = {
                "time":[],#np.zeros((T,1), dtype=np.float64),
                "position":[],#np.zeros((T, action_dof), dtype=np.float64),
                "action":[],#np.zeros((T, action_dof), dtype=np.float64),
                "force":[]#np.zeros((T, action_dof), dtype=np.float64)
            }
            
    def end(self):
        self.save()
                    
    def set_homepose(self, home_pose):
        assert home_pose.shape == (action_dof,)
        self.home_pose = home_pose.copy()

    def home_robot(self, home_pose=None):
        if home_pose is None:
            self.target_action = self.home_pose.copy()
        else:
            self.home_pose = home_pose.copy()
            self.target_action = home_pose.copy()
        
    def open_serial(self):
        self.ser=serial.Serial('/dev/ttyUSB0',115200)
        return 
        
    def write6(self, command_name, value):
        datanum = command[command_name][3]
        len_command = len(command[command_name])
        
        b = [0] * (datanum + 5)
        
        for i, v in enumerate(command[command_name]):
            b[i] = v
        
        for i in range(6):
            b[len_command + 2 * i] = data2bytes(value[i])[0]
            b[len_command + 2 * i + 1] = data2bytes(value[i])[1]
        
        b[-1] = checknum(b)
        putdata = data2str(b)
        self.ser.write(putdata)
        getdata = self.ser.read(9)
        
    def read6(self, command_name):
        datanum = command[command_name][3]
        len_command = len(command[command_name])
        
        b = [0] * (datanum+5)
        for i, v in enumerate(command[command_name]):
            b[i] = v
        b[-1] = checknum(b)
        putdata = data2str(b)
        self.ser.write(putdata)
        
        getdata = self.ser.read(20)
        ret = np.zeros(6)
        
        for i in range(6):
            if getdata[i*2+7] == 0xff and getdata[i*2+8] == 0xff:
                ret[i] = -1
            else:
                ret[i] = getdata[i*2+7] + (getdata[i*2+8]<<8)
        return ret
    
    def get_qpos(self):
        with self.lock:
            current_hand_angles = np.asarray(self.read6('getactangle'))            
            return current_hand_angles            
    
    def get_force(self):
        with self.lock:
            current_force = np.asarray(self.read6('getactforce'))
            return current_force
        
    def move_hand(self):
        self.fps = 100
        self.open_serial()
        self.hand_lock = Lock()

        self.write6('setspeed', [1000, 1000, 1000, 1000, 1000, 1000])
        self.write6('setpower', [200, 200, 200, 200, 200, 200])
        self.write6('setangle', [1000, 1000, 1000, 1000, 1000, 1000])
        
        while not self.exit.is_set():
            start_time = time.time()
            with self.lock:
                action = self.target_action.copy().astype(np.int32)
                self.write6('setangle', action)

            current_hand_angles = np.asarray(self.read6('getactangle'))
            current_force = np.asarray(self.read6('getactforce'))
            # current_action = np.asarray(self.read6('angleSet'))
            with self.lock:
                if self.capture_path is not None:
                    self.data["position"].append(current_hand_angles.copy())
                    self.data["time"].append(start_time)
                    self.data["action"].append(action.copy())
                    self.data["force"].append(current_force.copy())
            
            end_time = time.time()
            time.sleep(max(0, 1 / self.fps - (end_time - start_time)))

    def set_target_action(self, action):
        with self.lock:
            self.target_action = action.copy()
    
    def save(self):
        with self.lock:
            if self.capture_path is not None:       
                os.makedirs(os.path.join(self.capture_path), exist_ok=True)
                for name, value in self.data.items():                     
                    np.save(os.path.join(self.capture_path, f"{name}.npy"), np.array(value))
                    self.data[name] = value
            self.capture_path = None
                                    
    def quit(self):
        self.exit.set()
        self.save()
        self.thread.join()
        print("Inspire Exiting...")
