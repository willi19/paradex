import time
from scipy.spatial.transform import Rotation
import transforms3d as t3d

import numpy as np
from threading import Thread, Event, Lock
import json
import os

from xarm.wrapper import XArmAPI

action_dof = 6

def homo2cart(h):
    t = h[:3, 3] * 1000
    R = h[:3, :3]

    rpy = Rotation.from_matrix(R).as_euler("xyz")
    return np.concatenate([t, rpy])

def cart2homo(cart):
    pos = np.eye(4)
    pos[:3,3] = cart[:3] / 1000
    pos[:3, :3] = Rotation.from_euler('xyz', cart[3:]).as_matrix()
    return pos

def homo2aa(h):
    t = h[:3, 3] * 1000
    R = h[:3, :3]

    axis, angle = t3d.axangles.mat2axangle(R, unit_thresh=0.001)
    axis_angle = axis * angle
    return np.concatenate([t, axis_angle])

class XArmController:
    def __init__(self, ip):
        self.xarm_ip_address = ip
        
        self.fps = 100
        self.reset()
        
        self.lock = Lock()
        
        self.exit_event = Event()
        self.save_event = Event()
        self.error_event = Event()
        self.connect_event = Event()
        self.position_control_event = Event()
        
        self.thread = Thread(target=self.control_loop)
        self.thread.start()
        
        self.connect_event.set()
        
    def control_loop(self):
        with self.lock:
            self.action = np.array(self.arm.get_position(is_radian=True)[1][0])[:6]
        self.is_servo = True
        self.connect_event.set()
        
        while not self.exit_event.is_set():
            start_time = time.perf_counter()
            if self.arm.has_err_warn:
                self.error_event.set()
                
            with self.lock:
                action = self.action.copy()
                is_servo = self.is_servo
            
            is_joint_value = (action.shape == (6,))
            
            if not is_servo:
                self.arm.set_mode(0)  # 0: position control, 1: servo control
                self.arm.set_state(state=0)
                    
                if is_joint_value:
                    self.arm.set_servo_angle(angle=action.tolist(), is_radian=True, wait=True)
                else:
                    cart = homo2cart(action)
                    self.arm.set_position(x = cart[0],
                                            y = cart[1],
                                            z = cart[2],
                                            roll = cart[3],
                                            pitch = cart[4],
                                            yaw = cart[5],
                                            speed=100, 
                                            is_radian=True, 
                                            wait=True) # motion_type=1 if necessary to go home but this is too dangerous
                
                self.arm.set_mode(1)
                self.arm.set_state(state=0)
                self.position_control_event.set()
            
            else:
                if is_joint_value:
                    self.arm.set_servo_angle_j(angles=action.tolist(), is_radian=True)
                else:
                    aa = homo2aa(action)
                    self.arm.set_servo_cartesian_aa(aa, is_radian=True)
            
            if self.save_event.is_set():
                _, state = self.arm.get_joint_states(is_radian=True)
                self.data["position"].append(np.array(state[0])[:6])
                self.data["velocity"].append(np.array(state[1])[:6])
                self.data["torque"].append(np.array(state[2])[:6])
                self.data["time"].append(np.array(start_time))
                self.data["action"].append(action.copy())
                
                success, ik = self.arm.get_inverse_kinematics(cart)
                if success != 0:
                    print("ik not success")
                    ik = - np.ones(6)
                    
                self.data["action_qpos"].append(np.array(ik)[:6])
                    
            elapsed = time.perf_counter() - start_time
            time.sleep(max(0, (1 / self.fps) - elapsed))

    def start(self, save_path):            
        self.save_path = save_path
        os.makedirs(os.path.join(self.save_path), exist_ok=True)
        
        self.data = {
            "time":[],# np.zeros((T,1), dtype=np.float64),
            "position":[],# np.zeros((T, action_dof), dtype=np.float64),
            "velocity":[],# np.zeros((T, action_dof), dtype=np.float64),
            "torque":[],# np.zeros((T, action_dof), dtype=np.float64),
            "action":[], #np.zeros((T, 4, 4), dtype=np.float64),
            "action_qpos":[] #np.zeros((T, action_dof), dtype=np.float64) # x y z rpy
        }
        
        with self.lock:
            self.save_event.set()

    def stop(self):
        with self.lock:
            self.save_event.clear()
        for name, value in self.data.items():          
            np.save(os.path.join(self.save_path, f"{name}.npy"), np.array(value))
            self.data[name] = []
        self.save_path = None
        
    def end(self, set_break=False):
        self.exit_event.set()
        self.thread.join()
        
        if set_break:
            self.arm.motion_enable(enable=False)
        self.arm.disconnect()
        if self.save_event.is_set():
            self.stop()
    
    def move(self, action, is_servo=True):
        assert action.shape == (4,4) or action.shape == (6,)
        
        with self.lock:
            self.position_control_event.clear()
            self.action = action.copy()
            self.is_servo = is_servo
        
        if not is_servo:
            self.position_control_event.wait()
        
    def reset(self):
        self.arm = XArmAPI(self.xarm_ip_address, report_type="devlop")
        if self.arm.has_err_warn:
            self.arm.clean_warn()
            self.arm.clean_error()
            
        self.arm.motion_enable(enable=True)
        self.arm.set_mode(0)
        self.arm.set_state(state=0)
        time.sleep(0.1)
        
        
        self.arm.set_mode(1)
        self.arm.set_state(state=0)
    
    def get_data(self):
        with self.lock:
            qpos = np.array(self.arm.get_joint_states(is_radian=True)[1][0])[:6]
            cart = np.array(self.arm.get_position(is_radian=True)[1])

        pos = cart2homo(cart)
        return {
            "qpos": qpos,
            "position": pos,
            "time": time.time()
        }
    
    def is_error(self):
        return self.error_event.is_set()