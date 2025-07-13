import time
from scipy.spatial.transform import Rotation
import transforms3d as t3d

import numpy as np
from threading import Thread, Event, Lock
import json
import os

from xarm.wrapper import XArmAPI

from paradex.utils.file_io import config_dir, rsc_path
from paradex.robot import RobotWrapper

# get_joints_torque : torque(Nm)
# set_allow_approx_motion
"""
Settings allow to avoid overspeed near some singularities using approximate solutions
Note:
    1. only available if firmware_version >= 1.9.0

:param on_off: allow or not, True means allow, default is False
"""
# set_joint_maxjerk
# set_joint_maxacc

# Todo : Find PID value

action_dof = 6

class XArmController:
    def __init__(self, save_path=None):
        network_config = json.load(open(os.path.join(config_dir, "environment/network.json"), "r"))
        self.xarm_ip_address = network_config["xarm"]
        
        self.reset()
        
        self.capture_path = save_path
        if save_path is not None:
            os.makedirs(self.capture_path, exist_ok=True)

        self.lock = Lock()
        self.exit = Event()
        
        T = 60000
        self.data = {
            "time":np.zeros((T,1), dtype=np.float64),
            "position":np.zeros((T, action_dof), dtype=np.float64),
            "velocity":np.zeros((T, action_dof), dtype=np.float64),
            "torque":np.zeros((T, action_dof), dtype=np.float64),
            "action":np.zeros((T, 4, 4), dtype=np.float64),
            "action_qpos":np.zeros((T, action_dof), dtype=np.float64) # x y z rpy
        }
        
        self.cnt = 0

        self.target_action = np.array([
                                [0, 1 ,0, 300],
                                [0, 0, 1, -200],
                                [1, 0, 0, 200],
                                [0, 0, 0, 1]]
                                )
        
        self.homing = False
        
        self.init = False
        self.fps = 50
        self.robot_model = RobotWrapper(os.path.join(rsc_path, "robot", "xarm.urdf"))
        self.last_link_id = self.robot_model.get_link_index("link6")
        
        self.thread = Thread(target=self.move_arm)
        self.thread.daemon = True
        self.thread.start()

    def is_ready(self):
        return not self.homing

    def home_robot(self, homepose):
        assert homepose.shape == (4,4) or homepose.shape == (6,)
        if homepose.shape == (6,):
            self.robot_model.compute_forward_kinematics(homepose.copy())
            homepose = self.robot_model.get_link_pose(self.last_link_id)
        with self.lock:
            self.init = True
            self.homing = True
            self.target_action = homepose.copy()
    
    def set_action(self, action):
        assert action.shape == (4,4)
        
        if homepose.shape == (6,):
            self.robot_model.compute_forward_kinematics(homepose.copy)
            homepose = self.robot_model.get_link_pose(self.last_link_id)
            
        with self.lock:
            self.init = True
            self.target_action = action.copy()
            
    def homo2cart(self, h):
        t = h[:3, 3] * 1000
        R = h[:3, :3]

        rpy = Rotation.from_matrix(R).as_euler("xyz")
        return np.concatenate([t, rpy])
    
    def homo2aa(self, h):
        t = h[:3, 3] * 1000
        R = h[:3, :3]

        axis, angle = t3d.axangles.mat2axangle(R)
        axis_angle = axis * angle
        return np.concatenate([t, axis_angle])
    
    def get_qpos(self):
        with self.lock:
            qpos = np.array(self.arm.get_joint_states(is_radian=True)[1][0])[:6]
            return qpos
    
    def get_position(self):
        with self.lock:
            cart = np.array(self.arm.get_position(is_radian=True)[1])
            pos = np.eye(4)
            
            pos[:3,3] = cart[:3] / 1000
            pos[:3, :3] = Rotation.from_euler('xyz', cart[3:]).as_matrix()
            
            return pos
        
    def reset(self):
        self.arm = XArmAPI(self.xarm_ip_address, report_type="devlop")
        self.arm.motion_enable(enable=False)
        self.arm.motion_enable(enable=True)
        self.arm.set_mode(0)
        self.arm.set_state(state=0)
        
        if self.arm.has_err_warn:
            self.arm.clean_error()

        self.arm.motion_enable(enable=False)
        self.arm.motion_enable(enable=True)
        self.arm.set_mode(0)  # 0: position control, 1: servo control
        self.arm.set_state(state=0)
        time.sleep(0.1)
        
        
        self.arm.set_mode(1)
        self.arm.set_state(state=0)
        
        # self.arm.set_collision_sensitivity(5, wait=True)
         
    def move_arm(
        self
    ):
        # Control loop start
        while not self.exit.is_set():
            start_time = time.time()
            with self.lock:
                if not self.init:
                    time.sleep(0.01)
                    continue
                    
                action = self.target_action.copy()
                cart = self.homo2cart(action.copy())
                aa = self.homo2aa(action.copy())
                
                if self.homing:
                    self.arm.set_mode(0)  # 0: position control, 1: servo control
                    self.arm.set_state(state=0)
                    
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
                    time.sleep(0.1)
                    
                    self.homing = False
                
                else:
                    _, state = self.arm.get_joint_states(is_radian=True)
                    
                    self.data["position"][self.cnt] = np.array(state[0])[:6]
                    self.data["velocity"][self.cnt] = np.array(state[1])[:6]
                    self.data["torque"][self.cnt] = np.array(state[2])[:6]
                    self.data["time"][self.cnt, 0] = np.array(start_time)
                    self.data["action"][self.cnt] = action.copy()
                    self.data["action_qpos"] = np.array(self.arm.get_inverse_kinematics(cart)[1])[:6]
                    self.cnt += 1
                    # print(self.arm.get_position(is_radian=True)[1], cart, "asdf")
                    #self.arm.set_servo_cartesian(cart.copy(), is_radian=True)
                    self.arm.set_servo_cartesian_aa(aa, is_radian=True)
                    
            end_time = time.time()
            time.sleep(max(0, 1 / self.fps - (end_time - start_time)))

            

    def save(self):
        with self.lock:
            if self.capture_path is not None:       
                os.makedirs(os.path.join(self.capture_path, "arm"), exist_ok=True)
                for name, value in self.data.items():                     
                    np.save(os.path.join(self.capture_path, "arm", f"{name}.npy"), value[:self.cnt])
                    
    def quit(self):
        self.save()
        self.exit.set()
        self.thread.join()
        self.arm.motion_enable(enable=False)
        self.arm.disconnect()
        print("robot terminate")
        
