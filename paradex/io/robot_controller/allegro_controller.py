import time
import numpy as np
from threading import Thread, Event, Lock

import os

import rospy
from sensor_msgs.msg import JointState

from copy import deepcopy as copy

# List of all ROS Topics
JOINT_STATE_TOPIC = '/allegroHand/joint_states' 
GRAV_COMP_TOPIC = '/allegroHand/grav_comp_torques' 
COMM_JOINT_STATE_TOPIC = '/allegroHand/commanded_joint_states' 
JOINT_COMM_TOPIC = '/allegroHand/joint_cmd'

# Maximum permitted values
MAX_ANGLE = 2.1
MAX_TORQUE = 0.3

DEFAULT_VAL = None
action_dof = 16

class AllegroController:
    def __init__(self):
        
        self.current_joint_pose = DEFAULT_VAL
        self.grav_comp = DEFAULT_VAL
        self.cmd_joint_state = DEFAULT_VAL
        
        rospy.init_node('allegro_hand_node')
        
        self.save_event = Event()
        self.exit_event = Event()
        self.connection_event = Event()
        
        self.lock = Lock()       
        self.thread = Thread(target=self.control_loop)
        self.thread.start()
        
        self.connection_event.wait()
        
    def control_loop(self):
        rospy.Subscriber(JOINT_STATE_TOPIC, JointState, self._sub_callback_joint_state)
        rospy.Subscriber(GRAV_COMP_TOPIC, JointState, self._sub_callback_grav_comp)
        rospy.Subscriber(COMM_JOINT_STATE_TOPIC, JointState, self._sub_callback_cmd__joint_state)

        self.joint_comm_publisher = rospy.Publisher(JOINT_COMM_TOPIC, JointState, queue_size=-1)
        self.connection_event.wait()
        
        while not self.exit_event.is_set():
            start_time = time.perf_counter()
            joint_value = np.asarray(self.current_joint_pose.position)
            
            with self.lock:
                action = self.action.copy()
                joint_value = joint_value.copy()

            self._publish_action(action, absolute=True)
            
            if self.save_event.is_set():
                self.data["action"].append(action.copy())
                self.data["time"].append(time.time())
                self.data["joint_value"].append(joint_value.copy())

            elapsed = time.perf_counter() - start_time
            time_to_wait = max(0.0, 0.01 - elapsed)
            time.sleep(time_to_wait)

    def start(self, save_path):
        self.capture_path = save_path
        self.data = {
            "action": [],
            "time": [],
            "joint_value": []
        }
        
        self.save_event.set()

    def stop(self):
        self.save_event.clear()
        
        os.makedirs(os.path.join(self.capture_path), exist_ok=True)
        for name, value in self.data.items():                     
            np.save(os.path.join(self.capture_path, f"{name}.npy"), np.array(value))
                    
        self.capture_path = None
        
    def end(self):
        self.exit_event.set()
        self.thread.join()
        
        self.stop()
        
    def get_data(self):
        ret = {}
        with self.lock:
            ret["action"] = self.action.copy()
            ret["joint_value"] = self.joint_value.copy()
            ret["time"] = time.time()
            
        return ret
        
    def move(self, action):
        with self.lock:
            self.action = action.copy()
    
    def _sub_callback_joint_state(self, data):
        if not self.connection_event.is_set():
            with self.lock:
                self.action = np.asarray(data.position)
                self.joint_value = np.asarray(data.position)
                self.connection_event.set()
                
        self.current_joint_pose = data

    def _sub_callback_grav_comp(self, data):
        self.grav_comp = data

    def _sub_callback_cmd__joint_state(self, data):
        self.cmd_joint_state = data
    
    def _clip(self, action, value):
        return np.clip(action, -value, value)
    
    def _publish_action(self, desired_action = np.zeros(action_dof), absolute = True):
        if self.current_joint_pose == DEFAULT_VAL:
            print('No joint data received!')
            return

        action = self._clip(desired_action, MAX_ANGLE)
        current_angles = self.current_joint_pose.position
        
        if absolute is True:
            desired_angles = np.array(action)
        else:
            desired_angles = np.array(action) + np.array(current_angles)
        
        desired_js = copy(self.current_joint_pose)
        desired_js.position = list(desired_angles)
        desired_js.effort = list([])

        self.joint_comm_publisher.publish(desired_js)
    
    def is_error(self):
        return False