import time
import numpy as np
from multiprocessing import Process, shared_memory, Event, Lock

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
    def __init__(self, save_path=None):
        
        self.current_joint_pose = DEFAULT_VAL
        self.grav_comp = DEFAULT_VAL
        self.cmd_joint_state = DEFAULT_VAL
    
        self.allegro_home_pose = None
        
        self.capture_path = save_path
        if save_path is not None:
            os.makedirs(self.capture_path, exist_ok=True)

        self.hand_state_hist = np.zeros((60000, action_dof), dtype=np.float64)
        self.hand_timestamp = np.zeros((60000, 1), dtype=np.float64)        
        self.hand_action_hist = np.zeros((60000, action_dof), dtype=np.float64)


        self.exit = Event()

        self.shm = {}
        self.create_shared_memory("hand_target_action", action_dof * np.dtype(np.float32).itemsize)
        self.hand_target_action_array = np.ndarray((action_dof,), dtype=np.float32, buffer=self.shm["hand_target_action"].buf)

        self.hand_process = Process(target=self.move_hand)
        self.hand_process.start()


    def _sub_callback_joint_state(self, data):
        self.current_joint_pose = data

    def _sub_callback_grav_comp(self, data):
        self.grav_comp = data

    def _sub_callback_cmd__joint_state(self, data):
        self.cmd_joint_state = data
    
    def _clip(self, action, value):
        return np.clip(action, -value, value)
    
    def move(self, desired_action = np.zeros(action_dof), absolute = True):
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

    def set_homepose(self, allegro_home_pose):
        self.allegro_home_pose = allegro_home_pose.copy()
    
    def create_shared_memory(self, name, size):
        try:
            existing_shm = shared_memory.SharedMemory(name=name)
            existing_shm.close()
            existing_shm.unlink()
        except FileNotFoundError:
            pass
        self.shm[name] = shared_memory.SharedMemory(create=True, name=name, size=size)

    def home_robot(self):
        self.hand_target_action_array[:] = self.allegro_home_pose.copy()
        
    def move_hand(self):
        fps = 100
        self.hand_cnt = 0
        self.hand_lock = Lock()
        try:
            rospy.init_node('allegro_hand_node')
        except:
            pass

        rospy.Subscriber(JOINT_STATE_TOPIC, JointState, self._sub_callback_joint_state)
        rospy.Subscriber(GRAV_COMP_TOPIC, JointState, self._sub_callback_grav_comp)
        rospy.Subscriber(COMM_JOINT_STATE_TOPIC, JointState, self._sub_callback_cmd__joint_state)

        self.joint_comm_publisher = rospy.Publisher(JOINT_COMM_TOPIC, JointState, queue_size=-1)

        while not self.exit.is_set():
        
            start_time = time.time()
            action = self.hand_target_action_array.copy()
            if self.current_joint_pose == DEFAULT_VAL:
                current_hand_angles = np.zeros(action_dof)
            else:
                current_hand_angles = np.asarray(self.current_joint_pose.position)
            self.move(action)
            
            self.hand_state_hist[self.hand_cnt] = current_hand_angles.copy()
            self.hand_timestamp[self.hand_cnt] = start_time

            self.hand_action_hist[self.hand_cnt] = self.hand_target_action_array.copy()
            self.hand_cnt += 1

            end_time = time.time()
            time.sleep(max(0, 1 / fps - (end_time - start_time)))
        
        if self.capture_path is not None:
            os.makedirs(os.path.join(self.capture_path, "hand"), exist_ok=True)
            np.save(os.path.join(self.capture_path, "hand", f"state.npy"), self.hand_state_hist[:self.hand_cnt])
            np.save(os.path.join(self.capture_path, "hand", f"timestamp.npy"), self.hand_timestamp[:self.hand_cnt])
            np.save(os.path.join(self.capture_path, "hand", f"action.npy"), self.hand_action_hist[:self.hand_cnt])
        

    def set_target_action(self, action):
        self.hand_target_action_array[:] = action.copy()
        
    def quit(self):
        self.exit.set()
        self.hand_process.join()

        for key in self.shm.keys():
            self.shm[key].close()
            self.shm[key].unlink()

        print("Exiting...")
