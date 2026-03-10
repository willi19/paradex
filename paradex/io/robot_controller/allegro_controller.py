import time
import os
import numpy as np
from threading import Thread, Event, Lock

import rclpy
rclpy.init()

from rclpy.node import Node

from sensor_msgs.msg import JointState
from std_msgs.msg import Float64MultiArray

JOINT_STATE_TOPIC = '/joint_states'
COMMAND_TOPIC = '/allegro_hand_position_controller/commands'

MAX_ANGLE = 2.1
action_dof = 16

DEFAULT_VAL = None
JS_TO_CMD = [8, 6, 10, 3, 5, 2, 0, 1, 7, 15, 14, 12, 11, 13, 4, 9]

class AllegroController(Node):
    def __init__(self):
        super().__init__('allegro_hand_node')

        self.current_joint_pose = DEFAULT_VAL

        self.save_event = Event()
        self.exit_event = Event()
        self.connection_event = Event()

        self.lock = Lock()

        self.sub_js = self.create_subscription(JointState, JOINT_STATE_TOPIC, self._sub_callback_joint_state, 10)
        self.pub_cmd = self.create_publisher(Float64MultiArray, COMMAND_TOPIC, 10)

        self.action = np.zeros(action_dof, dtype=float)
        self.joint_value = np.zeros(action_dof, dtype=float)

        self._executor = rclpy.executors.SingleThreadedExecutor()
        self._executor.add_node(self)

        self.spin_thread = Thread(target=self._spin_loop, daemon=True)
        self.spin_thread.start()

        self.thread = Thread(target=self.control_loop, daemon=True)
        self.thread.start()

    def _spin_loop(self):
        while rclpy.ok() and not self.exit_event.is_set():
            self._executor.spin_once(timeout_sec=0.1)

    def control_loop(self):
        # wait first joint state
        rate_hz = 100.0
        dt = 1.0 / rate_hz

        while rclpy.ok() and not self.exit_event.is_set():
            if self.connection_event.wait(timeout=0.1):
                break
        
        if not self.connection_event.is_set():
            return
        
        while rclpy.ok() and not self.exit_event.is_set():
            start_time = time.perf_counter()
            with self.lock:
                action = self.action.copy()
                joint_value = self.joint_value.copy()

            self._publish_action(action, absolute=True)

            if self.save_event.is_set():
                self.data["action"].append(action.copy())
                self.data["time"].append(time.time())
                self.data["position"].append(joint_value.copy())

            elapsed = time.perf_counter() - start_time
            time.sleep(max(0.0, dt - elapsed))

    def start(self, save_path):
        self.capture_path = save_path
        self.data = {"action": [], "time": [], "position": []}
        self.save_event.set()

    def stop(self):
        self.save_event.clear()
        os.makedirs(self.capture_path, exist_ok=True)
        for name, value in self.data.items():
            np.save(os.path.join(self.capture_path, f"{name}.npy"), np.array(value))
        self.capture_path = None

    def end(self):
        self.exit_event.set()
        self.thread.join(timeout=1.0)
        self.spin_thread.join(timeout=1.0)
        try:
            self._executor.remove_node(self)
            self._executor.shutdown()
        except Exception:
            pass
        
        if self.save_event.is_set():
            self.stop()

    def move(self, action):
        action_np = np.asarray(action, dtype=float)
        assert action_np.shape[0] == action_dof
        with self.lock:
            self.action = action_np.copy()

    def _sub_callback_joint_state(self, msg: JointState):
        # JointState order must match controller's joint order.
        # Usually it does if the controller config uses same joint list.
        if len(msg.position) >= action_dof:
            pos = np.asarray(msg.position[:action_dof], dtype=float)
        else:
            # If not enough joints, ignore
            return

        with self.lock:
            self.joint_value = pos.copy()
            if self.current_joint_pose is DEFAULT_VAL:
                self.action = pos.copy()
                self.connection_event.set()

        self.current_joint_pose = msg

    def _clip(self, action, value):
        return np.clip(action, -value, value)

    def _publish_action(self, desired_action, absolute=True):
        if self.current_joint_pose is DEFAULT_VAL:
            self.get_logger().warn('No joint data received yet!')
            return

        action = self._clip(np.asarray(desired_action, dtype=float), MAX_ANGLE)

        # ForwardCommandController expects absolute positions in correct joint order
        if not absolute:
            with self.lock:
                current = self.joint_value.copy()
            action = action + current

        msg = Float64MultiArray()
        msg.data = action.tolist()
        self.pub_cmd.publish(msg)

    def get_data(self):
        """
        Get current hand state
        
        Returns:
            dict: Dictionary containing:
                - 'qpos': Current joint positions (16,)
                - 'action': Current target action (16,)
        """
        with self.lock:
            return {
                'qpos': self.joint_value[JS_TO_CMD],
                'action': self.action.copy(),
                'time': time.time()
            }
