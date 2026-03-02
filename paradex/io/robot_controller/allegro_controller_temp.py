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

# Logical joint order used by this codebase: thumb(0x), index(1x), middle(2x), ring(3x).
LOGICAL_JOINT_ORDER = [
    "ah_joint00", "ah_joint01", "ah_joint02", "ah_joint03",
    "ah_joint10", "ah_joint11", "ah_joint12", "ah_joint13",
    "ah_joint20", "ah_joint21", "ah_joint22", "ah_joint23",
    "ah_joint30", "ah_joint31", "ah_joint32", "ah_joint33",
]

# Position controller command order. Based on the observed behavior, the controller
# is consuming finger blocks as index -> middle -> ring -> thumb.
CONTROLLER_JOINT_ORDER = [
    "ah_joint30", "ah_joint31", "ah_joint32", "ah_joint33",
    "ah_joint00", "ah_joint01", "ah_joint02", "ah_joint03",
    "ah_joint10", "ah_joint11", "ah_joint12", "ah_joint13",
    "ah_joint20", "ah_joint21", "ah_joint22", "ah_joint23",
]

class AllegroController(Node):
    def __init__(self, addr=None, **_):
        # Events first so they always exist even if init is interrupted
        self.save_event = Event()
        self.exit_event = Event()
        self.connection_event = Event()

        super().__init__('allegro_hand_node')
        # addr is accepted for API compatibility with network configuration.
        self.device_addr = addr

        self.current_joint_pose = DEFAULT_VAL
        self._missing_joint_names_warned = False
        self._logical_to_controller_idx = np.array(
            [LOGICAL_JOINT_ORDER.index(name) for name in CONTROLLER_JOINT_ORDER],
            dtype=int,
        )

        self.lock = Lock()

        self.sub_js = self.create_subscription(JointState, JOINT_STATE_TOPIC, self._sub_callback_joint_state, 10)
        self.pub_cmd = self.create_publisher(Float64MultiArray, COMMAND_TOPIC, 10)

        self.action = np.zeros(action_dof, dtype=float)
        self.joint_value = np.zeros(action_dof, dtype=float)

        self.thread = Thread(target=self.control_loop, daemon=True)
        self.thread.start()

    def control_loop(self):
        # wait first joint state
        while rclpy.ok() and not self.connection_event.is_set():
            rclpy.spin_once(self, timeout_sec=0.01)  # 필수!
            time.sleep(0.01)

        rate_hz = 100.0
        dt = 1.0 / rate_hz

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
        if self.save_event.is_set():
            self.stop()

    def move(self, action):
        action = np.asarray(action, dtype=float)
        assert action.shape[0] == action_dof
        with self.lock:
            self.action = action.copy()

    def _sub_callback_joint_state(self, msg: JointState):
        if len(msg.name) != len(msg.position):
            self.get_logger().warn('Ignoring joint state with mismatched name/position lengths.')
            return

        name_to_pos = dict(zip(msg.name, msg.position))
        missing_names = [name for name in LOGICAL_JOINT_ORDER if name not in name_to_pos]
        if missing_names:
            if not self._missing_joint_names_warned:
                self.get_logger().warn(
                    f'Ignoring joint state missing Allegro joints: {missing_names}'
                )
                self._missing_joint_names_warned = True
            return

        pos = np.array([name_to_pos[name] for name in LOGICAL_JOINT_ORDER], dtype=float)

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

        controller_action = action[self._logical_to_controller_idx]

        msg = Float64MultiArray()
        msg.data = controller_action.tolist()
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
                'qpos': self.joint_value.copy(),
                'action': self.action.copy(),
                'time': time.time()
            }
