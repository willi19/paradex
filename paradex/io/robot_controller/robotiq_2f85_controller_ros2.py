import os
import time
from threading import Event, Lock, Thread
from typing import Optional

import numpy as np

import rclpy
from rclpy.executors import SingleThreadedExecutor
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64MultiArray


ACTION_DOF = 1
COMMAND_RATE_HZ = 30.0

# Paradex-side ROS2 contract for Robotiq 2F-85.
#
# The Robotiq ROS2 workspace should expose a topic controller matching this
# contract, typically by spawning a forward_command_controller for the gripper
# position interface.
DEFAULT_NAMESPACE = "robotiq"
COMMAND_SUFFIX = "/robotiq_gripper_controller/commands"
JOINT_STATE_SUFFIX = "/joint_states"
COMMAND_MSG_TYPE = "std_msgs/msg/Float64MultiArray"
STATE_MSG_TYPE = "sensor_msgs/msg/JointState"
JOINT_NAME = "robotiq_85_left_knuckle_joint"


class _StateReceiverNode(Node):
    def __init__(
        self,
        shared,
        namespace: str,
        command_suffix: str,
        joint_state_suffix: str,
        joint_name: str,
    ):
        node_suffix = f"_{namespace}" if namespace else ""
        super().__init__(f"robotiq_2f85_controller{node_suffix}")
        self._shared = shared
        self._joint_name = joint_name

        command_topic = _topic(namespace, command_suffix)
        joint_state_topic = _topic(namespace, joint_state_suffix)

        self.command_pub = self.create_publisher(Float64MultiArray, command_topic, 10)
        self.create_subscription(JointState, joint_state_topic, self._joint_state_cb, 10)

        self.get_logger().info(
            "Robotiq 2F-85 topics: "
            f"command={command_topic} ({COMMAND_MSG_TYPE}), "
            f"state={joint_state_topic} ({STATE_MSG_TYPE})"
        )

    def _joint_state_cb(self, msg: JointState):
        if len(msg.name) != len(msg.position):
            self.get_logger().warn("Ignoring joint state with mismatched name/position lengths.")
            return

        name_to_pos = dict(zip(msg.name, msg.position))
        if self._joint_name not in name_to_pos:
            return

        now = time.time()
        with self._shared.lock:
            self._shared.position = np.asarray([name_to_pos[self._joint_name]], dtype=np.float64)
            self._shared.latest_time = now


class _Shared:
    def __init__(self):
        self.lock = Lock()
        self.save_event = Event()
        self.action = np.zeros(ACTION_DOF, dtype=np.float64)
        self.command = np.zeros(ACTION_DOF, dtype=np.float64)
        self.position = None
        self.latest_time = None
        self.data = None


class Robotiq2F85ControllerROS2:
    """
    ROS2 topic adapter for Robotiq 2F-85 teleoperation.

    Paradex action contract:
        np.array([close_value])
        close_value = 0.0 open, 1.0 closed

    ROS2 topic contract:
        /<namespace>/robotiq_gripper_controller/commands
        std_msgs/msg/Float64MultiArray with one position command
    """

    def __init__(
        self,
        namespace: Optional[str] = DEFAULT_NAMESPACE,
        command_suffix: str = COMMAND_SUFFIX,
        joint_state_suffix: str = JOINT_STATE_SUFFIX,
        joint_name: str = JOINT_NAME,
        command_min: float = 0.0,
        command_max: float = 1.0,
        invert: bool = False,
        rate_hz: float = COMMAND_RATE_HZ,
        **_,
    ):
        if not rclpy.ok():
            rclpy.init()
            self._owns_rclpy = True
        else:
            self._owns_rclpy = False

        self.namespace = _clean_namespace(namespace)
        self.command_suffix = command_suffix
        self.joint_state_suffix = joint_state_suffix
        self.joint_name = joint_name
        self.command_min = float(command_min)
        self.command_max = float(command_max)
        self.invert = bool(invert)
        self.rate_hz = float(rate_hz)

        self.exit_event = Event()
        self.error_event = Event()
        self._shared = _Shared()
        self.capture_path = None

        self._node = _StateReceiverNode(
            self._shared,
            namespace=self.namespace,
            command_suffix=self.command_suffix,
            joint_state_suffix=self.joint_state_suffix,
            joint_name=self.joint_name,
        )

        self._executor = SingleThreadedExecutor()
        self._executor.add_node(self._node)

        self._state_thread = Thread(target=self._spin_state, daemon=True)
        self._cmd_thread = Thread(target=self._publish_loop, daemon=True)
        self._state_thread.start()
        self._cmd_thread.start()

    def _spin_state(self):
        try:
            self._executor.spin()
        except Exception as exc:
            self._node.get_logger().error(f"Robotiq state spin failed: {exc}")
            self.error_event.set()

    def _publish_loop(self):
        period = 1.0 / self.rate_hz if self.rate_hz > 0 else 1.0 / COMMAND_RATE_HZ
        while not self.exit_event.is_set():
            start = time.perf_counter()

            with self._shared.lock:
                action = self._shared.action.copy()
                position = None if self._shared.position is None else self._shared.position.copy()

            command = np.asarray([self._action_to_command(action[0])], dtype=np.float64)
            msg = Float64MultiArray()
            msg.data = command.tolist()
            self._node.command_pub.publish(msg)

            if self._shared.save_event.is_set():
                with self._shared.lock:
                    if self._shared.data is not None:
                        self._shared.command = command.copy()
                        self._shared.data["time"].append(time.time())
                        self._shared.data["action"].append(action.copy())
                        self._shared.data["command"].append(command.copy())
                        if position is not None:
                            self._shared.data["position"].append(position)
            else:
                with self._shared.lock:
                    self._shared.command = command.copy()

            elapsed = time.perf_counter() - start
            time.sleep(max(0.0, period - elapsed))

    def _action_to_command(self, close_value: float) -> float:
        close_value = float(np.clip(close_value, 0.0, 1.0))
        if self.invert:
            close_value = 1.0 - close_value
        return self.command_min + close_value * (self.command_max - self.command_min)

    def move(self, action):
        if action is None:
            return
        action = np.asarray(action, dtype=np.float64).reshape(-1)
        if action.shape != (ACTION_DOF,):
            raise ValueError(f"Robotiq 2F-85 action must have shape (1,), got {action.shape}")
        with self._shared.lock:
            self._shared.action = np.clip(action, 0.0, 1.0).copy()

    def start(self, save_path: str):
        self.capture_path = save_path
        os.makedirs(self.capture_path, exist_ok=True)
        with self._shared.lock:
            self._shared.data = {
                "time": [],
                "action": [],
                "command": [],
                "position": [],
            }
            self._shared.save_event.set()

    def stop(self):
        with self._shared.lock:
            self._shared.save_event.clear()
            data = self._shared.data
            save_path = self.capture_path
            self._shared.data = None
            self.capture_path = None

        if data is None or save_path is None:
            return

        os.makedirs(save_path, exist_ok=True)
        for name, values in data.items():
            np.save(os.path.join(save_path, f"{name}.npy"), np.array(values, dtype=object))

    def end(self):
        self.exit_event.set()
        if self._shared.save_event.is_set():
            self.stop()
        self._cmd_thread.join(timeout=2.0)
        try:
            self._executor.remove_node(self._node)
        except Exception:
            pass
        self._executor.shutdown()
        self._node.destroy_node()
        if self._owns_rclpy:
            rclpy.shutdown()
        self._state_thread.join(timeout=2.0)

    def get_data(self):
        with self._shared.lock:
            qpos = None if self._shared.position is None else self._shared.position.copy()
            action = self._shared.action.copy()
            command = self._shared.command.copy()
            cur_time = self._shared.latest_time if self._shared.latest_time else time.time()
        return {
            "qpos": qpos,
            "position": qpos,
            "action": action,
            "command": command,
            "time": cur_time,
        }

    def is_error(self):
        return self.error_event.is_set()


def _clean_namespace(namespace: Optional[str]) -> str:
    if namespace is None:
        return ""
    return str(namespace).strip("/")


def _topic(namespace: str, suffix: str) -> str:
    suffix = suffix if suffix.startswith("/") else f"/{suffix}"
    if not namespace:
        return suffix
    return f"/{namespace}{suffix}"
