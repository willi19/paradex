import os
import time
from threading import Event, Lock, Thread

import numpy as np
import rclpy
from rclpy.executors import SingleThreadedExecutor
from rclpy.node import Node
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import JointState


ACTION_DOF = 20
CONTROL_RATE_HZ = 50.0
RECV_RATE_HZ = 80.0
RECV_MIN_INTERVAL = 1.0 / RECV_RATE_HZ
STATE_TOPIC_SUFFIX = "/joint_states"
COMMAND_TOPIC_SUFFIX = "/joint_commands"


def _sensor_qos():
    return QoSProfile(
        reliability=ReliabilityPolicy.BEST_EFFORT,
        history=HistoryPolicy.KEEP_LAST,
        depth=10,
    )


class _Shared:
    def __init__(self):
        self.lock = Lock()
        self.connection_event = Event()
        self.save_event = Event()
        self.action = np.zeros(ACTION_DOF, dtype=np.float32)
        self.joint_value = np.zeros(ACTION_DOF, dtype=np.float32)
        self.data = None


class _WujiStateNode(Node):
    def __init__(self, shared, hand_name, state_topic):
        node_name = f"wuji_state_receiver_{hand_name.strip('/').replace('/', '_')}"
        super().__init__(node_name)
        self._shared = shared
        self._last_state_t = 0.0
        self._bad_shape_warned = False
        self.create_subscription(
            JointState,
            state_topic,
            self._joint_state_cb,
            _sensor_qos(),
        )

    def _joint_state_cb(self, msg):
        now = time.perf_counter()
        if now - self._last_state_t < RECV_MIN_INTERVAL:
            return
        self._last_state_t = now

        if len(msg.position) != ACTION_DOF:
            if not self._bad_shape_warned:
                self.get_logger().warn(
                    f"Ignoring Wuji joint state with {len(msg.position)} positions; expected {ACTION_DOF}."
                )
                self._bad_shape_warned = True
            return
        pos = np.asarray(msg.position, dtype=np.float32)
        with self._shared.lock:
            self._shared.joint_value = pos.copy()
            if not self._shared.connection_event.is_set():
                self._shared.action = pos.copy()
                self._shared.connection_event.set()


def _publish_loop(shared, pub_cmd, exit_event, get_clock):
    period = 1.0 / CONTROL_RATE_HZ
    while not exit_event.is_set():
        t0 = time.perf_counter()
        if shared.connection_event.is_set():
            with shared.lock:
                action = shared.action.copy()
                joint_value = shared.joint_value.copy()
                should_save = shared.save_event.is_set()
                data = shared.data

            msg = JointState()
            msg.header.stamp = get_clock().now().to_msg()
            msg.position = action.tolist()
            pub_cmd.publish(msg)

            if should_save and data is not None:
                data["action"].append(action.copy())
                data["time"].append(time.time())
                data["position"].append(joint_value.copy())

        sleep_for = period - (time.perf_counter() - t0)
        if sleep_for > 0:
            time.sleep(sleep_for)


class WujiControllerROS2:
    """Thin ROS2 controller for the wujihandros2 JointState command interface."""

    def __init__(self, hand_side="right", hand_name=None, namespace=None, **_):
        if not rclpy.ok():
            rclpy.init()
            self._owns_rclpy = True
        else:
            self._owns_rclpy = False

        self.hand_side = str(hand_side).strip().lower()
        if self.hand_side not in ("left", "right"):
            raise ValueError("hand_side must be 'left' or 'right'")

        if hand_name is None:
            hand_name = namespace
        if hand_name is None:
            hand_name = f"{self.hand_side}_hand"
        self.hand_name = str(hand_name).strip("/")
        self.state_topic = f"/{self.hand_name}{STATE_TOPIC_SUFFIX}"
        self.command_topic = f"/{self.hand_name}{COMMAND_TOPIC_SUFFIX}"

        self.exit_event = Event()
        self.error_event = Event()
        self.capture_path = None
        self._shared = _Shared()

        self._state_node = _WujiStateNode(self._shared, self.hand_name, self.state_topic)
        self._pub_cmd = self._state_node.create_publisher(
            JointState,
            self.command_topic,
            _sensor_qos(),
        )
        self._get_clock = self._state_node.get_clock

        self._executor = SingleThreadedExecutor()
        self._executor.add_node(self._state_node)
        self._state_thread = Thread(target=self._spin_state, daemon=True)
        self._cmd_thread = Thread(target=self._publish_loop_wrapper, daemon=True)
        self._state_thread.start()
        self._cmd_thread.start()

    @property
    def connection_event(self):
        return self._shared.connection_event

    @property
    def save_event(self):
        return self._shared.save_event

    def _spin_state(self):
        try:
            self._executor.spin()
        except Exception as exc:
            self._state_node.get_logger().error(f"Wuji state spin failed: {exc}")
            self.error_event.set()

    def _publish_loop_wrapper(self):
        try:
            _publish_loop(self._shared, self._pub_cmd, self.exit_event, self._get_clock)
        except Exception as exc:
            self._state_node.get_logger().error(f"Wuji command publish loop failed: {exc}")
            self.error_event.set()

    def move(self, action):
        action = np.asarray(action, dtype=np.float32)
        if action.shape == (5, 4):
            action = action.reshape(ACTION_DOF)
        if action.shape != (ACTION_DOF,):
            raise ValueError(f"Wuji action must have shape (20,) or (5, 4), got {action.shape}")
        with self._shared.lock:
            self._shared.action = action.copy()

    def start(self, save_path):
        self.capture_path = save_path
        self._shared.data = {"action": [], "time": [], "position": []}
        self._shared.save_event.set()

    def stop(self):
        self._shared.save_event.clear()
        if self.capture_path is not None and self._shared.data is not None:
            os.makedirs(self.capture_path, exist_ok=True)
            for name, value in self._shared.data.items():
                np.save(os.path.join(self.capture_path, f"{name}.npy"), np.asarray(value))
        self.capture_path = None
        self._shared.data = None

    def end(self):
        self.exit_event.set()
        if self._shared.save_event.is_set():
            self.stop()
        try:
            self._executor.remove_node(self._state_node)
        except Exception:
            pass
        self._executor.shutdown()
        self._state_node.destroy_node()
        if self._owns_rclpy:
            rclpy.shutdown()
        self._state_thread.join(timeout=2.0)
        self._cmd_thread.join(timeout=2.0)

    def get_data(self):
        with self._shared.lock:
            return {
                "qpos": self._shared.joint_value.copy(),
                "action": self._shared.action.copy(),
                "time": time.time(),
            }

    def is_error(self):
        return self.error_event.is_set()


WujiController = WujiControllerROS2
