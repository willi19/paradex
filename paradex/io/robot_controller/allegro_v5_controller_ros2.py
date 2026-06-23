import time
import os
import numpy as np
from threading import Thread, Event, Lock

import rclpy
from rclpy.node import Node
from rclpy.executors import SingleThreadedExecutor

from sensor_msgs.msg import JointState
from std_msgs.msg import Int32MultiArray

JOINT_STATE_SUFFIX = '/allegroHand_0/joint_states'
COMMAND_SUFFIX = '/allegroHand_0/joint_cmd'
TACTILE_SUFFIX = '/allegroHand_0/tactile_sensors'

MAX_ANGLE = 2.1
action_dof = 16
COMMAND_RATE_HZ = 100.0
RECV_RATE_HZ = 100.0
RECV_MIN_INTERVAL = 1.0 / RECV_RATE_HZ

DEFAULT_VAL = None

# Allegro v5 driver joint names: joint_<i>_0, finger blocks 0-3 thumb, 4-7 index, 8-11 middle, 12-15 ring.
# This matches our logical order (thumb, index, middle, ring), so command/state mapping is identity.
DRIVER_JOINT_ORDER = [f"joint_{i}_0" for i in range(16)]
LOGICAL_JOINT_ORDER = list(DRIVER_JOINT_ORDER)

LOGICAL_TO_DRIVER_IDX = np.array(
    [LOGICAL_JOINT_ORDER.index(name) for name in DRIVER_JOINT_ORDER],
    dtype=int,
)


class _StateReceiverNode(Node):
    def __init__(self, shared, namespace, node_suffix, tactile=False):
        super().__init__(f'allegro_v5_state_receiver{node_suffix}')
        self._shared = shared
        self._missing_warned = False
        self._last_state_t = 0.0
        self._last_tactile_t = 0.0
        topic = f'/{namespace}{JOINT_STATE_SUFFIX}' if namespace else JOINT_STATE_SUFFIX
        self.create_subscription(JointState, topic, self._cb, 10)
        if tactile:
            tactile_topic = (
                f'/{namespace}{TACTILE_SUFFIX}' if namespace else TACTILE_SUFFIX
            )
            self.create_subscription(Int32MultiArray, tactile_topic, self._tactile_cb, 10)

    def _tactile_cb(self, msg: Int32MultiArray):
        now = time.perf_counter()
        if now - self._last_tactile_t < RECV_MIN_INTERVAL:
            return
        self._last_tactile_t = now
        tactile = np.asarray(msg.data, dtype=np.int32)
        with self._shared.lock:
            self._shared.tactile = tactile

    def _cb(self, msg: JointState):
        now = time.perf_counter()
        if now - self._last_state_t < RECV_MIN_INTERVAL:
            return
        self._last_state_t = now
        if len(msg.name) != len(msg.position):
            self.get_logger().warn('Ignoring joint state with mismatched name/position lengths.')
            return

        name_to_pos = dict(zip(msg.name, msg.position))
        missing = [n for n in LOGICAL_JOINT_ORDER if n not in name_to_pos]
        if missing:
            if not self._missing_warned:
                self.get_logger().warn(f'Joint state missing Allegro joints: {missing}')
                self._missing_warned = True
            return

        pos = np.array([name_to_pos[n] for n in LOGICAL_JOINT_ORDER], dtype=float)

        with self._shared.lock:
            self._shared.joint_value = pos.copy()
            if not self._shared.connection_event.is_set():
                self._shared.action = pos.copy()
                self._shared.connection_event.set()


def _publish_loop(shared, pub_cmd, exit_event, get_clock):
    """Plain Python loop publishing commands at COMMAND_RATE_HZ.
    No executor / timer overhead -> tighter rate vs SingleThreadedExecutor.
    """
    period = 1.0 / COMMAND_RATE_HZ
    while not exit_event.is_set():
        t0 = time.perf_counter()
        if shared.connection_event.is_set():
            with shared.lock:
                action = shared.action.copy()
                joint_value = shared.joint_value.copy()
                tactile = None if shared.tactile is None else shared.tactile.copy()
            action = np.clip(action, -MAX_ANGLE, MAX_ANGLE)
            driver_action = action[LOGICAL_TO_DRIVER_IDX]
            msg = JointState()
            msg.header.stamp = get_clock().now().to_msg()
            msg.name = list(DRIVER_JOINT_ORDER)
            msg.position = driver_action.tolist()
            pub_cmd.publish(msg)
            if shared.save_event.is_set():
                shared.data["action"].append(action.copy())
                shared.data["time"].append(time.time())
                shared.data["position"].append(joint_value.copy())
                if "tactile" in shared.data and tactile is not None:
                    shared.data["tactile"].append(tactile)
        sleep_for = period - (time.perf_counter() - t0)
        if sleep_for > 0:
            time.sleep(sleep_for)


class _Shared:
    def __init__(self):
        self.lock = Lock()
        self.connection_event = Event()
        self.save_event = Event()
        self.action = np.zeros(action_dof, dtype=float)
        self.joint_value = np.zeros(action_dof, dtype=float)
        self.tactile = None
        self.data = None


class AllegroController:
    def __init__(self, hand_side=None, namespace=None, tactile=False, **_):
        self.tactile_enabled = bool(tactile)
        if not rclpy.ok():
            rclpy.init()
            self._owns_rclpy = True
        else:
            self._owns_rclpy = False

        self.exit_event = Event()
        self.error_event = Event()

        self._shared = _Shared()
        self.capture_path = None

        # Resolve namespace: explicit `namespace` overrides hand_side; e.g. "right" -> /right/...
        if namespace is None and hand_side is not None:
            namespace = str(hand_side).strip().lower()
        self.namespace = namespace.strip("/") if namespace else None
        node_suffix = f"_{self.namespace}" if self.namespace else ""

        # State callbacks need ROS spin; commands run in a plain Python thread
        # at fixed rate (lower overhead than a second executor + timer).
        self._state_node = _StateReceiverNode(
            self._shared, self.namespace, node_suffix, tactile=self.tactile_enabled,
        )
        cmd_topic = (
            f'/{self.namespace}{COMMAND_SUFFIX}' if self.namespace else COMMAND_SUFFIX
        )
        self._pub_cmd = self._state_node.create_publisher(JointState, cmd_topic, 10)
        self._get_clock = self._state_node.get_clock

        self._state_executor = SingleThreadedExecutor()
        self._state_executor.add_node(self._state_node)

        self._state_thread = Thread(target=self._spin_state, daemon=True)
        self._cmd_thread = Thread(target=self._publish_loop_wrapper, daemon=True)
        self._state_thread.start()
        self._cmd_thread.start()

    def _spin_state(self):
        try:
            self._state_executor.spin()
        except Exception as exc:
            self._state_node.get_logger().error(f"State spin failed: {exc}")
            self.error_event.set()

    def _publish_loop_wrapper(self):
        try:
            _publish_loop(self._shared, self._pub_cmd, self.exit_event, self._get_clock)
        except Exception as exc:
            self._state_node.get_logger().error(f"Command publish loop failed: {exc}")
            self.error_event.set()

    @property
    def connection_event(self):
        return self._shared.connection_event

    @property
    def save_event(self):
        return self._shared.save_event

    def move(self, action):
        action = np.asarray(action, dtype=float)
        assert action.shape[0] == action_dof
        with self._shared.lock:
            self._shared.action = action.copy()

    def start(self, save_path):
        self.capture_path = save_path
        data = {"action": [], "time": [], "position": []}
        if self.tactile_enabled:
            data["tactile"] = []
        self._shared.data = data
        self._shared.save_event.set()

    def stop(self):
        self._shared.save_event.clear()
        os.makedirs(self.capture_path, exist_ok=True)
        for name, value in self._shared.data.items():
            np.save(os.path.join(self.capture_path, f"{name}.npy"), np.array(value))
        self.capture_path = None

    def end(self):
        self.exit_event.set()
        if self._shared.save_event.is_set():
            self.stop()
        try:
            self._state_executor.remove_node(self._state_node)
        except Exception:
            pass
        self._state_executor.shutdown()
        self._state_node.destroy_node()
        if self._owns_rclpy:
            rclpy.shutdown()
        self._state_thread.join(timeout=2.0)
        self._cmd_thread.join(timeout=2.0)

    def get_data(self):
        with self._shared.lock:
            return {
                'qpos': self._shared.joint_value.copy(),
                'action': self._shared.action.copy(),
                'time': time.time(),
            }

    def is_error(self):
        return self.error_event.is_set()
