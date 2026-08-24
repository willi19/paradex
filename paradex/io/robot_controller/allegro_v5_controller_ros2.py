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

# Allegro v5 driver joint names: joint_<i>_0, finger blocks 0-3 index,
# 4-7 middle, 8-11 ring, 12-15 thumb.  This is also the exact action order
# produced by ``hand_regargetor.allegro_v5`` and used by CaptureSession.
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
        self.state_topic = topic
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
            self._shared.tactile_time = now

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
            self._shared.state_time = time.monotonic()
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
                max_velocity = (
                    None
                    if shared.max_command_velocity_rad_s is None
                    else shared.max_command_velocity_rad_s.copy()
                )
                last_published = (
                    None
                    if shared.last_published_action is None
                    else shared.last_published_action.copy()
                )
            action = np.clip(action, -MAX_ANGLE, MAX_ANGLE)
            if max_velocity is not None and last_published is not None:
                max_step = max_velocity * period
                action = last_published + np.clip(
                    action - last_published,
                    -max_step,
                    max_step,
                )
            with shared.lock:
                shared.last_published_action = action.copy()
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
        # ``action`` is the latest requested target.  The publish loop can
        # optionally slew-limit its actual output so an overloaded caller
        # cannot skip intermediate hand targets and create a joint jump.
        self.last_published_action = None
        self.max_command_velocity_rad_s = None
        self.joint_value = np.zeros(action_dof, dtype=float)
        self.state_time = None
        self.tactile = None
        # ``perf_counter`` time of the most recently accepted tactile packet.
        # It lets callers reject an old contact reading instead of treating it
        # as a current contact.
        self.tactile_time = None
        self.data = None


class AllegroController:
    def __init__(
        self,
        hand_side=None,
        namespace=None,
        tactile=False,
        command_enabled=True,
        **_,
    ):
        self.tactile_enabled = bool(tactile)
        self.command_enabled = bool(command_enabled)
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
        self._pub_cmd = (
            self._state_node.create_publisher(JointState, cmd_topic, 10)
            if self.command_enabled
            else None
        )
        self.state_topic = self._state_node.state_topic
        self.command_topic = cmd_topic
        self._get_clock = self._state_node.get_clock

        self._state_executor = SingleThreadedExecutor()
        self._state_executor.add_node(self._state_node)

        self._state_thread = Thread(target=self._spin_state, daemon=True)
        self._cmd_thread = (
            Thread(target=self._publish_loop_wrapper, daemon=True)
            if self.command_enabled
            else None
        )
        self._state_thread.start()
        if self._cmd_thread is not None:
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
        if not self.command_enabled:
            raise RuntimeError("Allegro command publishing is disabled for this controller.")
        action = np.asarray(action, dtype=float)
        assert action.shape[0] == action_dof
        with self._shared.lock:
            self._shared.action = action.copy()

    def set_command_slew_rate(self, max_velocity_rad_s=None, *, initial_action=None):
        """Limit physical command-output velocity, or pass ``None`` to disable.

        This is intentionally applied in the publisher rather than in callers:
        callers only hold a latest target, so their intermediate updates can be
        skipped under scheduling load.  A rate limit here guarantees that every
        outgoing joint command remains continuous.
        """

        if max_velocity_rad_s is None:
            with self._shared.lock:
                self._shared.max_command_velocity_rad_s = None
                if initial_action is not None:
                    self._shared.last_published_action = np.asarray(
                        initial_action, dtype=float
                    ).reshape(action_dof).copy()
            return

        velocity = np.asarray(max_velocity_rad_s, dtype=float)
        if velocity.ndim == 0:
            velocity = np.full(action_dof, float(velocity), dtype=float)
        velocity = velocity.reshape(-1)
        if (
            velocity.shape != (action_dof,)
            or not np.all(np.isfinite(velocity))
            or np.any(velocity < 0.0)
        ):
            raise ValueError("max_command_velocity_rad_s must be a finite non-negative 16-vector")
        with self._shared.lock:
            self._shared.max_command_velocity_rad_s = velocity.copy()
            if initial_action is not None:
                initial = np.asarray(initial_action, dtype=float).reshape(-1)
                if initial.shape != (action_dof,) or not np.all(np.isfinite(initial)):
                    raise ValueError("initial_action must be a finite 16-vector")
                self._shared.last_published_action = initial.copy()

    def wait_for_published_action(self, action, *, timeout_seconds=1.0, atol=1e-6):
        """Wait until the publish loop has emitted the specified hand command."""

        action = np.asarray(action, dtype=float).reshape(-1)
        if action.shape != (action_dof,) or not np.all(np.isfinite(action)):
            raise ValueError("action must be a finite 16-vector")
        deadline = time.monotonic() + timeout_seconds
        while time.monotonic() < deadline:
            with self._shared.lock:
                published = (
                    None
                    if self._shared.last_published_action is None
                    else self._shared.last_published_action.copy()
                )
            if published is not None and np.allclose(published, action, atol=atol, rtol=0.0):
                return True
            time.sleep(0.005)
        return False

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
        if self._cmd_thread is not None:
            self._cmd_thread.join(timeout=2.0)

    def get_data(self):
        with self._shared.lock:
            tactile = (
                None if self._shared.tactile is None else self._shared.tactile.copy()
            )
            return {
                'qpos': self._shared.joint_value.copy(),
                'action': self._shared.action.copy(),
                'published_action': (
                    None
                    if self._shared.last_published_action is None
                    else self._shared.last_published_action.copy()
                ),
                'joint_names': list(LOGICAL_JOINT_ORDER),
                'is_connected': self._shared.connection_event.is_set(),
                'state_topic': self.state_topic,
                'command_topic': self.command_topic,
                'tactile': tactile,
                'tactile_time': self._shared.tactile_time,
                'state_monotonic_time': self._shared.state_time,
                'time': time.time(),
            }

    def is_error(self):
        return self.error_event.is_set()
