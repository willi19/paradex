import time
import os
from threading import Thread, Event, Lock
from typing import Optional, List, Dict

import numpy as np

import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64MultiArray
from sensor_msgs.msg import JointState
from control_msgs.msg import DynamicJointState, DynamicInterfaceGroupValues


ACTION_DOF = 6

# Order used across existing Inspire controllers: little, ring, middle, index, thumb2, thumb1
ACTION_ORDER = ["little", "ring", "middle", "index", "thumb2", "thumb1"]

# EtherCAT/ROS2 order used by EthercatBridgeNode:
# thumb1, thumb2, index, middle, ring, little
ETHERCAT_ORDER = ["thumb1", "thumb2", "index", "middle", "ring", "little"]

# Limits in raw units (0.1 deg) for ACTION_ORDER
RAW_LIMITS = [1740, 1740, 1740, 1740, 1350, 1800]

# Per-joint degree offsets for EthercatBridgeNode order
JOINT_DEG_OFFSETS = {
    "thumb1": 180.0,
    "thumb2": 135.0,
    "index": 174.0,
    "middle": 174.0,
    "ring": 174.0,
    "little": 174.0,
}


def _action_to_raw(action: np.ndarray) -> np.ndarray:
    if action.shape != (ACTION_DOF,):
        raise ValueError("action must be shape (6,)")
    out = np.zeros(ACTION_DOF, dtype=np.float64)
    for i, v in enumerate(action):
        if v < 0:
            out[i] = -1
            continue
        v = float(v)
        v = max(0.0, min(1000.0, v))
        out[i] = RAW_LIMITS[i] * v / 1000.0
    return out


def _raw_to_radians(raw: float, joint_name: str) -> float:
    offset = JOINT_DEG_OFFSETS[joint_name]
    return (-0.1 * raw + offset) * np.pi / 180.0


class InspireF1ControllerROS2(Node):
    def __init__(self, hand_side: str = "left"):
        if not rclpy.ok():
            rclpy.init()
            self._owns_rclpy = True
        else:
            self._owns_rclpy = False

        super().__init__("inspire_f1_controller_ros2")

        if hand_side not in ("left", "right"):
            raise ValueError("hand_side must be 'left' or 'right'")
        self.hand_side = hand_side

        self.lock = Lock()
        self.exit_event = Event()
        self.save_event = Event()
        self.error_event = Event()

        self.action = np.zeros(ACTION_DOF, dtype=np.float64) + 1000
        self.latest_qpos = None
        self.latest_force = None
        self.latest_current = None
        self.latest_tactile = None
        self.latest_time = None

        self.save_path = None
        self.data = None

        self.cmd_pub = self.create_publisher(
            Float64MultiArray, "position_controller/commands", 1
        )
        self.joint_sub = self.create_subscription(
            JointState, "joint_states", self._joint_cb, 1
        )
        self.dynamic_sub = self.create_subscription(
            DynamicJointState, "dynamic_joint_states", self._dynamic_cb, 1
        )
        self.tactile_sub = self.create_subscription(
            DynamicInterfaceGroupValues, "tactile_sensor_states", self._tactile_cb, 1
        )

        self.spin_thread = Thread(target=self._spin, daemon=True)
        self.spin_thread.start()

        self.control_thread = Thread(target=self.control_loop, daemon=True)
        self.control_thread.start()

    def _spin(self):
        try:
            rclpy.spin(self)
        except Exception:
            self.error_event.set()

    def _joint_cb(self, msg: JointState):
        with self.lock:
            self.latest_qpos = np.array(msg.position, dtype=np.float64)
            self.latest_time = time.time()

    def _dynamic_cb(self, msg: DynamicJointState):
        with self.lock:
            for iv in msg.interface_values:
                if "actual_force" in iv.interface_names:
                    idx = iv.interface_names.index("actual_force")
                    self.latest_force = np.array([v[idx] for v in iv.values], dtype=np.float64)
                if "actuator_current" in iv.interface_names:
                    idx = iv.interface_names.index("actuator_current")
                    self.latest_current = np.array([v[idx] for v in iv.values], dtype=np.float64)
            self.latest_time = time.time()

    def _tactile_cb(self, msg: DynamicInterfaceGroupValues):
        group_name = f"{self.hand_side}_hand_tactile_sensor"
        tactile_map = {}
        for i, g in enumerate(msg.interface_groups):
            if g == group_name:
                if i >= len(msg.interface_values):
                    break
                iv = msg.interface_values[i]
                for name, val in zip(iv.interface_names, iv.values):
                    tactile_map[name] = val
                break
        if tactile_map:
            with self.lock:
                self.latest_tactile = tactile_map
                self.latest_time = time.time()

    def control_loop(self):
        self.fps = 30
        while not self.exit_event.is_set():
            start_time = time.time()
            with self.lock:
                action = self.action.copy()

            raw = _action_to_raw(action)
            # reorder to EtherCAT order and convert to radians
            raw_map = {name: raw[i] for i, name in enumerate(ACTION_ORDER)}
            cmd = []
            for name in ETHERCAT_ORDER:
                value = raw_map[name]
                if value < 0:
                    # keep current if unknown; fall back to 0 rad
                    cmd.append(0.0)
                else:
                    cmd.append(_raw_to_radians(value, name))
            msg = Float64MultiArray()
            msg.data = cmd
            self.cmd_pub.publish(msg)

            if self.save_event.is_set() and self.data is not None:
                with self.lock:
                    self.data["time"].append(self.latest_time if self.latest_time else time.time())
                    if self.latest_qpos is not None:
                        self.data["position"].append(self.latest_qpos.copy())
                    if self.latest_force is not None:
                        self.data["force"].append(self.latest_force.copy())
                    if self.latest_current is not None:
                        self.data["current"].append(self.latest_current.copy())
                    if self.latest_tactile is not None:
                        self.data["tactile"].append(self.latest_tactile.copy())
                    self.data["action"].append(action.copy())

            elapsed = time.time() - start_time
            time.sleep(max(0.0, (1.0 / self.fps) - elapsed))

    def start(self, save_path: str):
        self.save_path = save_path
        os.makedirs(self.save_path, exist_ok=True)
        with self.lock:
            self.data = {
                "time": [],
                "position": [],
                "force": [],
                "current": [],
                "tactile": [],
                "action": [],
            }
            self.save_event.set()

    def stop(self):
        with self.lock:
            self.save_event.clear()
            data = self.data
            save_path = self.save_path
            self.data = None
            self.save_path = None

        if data is None or save_path is None:
            return
        for name, values in data.items():
            np.save(os.path.join(save_path, f"{name}.npy"), np.array(values, dtype=object))

    def end(self):
        self.exit_event.set()
        self.control_thread.join()
        self.destroy_node()
        if self._owns_rclpy:
            rclpy.shutdown()
        self.spin_thread.join(timeout=2.0)
        if self.save_event.is_set():
            self.stop()

    def move(self, action: np.ndarray):
        assert action.shape == (ACTION_DOF,)
        with self.lock:
            self.action = action.copy()

    def get_data(self):
        with self.lock:
            qpos = None if self.latest_qpos is None else self.latest_qpos.copy()
            cur_time = self.latest_time if self.latest_time is not None else time.time()
            tactile = None if self.latest_tactile is None else dict(self.latest_tactile)
            force = None if self.latest_force is None else self.latest_force.copy()
            current = None if self.latest_current is None else self.latest_current.copy()
        return {
            "qpos": qpos,
            "position": None,
            "force": force,
            "current": current,
            "tactile": tactile,
            "time": cur_time,
        }

    def is_error(self):
        return self.error_event.is_set()
