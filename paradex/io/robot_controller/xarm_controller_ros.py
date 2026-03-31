import os
import time
from threading import Event, Lock, Thread

import numpy as np
from scipy.spatial.transform import Rotation
import transforms3d as t3d

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from xarm_msgs.srv import GetFloat32List, MoveCartesian, SetInt16

action_dof = 6


def cart2homo(cart):
    pos = np.eye(4)
    pos[:3, 3] = np.asarray(cart[:3], dtype=np.float64) / 1000.0
    pos[:3, :3] = Rotation.from_euler("xyz", np.asarray(cart[3:6], dtype=np.float64)).as_matrix()
    return pos


def aa2homo(aa):
    pos = np.eye(4)
    pos[:3, 3] = np.asarray(aa[:3], dtype=np.float64) / 1000.0
    axis_angle = np.asarray(aa[3:6], dtype=np.float64)
    angle = np.linalg.norm(axis_angle)
    if angle < 1e-12:
        pos[:3, :3] = np.eye(3)
    else:
        axis = axis_angle / angle
        pos[:3, :3] = t3d.axangles.axangle2mat(axis, angle)
    return pos


def homo2aa(h):
    t = h[:3, 3] * 1000.0
    axis, angle = t3d.axangles.mat2axangle(h[:3, :3], unit_thresh=0.001)
    return np.concatenate([t, axis * angle])


class XArmControllerROS(Node):
    def __init__(self, ip=None, hw_ns="xarm", fps=100, record_fps=135, is_tool_coord=False):
        del ip  # kept for compatibility with existing network_info schema

        if not rclpy.ok():
            rclpy.init()
            self._owns_rclpy = True
        else:
            self._owns_rclpy = False

        super().__init__("xarm_controller_ros")

        self.fps = float(fps)
        self.save_fps = float(record_fps)
        self.is_tool_coord = bool(is_tool_coord)
        self.hw_ns = hw_ns.strip("/")
        base = f"/{self.hw_ns}"

        self.lock = Lock()
        self.exit_event = Event()
        self.save_event = Event()
        self.error_event = Event()
        self.connect_event = Event()
        self.position_control_event = Event()

        self.latest_qpos = None
        self.latest_qvel = None
        self.latest_torque = None
        self.latest_joint_time = None
        self.last_pose_homo = np.eye(4, dtype=np.float64)

        self.action = self.last_pose_homo.copy()
        self.data = None
        self.save_path = None

        self.cli_set_mode = self.create_client(SetInt16, f"{base}/set_mode")
        self.cli_set_state = self.create_client(SetInt16, f"{base}/set_state")
        self.cli_get_position = self.create_client(GetFloat32List, f"{base}/get_position")
        self.cli_set_servo_cart_aa = self.create_client(MoveCartesian, f"{base}/set_servo_cartesian_aa")
        self.sub_joint_states = self.create_subscription(
            JointState, f"{base}/joint_states", self._joint_state_cb, 10
        )

        self.spin_thread = Thread(target=self._spin, daemon=True)
        self.spin_thread.start()

        self.reset()
        self.connect_event.set()

        self.control_thread = Thread(target=self.control_loop, daemon=True)
        self.control_thread.start()
        self.record_thread = Thread(target=self.record_loop, daemon=True)
        self.record_thread.start()

    def _spin(self):
        try:
            rclpy.spin(self)
        except Exception:
            self.error_event.set()

    def _joint_state_cb(self, msg):
        with self.lock:
            self.latest_qpos = np.asarray(msg.position[:6], dtype=np.float64)
            self.latest_qvel = np.asarray(msg.velocity[:6], dtype=np.float64)
            self.latest_torque = np.asarray(msg.effort[:6], dtype=np.float64)
            self.latest_joint_time = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9

    def _wait_services(self):
        clients = [
            (self.cli_set_mode, "set_mode"),
            (self.cli_set_state, "set_state"),
            (self.cli_get_position, "get_position"),
            (self.cli_set_servo_cart_aa, "set_servo_cartesian_aa"),
        ]
        for cli, name in clients:
            while not self.exit_event.is_set() and not cli.wait_for_service(timeout_sec=1.0):
                self.get_logger().info(f"waiting service: {name}")

    def _call_sync(self, client, req, timeout_sec=1.0):
        future = client.call_async(req)
        deadline = time.time() + float(timeout_sec)
        while rclpy.ok() and time.time() < deadline and not future.done():
            time.sleep(0.001)
        if not future.done():
            return None
        return future.result()

    def reset(self):
        self._wait_services()

        req_mode = SetInt16.Request()
        req_mode.data = 1
        res_mode = self._call_sync(self.cli_set_mode, req_mode)
        if res_mode is None or res_mode.ret != 0:
            raise RuntimeError(f"set_mode failed: {None if res_mode is None else res_mode.ret}")

        req_state = SetInt16.Request()
        req_state.data = 0
        res_state = self._call_sync(self.cli_set_state, req_state)
        if res_state is None or res_state.ret != 0:
            raise RuntimeError(f"set_state failed: {None if res_state is None else res_state.ret}")

        req_pos = GetFloat32List.Request()
        res_pos = self._call_sync(self.cli_get_position, req_pos)
        if res_pos is None or res_pos.ret != 0 or len(res_pos.datas) < 6:
            raise RuntimeError("get_position failed")

        with self.lock:
            self.last_pose_homo = cart2homo(np.asarray(res_pos.datas[:6], dtype=np.float64))
            self.action = self.last_pose_homo.copy()

    def _send_servo_aa(self, aa):
        req = MoveCartesian.Request()
        req.pose = np.asarray(aa, dtype=np.float32).tolist()
        req.speed = 0.0
        req.acc = 0.0
        req.mvtime = 0.0
        req.is_tool_coord = self.is_tool_coord
        return self._call_sync(self.cli_set_servo_cart_aa, req, timeout_sec=0.3)

    def control_loop(self):
        while not self.exit_event.is_set():
            start_time = time.perf_counter()

            with self.lock:
                action = self.action.copy()

            if action.shape == (4, 4):
                aa = homo2aa(action)
            elif action.shape == (6,):
                aa = np.asarray(action, dtype=np.float64)
            else:
                self.error_event.set()
                time.sleep(0.01)
                continue

            res = self._send_servo_aa(aa)
            if res is None:
                self.get_logger().warning("set_servo_cartesian_aa timeout")
                self.error_event.set()
            elif res.ret != 0:
                self.get_logger().warning(f"set_servo_cartesian_aa ret={res.ret}, msg={res.message}")
                self.error_event.set()

            elapsed = time.perf_counter() - start_time
            time.sleep(max(0.0, (1.0 / self.fps) - elapsed))

    def record_loop(self):
        while not self.exit_event.is_set():
            start_time = time.perf_counter()

            with self.lock:
                should_save = self.save_event.is_set() and self.data is not None
                qpos = None if self.latest_qpos is None else self.latest_qpos.copy()
                qvel = None if self.latest_qvel is None else self.latest_qvel.copy()
                torque = None if self.latest_torque is None else self.latest_torque.copy()
                sample_time = self.latest_joint_time if self.latest_joint_time is not None else time.time()
                action = self.action.copy()

            if should_save:
                if action.shape == (4, 4):
                    action_homo = action.copy()
                elif action.shape == (6,):
                    action_homo = aa2homo(action)
                else:
                    action_homo = np.eye(4, dtype=np.float64)

                with self.lock:
                    self.data["time"].append(sample_time)
                    self.data["position"].append(
                        qpos.copy() if qpos is not None else np.full(6, np.nan, dtype=np.float64)
                    )
                    self.data["velocity"].append(
                        qvel.copy() if qvel is not None else np.full(6, np.nan, dtype=np.float64)
                    )
                    self.data["torque"].append(
                        torque.copy() if torque is not None else np.full(6, np.nan, dtype=np.float64)
                    )
                    self.data["action"].append(action_homo.copy())
                    self.data["action_qpos"].append(
                        qpos.copy() if qpos is not None else np.full(6, np.nan, dtype=np.float64)
                    )

            elapsed = time.perf_counter() - start_time
            time.sleep(max(0.0, (1.0 / self.save_fps) - elapsed))

    def start(self, save_path):
        self.save_path = save_path
        os.makedirs(self.save_path, exist_ok=True)
        with self.lock:
            self.data = {
                "time": [],
                "position": [],
                "velocity": [],
                "torque": [],
                "action": [],
                "action_qpos": [],
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
            np.save(os.path.join(save_path, f"{name}.npy"), np.asarray(values, dtype=object))

    def end(self, set_break=False):
        del set_break  # no direct equivalent in ROS service interface

        self.exit_event.set()
        self.control_thread.join(timeout=2.0)
        self.record_thread.join(timeout=2.0)

        if self.save_event.is_set():
            self.stop()

        self.destroy_node()
        if self._owns_rclpy and rclpy.ok():
            rclpy.shutdown()
        self.spin_thread.join(timeout=2.0)

    def move(self, action, is_servo=True):
        del is_servo  # kept for compatibility with the legacy controller signature
        assert action.shape == (4, 4) or action.shape == (6,)
        with self.lock:
            self.action = action.copy()

    def get_data(self):
        req = GetFloat32List.Request()
        res = self._call_sync(self.cli_get_position, req, timeout_sec=0.3)
        if res is not None and res.ret == 0 and len(res.datas) >= 6:
            with self.lock:
                self.last_pose_homo = cart2homo(np.asarray(res.datas[:6], dtype=np.float64))

        with self.lock:
            qpos = self.latest_qpos.copy() if self.latest_qpos is not None else np.full(6, np.nan)
            pos = self.last_pose_homo.copy()
            current_time = self.latest_joint_time if self.latest_joint_time is not None else time.time()

        return {
            "qpos": qpos,
            "position": pos,
            "time": current_time,
        }

    def is_error(self):
        return self.error_event.is_set()
