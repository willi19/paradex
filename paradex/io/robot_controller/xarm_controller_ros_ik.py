import os
import time
from threading import Event, Lock, Thread

import numpy as np
from scipy.spatial.transform import Rotation
import transforms3d as t3d

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from xarm_msgs.srv import SetInt16
try:
    from xarm_msgs.msg import RobotMsg
except Exception:
    RobotMsg = None

try:
    from xarm_msgs.srv import MoveJoint
except Exception:
    MoveJoint = None

import pinocchio as pin


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


class PinocchioIKSolver:
    def __init__(
        self,
        urdf_path,
        ee_frame,
        max_iter=80,
        tol=1e-4,
        step_size=0.25,
        damping=1e-5,
    ):
        self.model = pin.buildModelFromUrdf(urdf_path)
        self.data = self.model.createData()

        if self.model.nq != self.model.nv:
            raise NotImplementedError("Only nq == nv robots are supported")
        if self.model.nq < 6:
            raise ValueError(f"Unexpected DoF for xArm model: nq={self.model.nq}")

        self.ee_frame = ee_frame
        self.ee_frame_id = self.model.getFrameId(ee_frame)
        if self.ee_frame_id >= len(self.model.frames):
            raise ValueError(f"Unknown end-effector frame: {ee_frame}")

        self.max_iter = int(max_iter)
        self.tol = float(tol)
        self.step_size = float(step_size)
        self.damping = float(damping)

        self.q_min = np.asarray(self.model.lowerPositionLimit, dtype=np.float64)
        self.q_max = np.asarray(self.model.upperPositionLimit, dtype=np.float64)

    def solve(self, target_homo, q_init):
        q = np.asarray(q_init, dtype=np.float64).copy()
        q = np.clip(q, self.q_min, self.q_max)
        target = pin.SE3(target_homo[:3, :3], target_homo[:3, 3])

        err_norm = np.inf
        for i in range(self.max_iter):
            pin.forwardKinematics(self.model, self.data, q)
            current = pin.updateFramePlacement(self.model, self.data, self.ee_frame_id)

            # 1) Error in LOCAL frame with SE3 log map
            err_se3 = pin.log(current.inverse() * target).vector
            err_norm = float(np.linalg.norm(err_se3))
            if err_norm < self.tol:
                return q, True, i + 1, err_norm

            # 2) Jacobian in LOCAL frame (frame-consistent with error)
            J = pin.computeFrameJacobian(
                self.model, self.data, q, self.ee_frame_id, pin.ReferenceFrame.LOCAL
            )

            # 3) Variable damping
            curr_damping = self.damping if err_norm > 0.01 else self.damping * 0.1

            lhs = J.T @ J + curr_damping * np.eye(self.model.nv, dtype=np.float64)
            rhs = J.T @ err_se3
            dq = np.linalg.solve(lhs, rhs)

            q = pin.integrate(self.model, q, dq * self.step_size)
            q = np.clip(q, self.q_min, self.q_max)

        return q, False, self.max_iter, err_norm

    def fk(self, q):
        q = np.asarray(q, dtype=np.float64)
        pin.forwardKinematics(self.model, self.data, q)
        pose = pin.updateFramePlacement(self.model, self.data, self.ee_frame_id)
        return pose.homogeneous


class XArmControllerROSIK(Node):
    """
    6D pose(AA) -> Pinocchio IK -> set_servo_angle_j.

    - `move(action)` accepts:
      - shape (6,): [x, y, z, ax, ay, az] where xyz is mm and axis-angle is rad.
      - shape (4,4): homogeneous transform.
    - For each command loop:
      1) solve IK with current joint state as seed,
      2) send `set_servo_angle_j`,
      3) keep controller I/O and logging behavior compatible with xarm_controller_ros.py.
    """

    def __init__(
        self,
        ip=None,
        hw_ns="xarm",
        fps=100,
        record_fps=135,
        is_tool_coord=False,
        urdf_path=None,
        ee_frame="link6",
        ik_max_iter=200,
        ik_tol=1e-4,
        ik_step_size=0.5,
        ik_damping=1e-5,
    ):
        del ip

        if not rclpy.ok():
            rclpy.init()
            self._owns_rclpy = True
        else:
            self._owns_rclpy = False

        super().__init__("xarm_controller_ros_ik")

        self.fps = float(fps)
        self.save_fps = float(record_fps)
        self.is_tool_coord = bool(is_tool_coord)
        self.hw_ns = hw_ns.strip("/")
        base = f"/{self.hw_ns}"

        if urdf_path is None:
            urdf_path = os.path.abspath(
                os.path.join(
                    os.path.dirname(__file__),
                    "..",
                    "..",
                    "..",
                    "rsc",
                    "robot",
                    "xarm",
                    "xarm.urdf",
                )
            )

        self.ik_solver = PinocchioIKSolver(
            urdf_path=urdf_path,
            ee_frame=ee_frame,
            max_iter=ik_max_iter,
            tol=ik_tol,
            step_size=ik_step_size,
            damping=ik_damping,
        )
        self.ee_frame = ee_frame

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
        self.latest_pose_time = None
        self.last_cmd_qpos = np.zeros(6, dtype=np.float64)

        self.action = self.last_pose_homo.copy()
        self.data = None
        self.save_path = None

        self.cli_set_mode = self.create_client(SetInt16, f"{base}/set_mode")
        self.cli_set_state = self.create_client(SetInt16, f"{base}/set_state")
        self.cli_set_servo_angle_j = None
        if MoveJoint is not None:
            self.cli_set_servo_angle_j = self.create_client(MoveJoint, f"{base}/set_servo_angle_j")

        self.sub_joint_states = self.create_subscription(
            JointState, f"{base}/joint_states", self._joint_state_cb, 10
        )
        self.sub_robot_states = None
        if RobotMsg is not None:
            self.sub_robot_states = self.create_subscription(
                RobotMsg, f"{base}/robot_states", self._robot_state_cb, 10
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

    def _robot_state_cb(self, msg):
        pose = np.asarray(msg.pose[:6], dtype=np.float64)
        stamp = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        with self.lock:
            self.last_pose_homo = cart2homo(pose)
            self.latest_pose_time = stamp

    def _wait_services(self):
        clients = [
            (self.cli_set_mode, "set_mode"),
            (self.cli_set_state, "set_state"),
        ]
        if self.cli_set_servo_angle_j is not None:
            clients.append((self.cli_set_servo_angle_j, "set_servo_angle_j"))

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

    def _send_servo_angle_j(self, joints):
        if self.cli_set_servo_angle_j is None or MoveJoint is None:
            return None
        req = MoveJoint.Request()
        req.angles = np.asarray(joints, dtype=np.float32).tolist()
        req.speed = 0.0
        req.acc = 0.0
        req.mvtime = 0.0
        req.wait = False
        if hasattr(req, "radius"):
            req.radius = 0.0
        return self._call_sync(self.cli_set_servo_angle_j, req, timeout_sec=0.3)

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

        # Initialize action from robot_states topic pose when available.
        deadline = time.time() + 1.0
        while time.time() < deadline and not self.exit_event.is_set():
            with self.lock:
                has_pose = self.latest_pose_time is not None
                if has_pose:
                    self.action = self.last_pose_homo.copy()
                    return
            time.sleep(0.01)
        with self.lock:
            self.action = self.last_pose_homo.copy()

    def control_loop(self):
        while not self.exit_event.is_set():
            start_time = time.perf_counter()

            with self.lock:
                action = self.action.copy()
                q_seed = None if self.latest_qpos is None else self.latest_qpos.copy()
                q_last = self.last_cmd_qpos.copy()

            if action.shape == (4, 4):
                target_homo = action
            elif action.shape == (6,):
                target_homo = aa2homo(action)
            else:
                self.error_event.set()
                time.sleep(0.01)
                continue

            if q_seed is None:
                q_seed = q_last

            q_cmd, ik_success, ik_iter, ik_error = self.ik_solver.solve(target_homo, q_seed)
            print(q_cmd)
            print(q_cmd)
            print(q_cmd)
            print(q_cmd)
            if not ik_success:
                self.get_logger().warning(
                    f"IK failed (iter={ik_iter}, err={ik_error:.3e}); command skipped"
                )
                self.error_event.set()
                elapsed = time.perf_counter() - start_time
                time.sleep(max(0.0, (1.0 / self.fps) - elapsed))
                continue

            res = self._send_servo_angle_j(q_cmd)
            if res is None:
                self.get_logger().warning("set_servo_angle_j timeout")
                self.error_event.set()
            elif res.ret != 0:
                self.get_logger().warning(f"set_servo_angle_j ret={res.ret}, msg={res.message}")
                self.error_event.set()

            with self.lock:
                self.last_cmd_qpos = q_cmd.copy()

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
        del set_break

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
        del is_servo
        assert action.shape == (4, 4) or action.shape == (6,)
        with self.lock:
            self.action = action.copy()

    def get_data(self):
        with self.lock:
            qpos = self.latest_qpos.copy() if self.latest_qpos is not None else np.full(6, np.nan)
            pos = self.last_pose_homo.copy()
            if self.latest_joint_time is not None:
                current_time = self.latest_joint_time
            elif self.latest_pose_time is not None:
                current_time = self.latest_pose_time
            else:
                current_time = time.time()

        return {
            "qpos": qpos,
            "position": pos,
            "time": current_time,
        }

    def is_error(self):
        return self.error_event.is_set()
