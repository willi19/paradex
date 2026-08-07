"""ROS 2 (franka_ros2) controller for the Franka FR3 arm.

Drives the arm through the already-running ``fr3_arm_controller``
(``joint_trajectory_controller/JointTrajectoryController``) via its
``follow_joint_trajectory`` action. This needs NO pylibfranka / direct
libfranka: the franka_ros2 driver (controller_manager + the arm controller +
joint_state_broadcaster + robot_state_publisher) must simply be running.

Interface parity with ``XArmController`` as consumed by
``src/calibration/handeye/capture.py``:
  - ``move(action, is_servo=False)`` : blocking move to a 7-dof target qpos
  - ``get_data() -> {"qpos": (7,), "position": (4,4), "time": float}``
  - ``end(set_break=False)``

Notes:
  - qpos is always ordered fr3_joint1..7 (== franka.urdf / Pinocchio order), read
    from /joint_states BY NAME (never by array index).
  - "position" is the fr3_link0 -> fr3_link8 pose from TF (robot-reported). Since
    franka.urdf's world_joint is identity, this equals RobotWrapper FK on
    fr3_link8, so eef.npy matches eef_fk.npy downstream.
  - Requires a ROS 2 environment (source /opt/ros/humble + the franka_ros2 ws).
    Does NOT import pinocchio.
"""

import time
import subprocess
import numpy as np
from threading import Thread, Event, Lock

import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from rclpy.executors import SingleThreadedExecutor

from sensor_msgs.msg import JointState
from trajectory_msgs.msg import JointTrajectoryPoint
from control_msgs.action import FollowJointTrajectory
from builtin_interfaces.msg import Duration as DurationMsg
from scipy.spatial.transform import Rotation
import tf2_ros

ARM_JOINT_NAMES = [f"fr3_joint{i}" for i in range(1, 8)]
JOINT_STATE_TOPIC = "/joint_states"
FOLLOW_ACTION = "/fr3_arm_controller/follow_joint_trajectory"
BASE_FRAME = "fr3_link0"
EEF_FRAME = "fr3_link8"

# For auto-readying the effort control path (see _ensure_arm_ready).
CM = "/controller_manager"
ARM_CONTROLLER = "fr3_arm_controller"
GRAVITY_CONTROLLER = "gravity_compensation_example_controller"
HW_COMPONENT = "FrankaHardwareInterface"


def _run_cli(cmd, timeout=25):
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        return r.returncode, (r.stdout or "") + (r.stderr or "")
    except subprocess.TimeoutExpired:
        return 124, "(timeout)"

# FR3 default / "ready" joint configuration ([0, -pi/4, 0, -3pi/4, 0, pi/2, pi/4]).
# This is the pose the robot boots to and matches the startup /joint_states.
FRANKA_HOME_QPOS = [0.0, -0.7853981633974483, 0.0, -2.356194490192345,
                    0.0, 1.5707963267948966, 0.7853981633974483]

action_dof = 7


class FrankaController(Node):
    def __init__(self, step_size=0.15, step_time=0.5, min_steps=3, max_steps=40,
                 connect_timeout=10.0, auto_ready=True):
        # A move is split into `n_steps` waypoints, n_steps ~ distance/step_size
        # clamped to [min_steps, max_steps]. Each waypoint gets `step_time` s, so
        # the joint speed stays ~constant (~step_size/step_time rad/s) and longer
        # moves simply get more steps.
        if not rclpy.ok():
            rclpy.init()
        super().__init__("franka_controller")

        self.step_size = float(step_size)   # rad per waypoint
        self.step_time = float(step_time)   # s per waypoint
        self.min_steps = int(min_steps)     # floor on number of waypoints
        self.max_steps = int(max_steps)     # ceiling on number of waypoints
        self.lock = Lock()
        self._name2pos = None
        self.exit_event = Event()

        self.create_subscription(JointState, JOINT_STATE_TOPIC, self._on_js, 10)
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)
        self._action = ActionClient(self, FollowJointTrajectory, FOLLOW_ACTION)

        self._executor = SingleThreadedExecutor()
        self._executor.add_node(self)
        self._spin_thread = Thread(target=self._spin, daemon=True)
        self._spin_thread.start()

        # Make the effort path ready (fr3_arm_controller active + effort claimed),
        # auto-recovering from a prior reflex / gravity-comp teaching session so
        # callers don't have to run the ros2 control CLI dance by hand.
        if auto_ready:
            self._ensure_arm_ready()

        if not self._action.wait_for_server(timeout_sec=connect_timeout):
            raise RuntimeError(
                f"franka action server {FOLLOW_ACTION} not available "
                f"(is fr3_arm_controller active?)"
            )
        t0 = time.time()
        while self.get_qpos() is None:
            if time.time() - t0 > connect_timeout:
                raise RuntimeError(f"no {JOINT_STATE_TOPIC} received")
            time.sleep(0.05)

    # --- effort readiness / reflex recovery -----------------------------
    def _effort_ready(self):
        """True if fr3_joint1/effort is available AND claimed (effort control live)."""
        rc, out = _run_cli(["ros2", "control", "list_hardware_interfaces", "-c", CM])
        for line in out.splitlines():
            if "fr3_joint1/effort" in line:
                return "[available]" in line and "[claimed]" in line
        return False

    def _ensure_arm_ready(self, verbose=True):
        """Ensure the effort path is live: re-activate the hardware component and
        (re)activate fr3_arm_controller so effort becomes available+claimed. Safe
        to call anytime; a no-op if already ready. Returns True on success."""
        if self._effort_ready():
            return True
        if verbose:
            print("[franka] effort 준비 안 됨 -> 자동 복구 (set_hardware active + fr3_arm 재활성화)...")
        # 1) bring the hardware component out of any error state
        _run_cli(["ros2", "control", "set_hardware_component_state", HW_COMPONENT, "active"])
        time.sleep(0.4)
        # 2) release effort from whatever holds it (teaching's gravity comp, or a
        #    stuck fr3_arm), then re-activate fr3_arm with a fresh command-mode
        #    switch so franka_hardware sets active_mode_ = Effort. Errors ignored
        #    (controller may not be loaded/active).
        _run_cli(["ros2", "control", "switch_controllers", "-c", CM,
                  "--deactivate", GRAVITY_CONTROLLER])
        time.sleep(0.2)
        _run_cli(["ros2", "control", "switch_controllers", "-c", CM,
                  "--deactivate", ARM_CONTROLLER])
        time.sleep(0.2)
        _run_cli(["ros2", "control", "switch_controllers", "-c", CM,
                  "--activate", ARM_CONTROLLER, "--strict"])
        time.sleep(0.4)
        ok = self._effort_ready()
        if verbose:
            print("[franka] effort ready." if ok else
                  "[franka][warn] effort 자동 복구 실패 — 런치 재시작이 필요할 수 있습니다.")
        return ok

    # --- ROS plumbing ---------------------------------------------------
    def _spin(self):
        while rclpy.ok() and not self.exit_event.is_set():
            self._executor.spin_once(timeout_sec=0.1)

    def _on_js(self, msg):
        with self.lock:
            self._name2pos = dict(zip(msg.name, msg.position))

    def _wait_future(self, future, timeout):
        t0 = time.time()
        while not future.done():
            if self.exit_event.is_set():
                raise RuntimeError("franka controller shutting down")
            if time.time() - t0 > timeout:
                raise TimeoutError("franka action future timed out")
            time.sleep(0.01)
        return future.result()

    # --- state readers --------------------------------------------------
    def get_qpos(self):
        with self.lock:
            n2p = None if self._name2pos is None else dict(self._name2pos)
        if n2p is None or any(n not in n2p for n in ARM_JOINT_NAMES):
            return None
        return np.array([n2p[n] for n in ARM_JOINT_NAMES], dtype=np.float64)

    def get_eef(self):
        try:
            tf = self.tf_buffer.lookup_transform(BASE_FRAME, EEF_FRAME, rclpy.time.Time())
        except tf2_ros.TransformException:
            return None
        t, q = tf.transform.translation, tf.transform.rotation
        mat = np.eye(4)
        mat[:3, :3] = Rotation.from_quat([q.x, q.y, q.z, q.w]).as_matrix()
        mat[:3, 3] = [t.x, t.y, t.z]
        return mat

    def get_data(self):
        qpos, pos = self.get_qpos(), self.get_eef()
        t0 = time.time()
        while (qpos is None or pos is None) and time.time() - t0 < 2.0:
            time.sleep(0.05)
            qpos, pos = self.get_qpos(), self.get_eef()
        if qpos is None or pos is None:
            raise RuntimeError("franka get_data: state unavailable")
        return {"qpos": qpos, "position": pos, "time": time.time()}

    # --- motion ---------------------------------------------------------
    def move(self, action, is_servo=False, speed=None, max_retries=3, tol=0.05):
        action = np.asarray(action, dtype=float).reshape(-1)
        assert action.shape == (action_dof,), \
            f"franka move expects ({action_dof},) qpos, got {action.shape}"
        assert not is_servo, \
            "FrankaController supports only blocking trajectory moves (is_servo=False)"

        last_result = None
        for attempt in range(max_retries):
            # Recover from a prior reflex (effort dead) before (re)trying.
            self._ensure_arm_ready(verbose=(attempt > 0))

            # Split into waypoints: n_steps ~ distance/step_size, clamped to
            # [min_steps, max_steps]. Each step gets step_time s, so speed stays
            # bounded and longer moves get more steps. Re-read q each attempt
            # (after a reflex the arm may be elsewhere).
            q_now = self.get_qpos()
            if q_now is None:
                q_now = action.copy()
            delta = action - q_now
            max_delta = float(np.max(np.abs(delta)))
            n_steps = int(np.ceil(max_delta / self.step_size)) if max_delta > 0 else 1
            n_steps = int(np.clip(n_steps, self.min_steps, self.max_steps))

            goal = FollowJointTrajectory.Goal()
            goal.trajectory.joint_names = list(ARM_JOINT_NAMES)
            for i in range(1, n_steps + 1):
                q_i = q_now + (float(i) / n_steps) * delta
                pt = JointTrajectoryPoint()
                pt.positions = [float(x) for x in q_i]
                t = i * self.step_time
                pt.time_from_start = DurationMsg(sec=int(t), nanosec=int((t - int(t)) * 1e9))
                goal.trajectory.points.append(pt)
            total_time = n_steps * self.step_time

            goal_handle = self._wait_future(self._action.send_goal_async(goal), timeout=15.0)
            if not goal_handle.accepted:
                raise RuntimeError("franka trajectory goal rejected by controller")
            last_result = self._wait_future(
                goal_handle.get_result_async(), timeout=total_time + 15.0
            )
            time.sleep(0.3)  # settle before reading state

            # Success = actually reached the target AND effort still alive (no
            # reflex). JTC reports "success" on time even if a reflex killed the
            # motion, so verify against the real state.
            q_after = self.get_qpos()
            reached = q_after is not None and float(np.max(np.abs(action - q_after))) < tol
            if reached and self._effort_ready():
                return last_result
            print(f"[franka] move 미완/reflex 감지 (attempt {attempt + 1}/{max_retries}) "
                  f"— 복구 후 재시도...")

        print("[franka][warn] move가 목표 도달 실패 (reflex 반복). 더 느리게(step_time↑) "
              "하거나 런치 재시작을 고려하세요.")
        return last_result

    # --- lifecycle ------------------------------------------------------
    def end(self, set_break=False):
        self.exit_event.set()
        if self._spin_thread.is_alive():
            self._spin_thread.join(timeout=2.0)
        try:
            self._executor.shutdown()
        except Exception:
            pass
        self.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
