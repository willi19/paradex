"""Collect a hand-eye calibration trajectory for the Franka (FR3) arm.

Franka mirror of ``src/capture/robot/xarm_teaching.py``.

The robot is driven through the ``franka_ros2`` stack. By default this script
puts the arm into free-drive for you by swapping the active ros2_control
controller:

    fr3_arm_controller  ->  gravity_compensation_example_controller

The gravity-compensation controller commands zero torque (libfranka adds the
gravity term), so the arm becomes compliant and you can hand-guide it while the
robot STAYS in Execution mode -- meaning ``/joint_states`` and TF keep streaming
and this script can read every captured pose. On exit it ALWAYS switches back to
``fr3_arm_controller`` (even on Ctrl+C / error), so the arm is stiff again.

Prerequisites (this script does NOT do these):
  - robot in Execution mode (not Programming), FCI active, control token held by
    FCI -- go press the blue circle button on the robot if a takeover is needed.
  - the franka_ros2 driver up:
      ros2 launch franka_fr3_moveit_config moveit.launch.py \
          robot_ip:=172.16.1.11 use_fake_hardware:=false load_gripper:=false
  - ufw must allow the robot (sudo ufw allow from <robot_ip>) or FCI UDP is
    dropped.

Per captured pose it saves into ``--save_path`` (default
``system/current/hecalib/franka``, read by ``get_handeye_calib_traj("franka")``):
  - ``{idx}_qpos.npy`` : (7,) arm joint angles ordered fr3_joint1..7
  - ``{idx}_aa.npy``   : (4,4) fr3_link0 -> fr3_link8 pose (from TF)

Controls (type the key + Enter, same as xarm_teaching.py):
  c -> capture current pose
  q -> finish (restores fr3_arm_controller and exits)

Use ``--manual-freedrive`` to skip all controller swapping (you enable free-drive
yourself); then this script only reads + captures.
"""

import os
import time
import argparse
import subprocess
import shutil
from glob import glob
from datetime import datetime
from threading import Event, Lock

import numpy as np
from scipy.spatial.transform import Rotation

import rclpy
rclpy.init()
from rclpy.node import Node
from rclpy.executors import SingleThreadedExecutor

from sensor_msgs.msg import JointState
from franka_msgs.msg import FrankaRobotState
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
import tf2_ros

from paradex.utils.keyboard_listener import listen_keyboard

# Arm joints in URDF order. Index /joint_states BY NAME (/franka/joint_states
# scrambles joint2/3; /joint_states may also carry gripper fingers).
ARM_JOINT_NAMES = [f"fr3_joint{i}" for i in range(1, 8)]
JOINT_STATE_TOPIC = "/joint_states"
BASE_FRAME = "fr3_link0"    # robot base (root of the arm chain)
EEF_FRAME = "fr3_link8"     # flange (matches franka.urdf end link)

# Reflex detection: /franka_robot_state_broadcaster/robot_state carries robot_mode.
# Hitting a joint limit while free-driving -> ROBOT_MODE_REFLEX -> arm locks until
# error recovery. We watch this and auto-recover.
ROBOT_STATE_TOPIC = "/franka_robot_state_broadcaster/robot_state"
ROBOT_MODE_REFLEX = 4  # FrankaRobotState.ROBOT_MODE_REFLEX
REFLEX_RECOVER_COOLDOWN = 3.0  # s, avoid spamming recovery
# Break signals (a reflex puts franka_hardware read() in ERROR):
#  - if the whole ros2_control_node dies, /joint_states stops -> stale timing.
#  - if the node lives, the broadcaster keeps re-publishing the FROZEN state, so
#    timing looks fine but the joint values stop changing byte-for-byte. Under
#    gravity comp a healthy arm always micro-moves, so exactly-frozen q == broken.
JS_STALE_SEC = 1.0     # no joint_states message at all -> node crashed
FROZEN_SEC = 2.0       # joint values unchanged (exact) this long -> hardware ERROR

# ros2_control controllers for the free-drive swap.
CM = "/controller_manager"
ARM_CONTROLLER = "fr3_arm_controller"
GRAVITY_CONTROLLER = "gravity_compensation_example_controller"
GRAVITY_TYPE = "franka_example_controllers/GravityCompensationExampleController"
# When a reflex fires, franka_hardware's read() returns ERROR and the whole
# hardware component drops to an error state (effort interfaces unavailable).
# Re-activating THIS component restores them without relaunching.
HW_COMPONENT = "FrankaHardwareInterface"


# --------------------------------------------------------------------------
# ros2_control controller management (free-drive swap) via the ros2 CLI.
# --------------------------------------------------------------------------
def _run(cmd, timeout=30):
    print("  $ " + " ".join(cmd))
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        return 124, "(timeout)"
    out = (r.stdout or "") + (r.stderr or "")
    if out.strip():
        print("    " + out.strip().replace("\n", "\n    "))
    return r.returncode, out


def error_recovery():
    """Clear a franka reflex/error (e.g. communication_constraints_violation)
    so command interfaces become available again before switching."""
    print("[recovery] franka error recovery...")
    _run(["ros2", "action", "send_goal", "/action_server/error_recovery",
          "franka_msgs/action/ErrorRecovery", "{}"], timeout=25)


def recover_reflex():
    """Recover from a joint-limit/other reflex and make gravity comp compliant again.

    After a reflex the robot error is cleared by error_recovery (LED green), but
    franka_hardware leaves the effort command interfaces *unavailable* and the
    (still-active) gravity_compensation controller can't command -> arm stays
    stiff. Only a fresh command-mode switch restarts the FCI effort loop, so we
    deactivate then re-activate the controller.
    """
    print("\n[auto-recover] reflex 감지 — 복구 시퀀스...")
    error_recovery()
    time.sleep(0.3)
    # 1) The reflex dropped the FrankaHardwareInterface component to an error
    #    state, so effort interfaces were unavailable and no controller switch
    #    could restore them. Re-activate the hardware component (no relaunch).
    #    This also resets active_mode_ to None inside franka_hardware.
    _run(["ros2", "control", "set_hardware_component_state", HW_COMPONENT, "active"],
         timeout=20)
    time.sleep(0.3)
    error_recovery()
    time.sleep(0.2)
    # 2) Effort is available again but active_mode_ is None -> write() sends
    #    nothing and the arm stays stiff. Force a fresh command-mode switch by
    #    cycling gravity comp (deactivate now works since effort is available),
    #    which sets active_mode_ = Effort and makes the arm compliant.
    _run(["ros2", "control", "switch_controllers", "-c", CM,
          "--deactivate", GRAVITY_CONTROLLER], timeout=15)
    time.sleep(0.2)
    _run(["ros2", "control", "switch_controllers", "-c", CM,
          "--activate", GRAVITY_CONTROLLER, "--strict"], timeout=15)
    print("[auto-recover] 완료 — 관절을 한계에서 반대 방향으로 살짝 빼세요.")


def archive_existing(save_path):
    """Version-guard: before a new teaching session, move any existing poses to
    a timestamped ``<save_path>_versions/<stamp>/`` dir so they are never
    overwritten. Returns the archive path (or None if there was nothing)."""
    existing = glob(os.path.join(save_path, "*_qpos.npy"))
    if not existing:
        return None
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    archive = os.path.join(os.path.dirname(save_path),
                           os.path.basename(save_path) + "_versions", stamp)
    os.makedirs(archive, exist_ok=True)
    for f in glob(os.path.join(save_path, "*.npy")):
        shutil.move(f, os.path.join(archive, os.path.basename(f)))
    print(f"[version] 기존 포즈 {len(existing)}개를 백업하고 새 세션을 시작합니다:\n"
          f"          {archive}")
    return archive


def preflight():
    """Fail fast with a clear message if the stack isn't ready."""
    rc, out = _run(["ros2", "control", "list_controllers", "-c", CM], timeout=20)
    if rc != 0:
        raise RuntimeError(
            "controller_manager에 접속 불가 — franka 런치가 안 떠있거나 로봇이 "
            "Execution 모드가 아닙니다.\n"
            "  먼저: 로봇을 Execution 모드로 + moveit.launch "
            "(use_fake_hardware:=false load_gripper:=false) 실행 후 다시 시도."
        )
    if ARM_CONTROLLER not in out:
        raise RuntimeError(f"{ARM_CONTROLLER}가 컨트롤러 목록에 없습니다:\n{out}")


def enter_free_drive():
    # Clear any lingering reflex so effort interfaces are available for the swap.
    error_recovery()
    print(f"[free-drive] {GRAVITY_CONTROLLER} 로드 (inactive)...")
    # rc != 0 often just means 'already loaded' from a prior run -> continue.
    _run(["ros2", "run", "controller_manager", "spawner", GRAVITY_CONTROLLER,
          "-c", CM, "-t", GRAVITY_TYPE, "--inactive",
          "--controller-manager-timeout", "10"], timeout=40)

    print("\n⚠️  3초 후 팔이 중력보상(compliant)으로 전환됩니다 — 팔을 잡으세요!")
    for i in (3, 2, 1):
        print(f"    {i}...", flush=True)
        time.sleep(1)

    rc, out = _run(["ros2", "control", "switch_controllers", "-c", CM,
                    "--activate", GRAVITY_CONTROLLER,
                    "--deactivate", ARM_CONTROLLER, "--strict"], timeout=20)
    if rc != 0:
        # atomic switch failed -> fr3_arm_controller is still active, arm safe.
        raise RuntimeError(f"gravity_compensation 전환 실패 (팔은 fr3_arm_controller 유지):\n{out}")
    print("[free-drive] ON — 손으로 움직이며 'c'로 캡처, 'q'로 종료.")


def exit_free_drive():
    print("\n[restore] fr3_arm_controller로 복구 (팔이 다시 뻣뻣해집니다)...")
    error_recovery()  # clear any reflex from teaching so the swap-back is accepted
    rc, out = _run(["ros2", "control", "switch_controllers", "-c", CM,
                    "--activate", ARM_CONTROLLER,
                    "--deactivate", GRAVITY_CONTROLLER, "--strict"], timeout=20)
    if rc != 0:
        print("[restore][경고] 컨트롤러 복구 실패! 팔이 아직 compliant일 수 있습니다. "
              "수동으로 fr3_arm_controller를 활성화하세요:\n" + out)
    _run(["ros2", "control", "unload_controller", GRAVITY_CONTROLLER, "-c", CM], timeout=15)
    print("[restore] 완료.")


# --------------------------------------------------------------------------
# ROS 2 state reader (joints by name + EE pose from TF).
# --------------------------------------------------------------------------
class FrankaTeachingRecorder(Node):
    def __init__(self):
        super().__init__("franka_teaching_recorder")

        self.lock = Lock()
        self._name2pos = None
        self._printed_names = False

        self.create_subscription(
            JointState, JOINT_STATE_TOPIC, self._on_joint_state, 10
        )
        # robot_state is 1 kHz; depth=1 so we only ever process the latest,
        # keeping CPU (and thus control-loop jitter) low.
        self._robot_mode = None
        self._last_js = None
        self._prev_q = None
        self._last_change = None
        self.create_subscription(
            FrankaRobotState, ROBOT_STATE_TOPIC, self._on_robot_state,
            QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE,
                       history=HistoryPolicy.KEEP_LAST),
        )
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

    def _on_robot_state(self, msg: FrankaRobotState):
        self._robot_mode = msg.robot_mode

    def get_robot_mode(self):
        return self._robot_mode

    def is_reflexed(self):
        """True if the arm is locked by a reflex/hardware error.

        Two break modes (robot_mode is unreliable — it freezes, never REFLEX):
          - node crashed: /joint_states stops -> message timing goes stale.
          - node alive, hardware in ERROR: state freezes but the broadcaster
            keeps re-publishing it, so the joint values stop changing exactly.
        """
        now = time.time()
        if self._robot_mode == ROBOT_MODE_REFLEX:
            return True
        if self._last_js is not None and (now - self._last_js) > JS_STALE_SEC:
            return True
        if self._last_change is not None and (now - self._last_change) > FROZEN_SEC:
            return True
        return False

    def _on_joint_state(self, msg: JointState):
        name2pos = dict(zip(msg.name, msg.position))
        with self.lock:
            self._name2pos = name2pos
        self._last_js = time.time()
        # Track real motion: frozen (byte-identical) arm q == hardware ERROR.
        if all(n in name2pos for n in ARM_JOINT_NAMES):
            q = tuple(name2pos[n] for n in ARM_JOINT_NAMES)
            if q != self._prev_q:
                self._prev_q = q
                self._last_change = time.time()
        if not self._printed_names:
            self._printed_names = True
            print(f"[joint_states] receiving. joints: {list(msg.name)}")

    def get_qpos(self):
        with self.lock:
            name2pos = None if self._name2pos is None else dict(self._name2pos)
        if name2pos is None:
            return None
        missing = [n for n in ARM_JOINT_NAMES if n not in name2pos]
        if missing:
            print(f"[warn] missing joints in {JOINT_STATE_TOPIC}: {missing}")
            return None
        return np.array([name2pos[n] for n in ARM_JOINT_NAMES], dtype=np.float64)

    def get_eef(self):
        try:
            tf = self.tf_buffer.lookup_transform(
                BASE_FRAME, EEF_FRAME, rclpy.time.Time()
            )
        except tf2_ros.TransformException as e:
            print(f"[warn] TF {BASE_FRAME}->{EEF_FRAME} not available: {e}")
            return None
        t = tf.transform.translation
        q = tf.transform.rotation
        mat = np.eye(4)
        mat[:3, :3] = Rotation.from_quat([q.x, q.y, q.z, q.w]).as_matrix()
        mat[:3, 3] = [t.x, t.y, t.z]
        return mat


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--save_path", default="system/current/hecalib/franka")
    parser.add_argument("--manual-freedrive", action="store_true",
                        help="Skip the controller swap; you enable free-drive "
                             "yourself and this script only reads + captures.")
    args = parser.parse_args()

    os.makedirs(args.save_path, exist_ok=True)
    # Never overwrite a previous teaching set: archive it first.
    archive_existing(args.save_path)

    stop_event = Event()
    save_event = Event()
    listen_keyboard({"q": stop_event, "c": save_event})

    node = FrankaTeachingRecorder()
    executor = SingleThreadedExecutor()
    executor.add_node(node)

    print(f"Saving to: {os.path.abspath(args.save_path)}")

    free_drive_on = False
    idx = 0
    try:
        if not args.manual_freedrive:
            preflight()
            enter_free_drive()
            free_drive_on = True
        else:
            print("[manual] 컨트롤러는 건드리지 않습니다. free-drive는 직접 켜세요.")
            print("  c = 캡처, q = 종료")

        last_recover = 0.0
        while not stop_event.is_set():
            executor.spin_once(timeout_sec=0.1)

            # Auto-recover if a reflex locked the arm (e.g. hit a joint limit).
            if free_drive_on and node.is_reflexed():
                if time.time() - last_recover > REFLEX_RECOVER_COOLDOWN:
                    recover_reflex()
                    # recover_reflex() blocked for several seconds; drain fresh
                    # joint_states so the staleness signal resets (else it would
                    # immediately re-trigger). Then start the cooldown clock.
                    for _ in range(20):
                        executor.spin_once(timeout_sec=0.05)
                    last_recover = time.time()
                continue

            if not save_event.is_set():
                continue
            save_event.clear()

            qpos = node.get_qpos()
            eef = node.get_eef()
            if qpos is None or eef is None:
                print("[skip] state not ready yet, pose NOT saved. Try again.")
                continue

            np.save(os.path.join(args.save_path, f"{idx}_qpos.npy"), qpos)
            np.save(os.path.join(args.save_path, f"{idx}_aa.npy"), eef)
            print(f"Saved pose {idx}: qpos={np.round(qpos, 4).tolist()}")
            idx += 1

    except KeyboardInterrupt:
        print("Interrupted by user.")
    except RuntimeError as e:
        print(f"[error] {e}")
    finally:
        # ALWAYS restore fr3_arm_controller if we swapped it.
        if free_drive_on:
            exit_free_drive()
        node.destroy_node()
        rclpy.shutdown()
        print(f"Recording session ended. {idx} pose(s) saved.")


if __name__ == "__main__":
    main()
