import os
import time
import glob
import subprocess
import numpy as np
import chime
chime.theme('pokemon')

from paradex.io.camera_system.remote_camera_controller import remote_camera_controller
from paradex.io.camera_system.signal_generator import UTGE900
from paradex.io.camera_system.timestamp_monitor import TimestampMonitor
from paradex.io.robot_controller import get_arm, get_hand
from paradex.io.tactile import HumanTactileRecorder
from paradex.utils.path import shared_dir
from paradex.retargetor.state import HandStateExtractor
from paradex.retargetor.unimanual import Retargetor
from paradex.calibration.utils import save_current_camparam, save_current_C2R
from paradex.utils.system import network_info


_VIVE_MAX_LINEAR_SPEED_M_S = 0.70
_VIVE_MAX_ANGULAR_SPEED_DEG_S = 240.0
_VIVE_POSITION_MARGIN_M = 0.003
_VIVE_ROTATION_MARGIN_DEG = 1.5
_VIVE_MAX_COMMAND_DT_S = 0.05


class _HandCommandRateLimiter:
    """Sample-and-hold hand targets at a fixed maximum update rate.

    The Allegro alignment UI computes a fresh MANUS target on every browser
    loop, but only calls ``hand.move`` at 30 Hz.  The Allegro driver keeps
    publishing that most recent target between updates.  CaptureSession used
    to overwrite the driver target on every ~10 ms teleop iteration instead,
    which exposed high-frequency MANUS measurement noise to the physical hand.
    """

    def __init__(self, rate_hz):
        if rate_hz is None:
            self.period_s = None
            self.next_send_time = None
            return
        rate_hz = float(rate_hz)
        if not np.isfinite(rate_hz) or rate_hz <= 0.0:
            raise ValueError("hand_command_rate_hz must be a positive finite value or None")
        self.period_s = 1.0 / rate_hz
        self.next_send_time = None

    def is_due(self, now):
        """Reserve one target update when the next sample time has arrived."""
        if self.period_s is None:
            return True
        now = float(now)
        if self.next_send_time is not None and now < self.next_send_time:
            return False
        # Reset from ``now`` rather than catching up with queued stale targets:
        # this is the same latest-target sample-and-hold behavior as the UI.
        self.next_send_time = now + self.period_s
        return True


def _normalize_optional_name(name):
    if name is not None and isinstance(name, str) and name.strip().lower() in ("", "none", "null"):
        return None
    return name


class _PoseCommandLimiter:
    def __init__(self, initial_pose, timestamp):
        self.last_sent_pose = np.asarray(initial_pose, dtype=float).copy()
        self.last_raw_pose = self.last_sent_pose.copy()
        self.last_sample_time = float(timestamp)
        self.rejected_count = 0

    def filter(self, candidate_pose, timestamp):
        candidate_pose = np.asarray(candidate_pose, dtype=float)
        timestamp = float(timestamp)
        dt = min(
            max(timestamp - self.last_sample_time, 0.0),
            _VIVE_MAX_COMMAND_DT_S,
        )
        self.last_sample_time = timestamp

        translation_delta = np.linalg.norm(
            candidate_pose[:3, 3] - self.last_raw_pose[:3, 3]
        )
        relative_rotation = (
            self.last_raw_pose[:3, :3].T @ candidate_pose[:3, :3]
        )
        cosine = np.clip(
            (np.trace(relative_rotation) - 1.0) / 2.0,
            -1.0,
            1.0,
        )
        rotation_delta_deg = np.degrees(np.arccos(cosine))

        translation_limit = (
            _VIVE_MAX_LINEAR_SPEED_M_S * dt + _VIVE_POSITION_MARGIN_M
        )
        rotation_limit_deg = (
            _VIVE_MAX_ANGULAR_SPEED_DEG_S * dt
            + _VIVE_ROTATION_MARGIN_DEG
        )
        accepted = (
            np.isfinite(candidate_pose).all()
            and translation_delta <= translation_limit
            and rotation_delta_deg <= rotation_limit_deg
        )
        if accepted:
            filtered_pose = self.last_sent_pose.copy()
            filtered_pose[:3, 3] += (
                candidate_pose[:3, 3] - self.last_raw_pose[:3, 3]
            )
            filtered_pose[:3, :3] = (
                self.last_sent_pose[:3, :3] @ relative_rotation
            )
            self.last_sent_pose = filtered_pose
            self.rejected_count = 0
        else:
            filtered_pose = None
            self.rejected_count += 1

        self.last_raw_pose = candidate_pose.copy()
        return filtered_pose, translation_delta, rotation_delta_deg


class CaptureSession():
    def __init__(
        self,
        camera=False,
        arm=None,
        hand=None,
        hand_left=None,
        hand_right=None,
        teleop=None,
        tactile=False,
        ip=False,
        hand_side="right",
        events=None,
        realsense=False,
        arm_kwargs=None,
        hand_kwargs=None,
        camera_pc_list=None,
        timestamp=True,
        hand_scale=1.0,
        hand_command_rate_hz=None,
        use_vive=True,
        use_manus=True,
        require_left_control=None,
        hand_action_provider=None,
        arm_command_enabled_provider=None,
        human_tactile=False,
        human_tactile_port="/dev/ttyACM0",
        human_tactile_baud_rate=115200,
        human_tactile_reset_wait=2.0,
        human_tactile_plot_realtime=False,
        human_tactile_plot_refresh_interval=0.02,
        human_tactile_plot_max_samples=200,
    ):
        arm = _normalize_optional_name(arm)
        hand = _normalize_optional_name(hand)
        hand_left = _normalize_optional_name(hand_left)
        hand_right = _normalize_optional_name(hand_right)

        if teleop == "vive" and hand_side not in ("right", "bimanual"):
            raise ValueError(
                "VIVE teleop supports --hand-side right or bimanual."
            )

        if realsense:
            from paradex.io.camera_system.realsense_controller import realsense_controller
            self.realsense = realsense_controller()
        else:
            self.realsense = None
        
        if arm is None and hand is None and hand_left is None and hand_right is None and teleop is not None:
            raise ValueError("Teleop device requires at least one of arm or hand to be specified.")
        
        if camera:
            self.camera = remote_camera_controller(name="dataset_acquisition", pc_list=camera_pc_list)
            self.sync_generator = UTGE900(**network_info["signal_generator"]["param"])
            self.timestamp_monitor = (
                TimestampMonitor(**network_info["timestamp"]["param"])
                if timestamp
                else None
            )
        else:
            self.camera = None
            self.timestamp_monitor = None
            self.sync_generator = None

     

        if hand_side not in ["right", "left", "bimanual"]:
            raise ValueError("Not supported hand side")

        self.events = events
        self.teleop_name = teleop
        self.use_vive = bool(use_vive)
        self.use_manus = bool(use_manus)
        self.hand_command_rate_hz = hand_command_rate_hz
        self._hand_command_limiter = _HandCommandRateLimiter(
            hand_command_rate_hz
        )
        self.hand_action_provider = hand_action_provider
        self.arm_command_enabled_provider = arm_command_enabled_provider
        if (
            self.arm_command_enabled_provider is not None
            and not callable(self.arm_command_enabled_provider)
        ):
            raise ValueError("arm_command_enabled_provider must be callable")
        if self.hand_action_provider is not None and str(hand_side).lower() == "bimanual":
            raise ValueError("hand_action_provider currently supports one robot hand.")
        if require_left_control is None:
            # A left glove is only needed for the legacy gesture pause/control
            # path.  Manus-only hand teleoperation can run from one glove.
            require_left_control = self.use_vive and self.use_manus
        self.require_left_control = bool(require_left_control)

        if hand_side == "right":
            self.hand_side = "Right"
            self.hand_side_opposite = "Left"
        elif hand_side == "left":
            self.hand_side = "Left"
            self.hand_side_opposite = "Right"
        else:
            self.hand_side = "Bimanual"

        self.arm = None
        self.arm_left = None
        self.arm_right = None
        self.arm_name = None
        if arm is not None:
            arm_kwargs = {} if arm_kwargs is None else dict(arm_kwargs)
            self.arm_name = arm
            if self.hand_side == "Bimanual":
                left_kwargs = arm_kwargs.pop("left", arm_kwargs.copy())
                right_kwargs = arm_kwargs.pop("right", arm_kwargs.copy())
                left_kwargs.setdefault("namespace", "left")
                right_kwargs.setdefault("namespace", "right")
                self.arm_right = get_arm(arm, **right_kwargs)
                self.arm_left = get_arm(arm, **left_kwargs)
            else:
                self.arm = get_arm(arm, **arm_kwargs)
        
        self.hand = None
        self.hand_left = None
        self.hand_right = None
        self.hand_name = None
        self.hand_name_left = None
        self.hand_name_right = None
        hand_kwargs = {} if hand_kwargs is None else dict(hand_kwargs)

        if self.hand_side == "Bimanual":
            left_name = hand_left if hand_left is not None else hand
            right_name = hand_right if hand_right is not None else hand
            shared_hand_kwargs = {
                key: value
                for key, value in hand_kwargs.items()
                if key not in ("left", "right")
            }
            left_hand_kwargs = {
                **shared_hand_kwargs,
                **hand_kwargs.get("left", {}),
            }
            right_hand_kwargs = {
                **shared_hand_kwargs,
                **hand_kwargs.get("right", {}),
            }
            if left_name is not None:
                self.hand_name_left = left_name
                self.hand_left = get_hand(
                    hand_name=left_name,
                    tactile=tactile,
                    ip=ip,
                    hand_side="left",
                    **left_hand_kwargs,
                )
            if right_name is not None:
                self.hand_name_right = right_name
                self.hand_right = get_hand(
                    hand_name=right_name,
                    tactile=tactile,
                    ip=ip,
                    hand_side="right",
                    **right_hand_kwargs,
                )
        else:
            if hand is not None:
                self.hand_name = hand
                self.hand = get_hand(
                    hand_name=hand,
                    tactile=tactile,
                    ip=ip,
                    hand_side=self.hand_side.lower(),
                    **hand_kwargs,
                )

        

            
        if teleop is not None:
            if teleop == "xsens":
                # if arm == "openarm":
                from paradex.io.teleop.xsens.receiver import XSensReceiver
                self.teleop_device = XSensReceiver(**network_info["xsens"]["param"])
            elif teleop == "vive":
                from paradex.io.teleop.vive.receiver import ViveManusROSReceiver
                self.teleop_device = ViveManusROSReceiver(
                    hand_side=hand_side,
                    require_left_control=self.require_left_control,
                    use_vive=self.use_vive,
                    **({"use_manus": False} if not self.use_manus else {}),
                )
            else:
                raise ValueError(f"Unsupported teleop device: {teleop}")

            # elif teleop == "occulus":
            #     from paradex.io.teleop.oculus.receiver import OculusReceiver
            #     self.teleop_device = OculusReceiver()
            if arm != 'openarm':
                self.retargetor = Retargetor(
                    arm_name=arm,
                    hand_name=None if self.hand_action_provider is not None else hand,
                    hand_side=self.hand_side,
                    hand_name_left=self.hand_name_left,
                    hand_name_right=self.hand_name_right,
                    hand_scale=hand_scale,
                    teleop_name=teleop,
                )
                self.state_extractor = HandStateExtractor()

        else:
            self.teleop_device = None

        self.human_tactile = None
        if human_tactile:
            self.human_tactile = HumanTactileRecorder(
                port=human_tactile_port,
                baud_rate=human_tactile_baud_rate,
                reset_wait=human_tactile_reset_wait,
                plot_realtime=human_tactile_plot_realtime,
                plot_refresh_interval=human_tactile_plot_refresh_interval,
                plot_max_samples=human_tactile_plot_max_samples,
            )
            self.human_tactile.connect()
            
        self.save_path = None
        self._camera_capture_started = False
        self._timestamp_monitor_started = False
        self._sync_generator_started = False
            
    def start(self, save_path): # Start recording on all sensors
        print("Starting new capture session, saving to:", save_path)
        self.save_path = save_path
        os.makedirs(os.path.join(shared_dir, save_path, "raw"), exist_ok=True)

        if self.human_tactile is not None:
            self.human_tactile.start(os.path.join(shared_dir, save_path, "raw", "human_tactile"))
        
        if self.arm is not None:
            self.arm.start(os.path.join(shared_dir, save_path, "raw", "arm"))
        if self.arm_left is not None:
            self.arm_left.start(os.path.join(shared_dir, save_path, "raw", "arm_left"))
        if self.arm_right is not None:
            self.arm_right.start(os.path.join(shared_dir, save_path, "raw", "arm_right"))
        
        if self.hand is not None:
            self.hand.start(os.path.join(shared_dir, save_path, "raw", "hand"))
        if self.hand_left is not None:
            self.hand_left.start(os.path.join(shared_dir, save_path, "raw", "hand_left"))
        if self.hand_right is not None:
            self.hand_right.start(os.path.join(shared_dir, save_path, "raw", "hand_right"))
            
        if self.teleop_device is not None:
            self.teleop_device.start(os.path.join(shared_dir, save_path, "raw", "teleop"))
            self.state_hist = []
            self.state_time = []

        if self.camera is not None:
            try:
                # Preserve the original Paradex synchronization contract:
                # every camera is configured, armed and acquiring before the
                # first UTG edge is emitted.
                self.camera.start("video", True, os.path.join(save_path, "raw"))
                self._camera_capture_started = True
                if self.timestamp_monitor is not None:
                    self.timestamp_monitor.start(os.path.join(shared_dir, save_path, "raw", "timestamps"))
                    self._timestamp_monitor_started = True
                self.sync_generator.start(fps=30)
                self._sync_generator_started = True
                self.camera.validate(timeout=10.0)
            except Exception:
                try:
                    # Freeze the shared frame boundary before tearing down any
                    # receiver. Direct Aravis streams do not need trigger
                    # pulses for EOS/finalization.
                    if self._sync_generator_started:
                        self.sync_generator.stop()
                        self._sync_generator_started = False
                    try:
                        if self._timestamp_monitor_started and self.timestamp_monitor is not None:
                            self.timestamp_monitor.stop()
                            self._timestamp_monitor_started = False
                    finally:
                        if self._camera_capture_started:
                            self.camera.stop()
                finally:
                    self._camera_capture_started = False
                raise
        
        if self.realsense is not None:
            self.realsense.start(
                save_path=os.path.join(shared_dir, save_path, "depth_cam"),
                fps=30,
                use_depth=True,
            )
        
        
    def stop(self):
        if self.arm is not None:
            self.arm.stop()
        if self.arm_left is not None:
            self.arm_left.stop()
        if self.arm_right is not None:
            self.arm_right.stop()
            
        if self.hand is not None:
            self.hand.stop()
        if self.hand_left is not None:
            self.hand_left.stop()
        if self.hand_right is not None:
            self.hand_right.stop()

        if self.human_tactile is not None:
            self.human_tactile.stop()
                
        if self.teleop_device is not None:
            self.teleop_device.stop()
            os.makedirs(os.path.join(shared_dir, self.save_path, "raw", "state"), exist_ok=True)
            np.save(os.path.join(shared_dir, self.save_path, "raw", "state", "state_hist.npy"), np.array(self.state_hist))
            np.save(os.path.join(shared_dir, self.save_path, "raw", "state", "state_time.npy"), np.array(self.state_time))

        if self.camera is not None:
            try:
                if self._camera_capture_started:
                    print("Stopping camera and saving calibration data...")
                    # UTG is the shared clock. Stop it first so every camera
                    # sees the same final trigger edge. The old aravissrc path
                    # needed pulses during teardown; the direct Aravis/appsrc
                    # backend does not.
                    if self._sync_generator_started:
                        self.sync_generator.stop()
                        self._sync_generator_started = False
                    try:
                        if self._timestamp_monitor_started and self.timestamp_monitor is not None:
                            self.timestamp_monitor.stop()
                            self._timestamp_monitor_started = False
                    finally:
                        self.camera.stop()
                        self._camera_capture_started = False
                    print("Camera stopped.")

                    save_current_camparam(os.path.join(shared_dir, self.save_path))
                    if self.arm is not None or self.arm_left is not None or self.arm_right is not None:
                        if self.arm_name == "xarm":
                            save_current_C2R(os.path.join(shared_dir, self.save_path))
                        elif self.arm_name == "openarm":
                            save_current_C2R(os.path.join(shared_dir, self.save_path), arm="openarm")
            finally:
                if self._sync_generator_started:
                    self.sync_generator.stop()
                    self._sync_generator_started = False

        if self.realsense is not None:
            self.realsense.stop()
        self.save_path = None

    def end(self):
        if self.arm is not None:
            self.arm.end()
        if self.arm_left is not None:
            self.arm_left.end()
        if self.arm_right is not None:
            self.arm_right.end()
        
        if self.hand is not None:
            self.hand.end()
        if self.hand_left is not None:
            self.hand_left.end()
        if self.hand_right is not None:
            self.hand_right.end()
        if self.teleop_device is not None:
            self.teleop_device.end()
        
        if self.camera is not None:
            self.camera.end()
            if self.timestamp_monitor is not None:
                self.timestamp_monitor.end()
            self.sync_generator.end()
        if self.realsense is not None:
            self.realsense.end()
        if self.human_tactile is not None:
            self.human_tactile.close()
    
    def teleop(
        self,
        session_events=None,
        state_policy="gesture_control",
        loop_callback=None,
        bimanual_state_provider=None,
    ):
        if self.teleop_device is None:
            raise ValueError("No teleop device initialized.")
        if state_policy not in ["gesture_control", "keyboard_control"]:
            raise ValueError(f"Unknown state_policy: {state_policy}")
        if bimanual_state_provider is not None and self.hand_side != "Bimanual":
            raise ValueError("bimanual_state_provider requires hand_side='bimanual'.")

        if session_events is None:
            session_events = self.events

        chime.warning(sync=True)
        exit_counter = 0
        stop_counter = 0

        if self.hand_side == "Bimanual":
            if self.arm_left is not None and self.arm_right is not None:
                home_pose = {
                    "Left": self.arm_left.get_data()["position"],
                    "Right": self.arm_right.get_data()["position"],
                }
            else:
                home_pose = {"Left": np.eye(4), "Right": np.eye(4)}
        else:
            home_pose = self.arm.get_data()["position"] if self.arm is not None else np.eye(4)

        self.retargetor.start(home_pose)
        vive_arm_limiters = {}
        if getattr(self, "teleop_name", None) == "vive":
            limiter_start_time = time.monotonic()
            if self.hand_side == "Bimanual":
                vive_arm_limiters = {
                    side: _PoseCommandLimiter(
                        home_pose[side],
                        limiter_start_time,
                    )
                    for side in ("Left", "Right")
                }
            elif self.arm is not None:
                vive_arm_limiters[self.hand_side] = _PoseCommandLimiter(
                    home_pose,
                    limiter_start_time,
                )

        def move_arm_if_safe(side, arm_controller, wrist_pose):
            if arm_controller is None:
                return
            limiter = vive_arm_limiters.get(side)
            if limiter is None:
                arm_controller.move(wrist_pose.copy())
                return

            filtered_pose, translation_delta, rotation_delta_deg = limiter.filter(
                wrist_pose,
                time.monotonic(),
            )
            if filtered_pose is not None:
                arm_controller.move(filtered_pose)
            elif limiter.rejected_count == 1 or limiter.rejected_count % 100 == 0:
                print(
                    f"Rejected {side} VIVE arm command: "
                    f"translation={translation_delta * 1000.0:.1f} mm, "
                    f"rotation={rotation_delta_deg:.1f} deg"
                )

        def move_hands_if_due(hand_commands):
            """Send a coherent hand target set at the configured UI-rate cap."""
            valid_commands = [
                (hand_controller, action)
                for hand_controller, action in hand_commands
                if hand_controller is not None and action is not None
            ]
            if not valid_commands:
                return
            limiter = getattr(self, "_hand_command_limiter", None)
            if limiter is not None and not limiter.is_due(time.monotonic()):
                return
            for hand_controller, action in valid_commands:
                hand_controller.move(action)

        arm_deadman_was_enabled = None

        def arm_commands_enabled():
            provider = getattr(self, "arm_command_enabled_provider", None)
            if provider is None:
                return True
            try:
                return bool(provider())
            except Exception as exc:
                print(f"Arm command enable provider failed; holding arm: {exc}")
                return False

        while True:
            if session_events is not None and session_events["exit"].is_set():
                chime.success(sync=True)
                return "exit"

            data = self.teleop_device.get_data()

            
            if self.hand_side != "Bimanual":
                if data[self.hand_side] is None:
                    print("No data from teleop device...")
                    if session_events is not None:
                        if self.save_path is None and session_events["save"].is_set():
                            return "start"
                        if self.save_path is not None and session_events["stop"].is_set():
                            chime.info(sync=True)
                            return "stop"
                    time.sleep(0.01)
                    continue

                if hasattr(self.teleop_device, "get_state"):
                    state = self.teleop_device.get_state()
                    if state is None:
                        # Keyboard-controlled, Manus-only sessions have no
                        # left-glove pause gesture.  Continue hand-only
                        # teleoperation instead of waiting for that optional
                        # control stream.
                        if state_policy == "keyboard_control":
                            state = 0
                        else:
                            time.sleep(0.01)
                            continue
                else:
                    state = self.state_extractor.get_state(
                        data[self.hand_side_opposite]
                    )

                if self.save_path is not None:
                    self.state_hist.append(state)
                    self.state_time.append(time.time())

                if loop_callback is not None:
                    try:
                        loop_callback(self)
                    except Exception as exc:
                        print(f"teleop loop_callback failed: {exc}")
                        loop_callback = None
                    
                if state == 0:
                    hand_action_provider = getattr(self, "hand_action_provider", None)
                    arm_enabled = arm_commands_enabled()
                    if arm_enabled:
                        # Rebase VIVE to the *actual* current arm pose when a
                        # deadman switch is re-pressed.  Tracker movement while
                        # released can therefore never become an arm jump.
                        if (
                            getattr(self, "arm_command_enabled_provider", None) is not None
                            and arm_deadman_was_enabled is not True
                            and self.arm is not None
                        ):
                            arm_home_pose = self.arm.get_data()["position"]
                            self.retargetor.start(arm_home_pose)
                            vive_arm_limiters[self.hand_side] = _PoseCommandLimiter(
                                arm_home_pose, time.monotonic()
                            )
                        wrist_pose, hand_action = self.retargetor.get_action(data)
                        if hand_action_provider is not None:
                            hand_action = hand_action_provider()
                    elif hand_action_provider is not None:
                        # The external pedal/tactile hand keeps operating
                        # while the xArm deadman switch is released.  Avoid
                        # calculating a VIVE arm pose during this interval so
                        # the re-enable rebase has no stale tracker delta.
                        wrist_pose = None
                        hand_action = hand_action_provider()
                    else:
                        wrist_pose = None
                        hand_action = None
                    move_hands_if_due(((self.hand, hand_action),))

                    if arm_enabled and wrist_pose is not None:
                        move_arm_if_safe(self.hand_side, self.arm, wrist_pose)
                    arm_deadman_was_enabled = arm_enabled

                if state == 1:   
                    self.retargetor.stop()
                    arm_deadman_was_enabled = None
                
                if state == 2:
                    self.retargetor.stop()
                    if state_policy == "gesture_control":
                        stop_counter += 1
                
                elif state_policy == "gesture_control":
                    stop_counter = 0
                    
                if state == 3:
                    if state_policy == "gesture_control":
                        exit_counter += 1
                
                elif state_policy == "gesture_control":
                    exit_counter = 0

                if state_policy == "gesture_control":
                    if exit_counter > 90:
                        chime.success(sync=True)
                        return "exit"
                
                    if stop_counter > 90:
                        chime.info(sync=True)
                        return "stop"

                if session_events is not None:
                    if self.save_path is None and session_events["save"].is_set():
                        return "start"
                    if self.save_path is not None and session_events["stop"].is_set():
                        chime.info(sync=True)
                        return "stop"
            
            else:
                if data["Left"] is None or data["Right"] is None:
                    print("No data from teleop device...")
                    if session_events is not None:
                        if self.save_path is None and session_events["save"].is_set():
                            return "start"
                        if self.save_path is not None and session_events["stop"].is_set():
                            chime.info(sync=True)
                            return "stop"
                    time.sleep(0.01)
                    continue

                state = 0 if bimanual_state_provider is None else bimanual_state_provider()
                if state not in (0, 1):
                    raise ValueError(f"Bimanual state provider returned invalid state: {state}")

                if self.save_path is not None:
                    self.state_hist.append(state)
                    self.state_time.append(time.time())

                if loop_callback is not None:
                    try:
                        loop_callback(self)
                    except Exception as exc:
                        print(f"teleop loop_callback failed: {exc}")
                        loop_callback = None

                if state == 0:
                    wrist_pose_left, wrist_pose_right, hand_action_left, hand_action_right = self.retargetor.get_action(data)

                    move_arm_if_safe("Left", self.arm_left, wrist_pose_left)
                    move_arm_if_safe("Right", self.arm_right, wrist_pose_right)
                    move_hands_if_due(
                        (
                            (self.hand_left, hand_action_left),
                            (self.hand_right, hand_action_right),
                        )
                    )
                else:
                    self.retargetor.stop()

                if session_events is not None:
                    if self.save_path is None and session_events["save"].is_set():
                        return "start"
                    if self.save_path is not None and session_events["stop"].is_set():
                        chime.info(sync=True)
                        return "stop"

            # else:
            #     if data is None:
            #         continue
            #     if self.hand_left is not None:
            #         wrist_pose, hand_action_left, hand_action_right = self.retargetor.get_action(data)
                    
                    
            #         self.hand_left.move(hand_action_left)
            #     if self.events["save"].is_set():
            #         return "saving"
            #     if self.events["stop"].is_set():
            #         return "stop"
            #     if self.events["exit"].is_set():
            #         return "exit"
            time.sleep(0.01)

    def move(self, action_dict):
        if "arm" in action_dict and self.arm is not None:
            self.arm.move(action_dict["arm"])
        if "hand" in action_dict and self.hand is not None:
            self.hand.move(action_dict["hand"])
