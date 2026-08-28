import os
import time
import glob
import subprocess
import json
from pathlib import Path
import numpy as np
import chime
chime.theme('pokemon')

from paradex.io.camera_system.remote_camera_controller import remote_camera_controller
from paradex.io.camera_system.signal_generator import UTGE900
from paradex.io.camera_system.timestamp_monitor import TimestampMonitor
from paradex.io.robot_controller import get_arm, get_hand
from paradex.io.tactile import HumanTactileRecorder
from paradex.retargetor.allegro_alignment import (
    ALLEGRO_V5_DRIVER_JOINT_NAMES,
    retargeter_action_to_live_controller_qpos,
)
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
_ALLEGRO_UI_ALIGNED_HANDS = frozenset(("allegro_v5",))


class _AllegroTeleopDiagnosticLogger:
    """Keep aligned MANUS/command/feedback samples for offline diagnosis.

    Recording stays in memory while teleoperating so the diagnostic option
    cannot add filesystem latency to the command path.  ``flush`` writes one
    compressed ``.npz`` file when the CaptureSession ends.
    """

    FORMAT_VERSION = 1

    def __init__(self, path):
        self.path = Path(path)
        self._frame_names = None
        self._ergonomic_fields = None
        self._feedback_joint_names = None
        self._samples = []

    @staticmethod
    def _vector(value, size=None):
        if value is None:
            return None
        try:
            result = np.asarray(value, dtype=np.float64).reshape(-1)
        except (TypeError, ValueError):
            return None
        if (size is not None and result.shape != (size,)) or not np.all(np.isfinite(result)):
            return None
        return result.copy()

    def record(
        self,
        *,
        teleop_data,
        state,
        hand_name,
        retargeter_action,
        controller_target,
        feedback,
    ):
        """Record exactly one physical Allegro command and its input/state."""
        frame = teleop_data.get("Right") if isinstance(teleop_data, dict) else None
        if not isinstance(frame, dict) or "wrist" not in frame:
            return

        if self._frame_names is None:
            self._frame_names = tuple(sorted(str(name) for name in frame))
        if any(name not in frame for name in self._frame_names):
            return
        try:
            transforms = np.asarray(
                [frame[name] for name in self._frame_names], dtype=np.float64
            )
        except (TypeError, ValueError):
            return
        if transforms.shape != (len(self._frame_names), 4, 4) or not np.all(np.isfinite(transforms)):
            return

        ergonomics = teleop_data.get("ergonomics", {}).get("Right", {})
        if not isinstance(ergonomics, dict):
            ergonomics = {}
        if self._ergonomic_fields is None:
            self._ergonomic_fields = tuple(sorted(str(name) for name in ergonomics))
        ergonomic_values = np.asarray(
            [ergonomics.get(name, np.nan) for name in self._ergonomic_fields],
            dtype=np.float64,
        )
        feedback = feedback if isinstance(feedback, dict) else {}
        if self._feedback_joint_names is None:
            feedback_names = tuple(str(name) for name in feedback.get("joint_names", ()))
            if len(feedback_names) == 16 and len(set(feedback_names)) == 16:
                self._feedback_joint_names = feedback_names
        tactile = self._vector(feedback.get("tactile"))
        self._samples.append(
            {
                "sample_monotonic_s": time.monotonic(),
                "teleop_wall_time_s": float(teleop_data.get("time", time.time())),
                "state": -1 if state is None else int(state),
                "hand_name": str(hand_name),
                "manus_transforms": transforms.copy(),
                "manus_ergonomics_deg": ergonomic_values.copy(),
                "retargeter_action": self._vector(retargeter_action, 16),
                "controller_target": self._vector(controller_target, 16),
                "feedback_qpos": self._vector(feedback.get("qpos"), 16),
                "feedback_action": self._vector(feedback.get("action"), 16),
                "feedback_connected": bool(feedback.get("is_connected", False)),
                "tactile": tactile,
            }
        )

    @staticmethod
    def _stack(samples, key, shape):
        result = np.full((len(samples), *shape), np.nan, dtype=np.float64)
        for index, sample in enumerate(samples):
            value = sample[key]
            if value is not None and value.shape == shape:
                result[index] = value
        return result

    def flush(self):
        """Write a self-describing compressed archive, including empty logs."""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        samples = self._samples
        frame_names = self._frame_names or ()
        ergonomic_fields = self._ergonomic_fields or ()
        if samples:
            transforms = np.stack([sample["manus_transforms"] for sample in samples])
            ergonomics = np.stack([sample["manus_ergonomics_deg"] for sample in samples])
        else:
            transforms = np.empty((0, len(frame_names), 4, 4), dtype=np.float64)
            ergonomics = np.empty((0, len(ergonomic_fields)), dtype=np.float64)
        tactile_width = max(
            (0 if sample["tactile"] is None else len(sample["tactile"]) for sample in samples),
            default=0,
        )
        tactile = np.full((len(samples), tactile_width), np.nan, dtype=np.float64)
        for index, sample in enumerate(samples):
            value = sample["tactile"]
            if value is not None:
                tactile[index, : len(value)] = value

        metadata = {
            "format": "paradex_allegro_teleop_diagnostic",
            "version": self.FORMAT_VERSION,
            "sample_count": len(samples),
            "notes": (
                "Each row is one hand.move() call. manus_transforms is the Right "
                "frame actually consumed by retargeting (therefore VIVE-reparented "
                "when VIVE is enabled). feedback_qpos is measured ROS feedback "
                "sampled immediately before the command."
            ),
        }
        np.savez_compressed(
            self.path,
            metadata_json=np.asarray(json.dumps(metadata, sort_keys=True)),
            sample_monotonic_s=np.asarray(
                [sample["sample_monotonic_s"] for sample in samples], dtype=np.float64
            ),
            teleop_wall_time_s=np.asarray(
                [sample["teleop_wall_time_s"] for sample in samples], dtype=np.float64
            ),
            state=np.asarray([sample["state"] for sample in samples], dtype=np.int16),
            hand_name=np.asarray([sample["hand_name"] for sample in samples], dtype=str),
            manus_joint_names=np.asarray(frame_names, dtype=str),
            manus_transforms=transforms,
            manus_ergonomic_fields=np.asarray(ergonomic_fields, dtype=str),
            manus_ergonomics_deg=ergonomics,
            feedback_joint_names=np.asarray(
                self._feedback_joint_names or (), dtype=str
            ),
            retargeter_action=self._stack(samples, "retargeter_action", (16,)),
            controller_target=self._stack(samples, "controller_target", (16,)),
            feedback_qpos=self._stack(samples, "feedback_qpos", (16,)),
            feedback_action=self._stack(samples, "feedback_action", (16,)),
            feedback_connected=np.asarray(
                [sample["feedback_connected"] for sample in samples], dtype=bool
            ),
            tactile=tactile,
        )
        print(
            f"Saved Allegro teleop diagnostic: {self.path} "
            f"({len(samples)} command samples)"
        )


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


def _is_allegro_ui_aligned_hand(hand_name):
    return hand_name in _ALLEGRO_UI_ALIGNED_HANDS


def _allegro_v5_feedback_hold_target(feedback):
    """Return the current V5 feedback in the exact live command order.

    This mirrors the alignment UI's pause behavior: one named feedback pose is
    frozen as the controller target, instead of merely ceasing to replace the
    preceding live MANUS target.
    """
    if not isinstance(feedback, dict) or not feedback.get("is_connected", False):
        return None
    try:
        values = np.asarray(feedback["qpos"], dtype=np.float64).reshape(-1)
        names = tuple(str(name) for name in feedback["joint_names"])
    except (KeyError, TypeError, ValueError):
        return None
    if (
        values.shape != (len(names),)
        or len(set(names)) != len(names)
        or not np.all(np.isfinite(values))
    ):
        return None
    by_name = dict(zip(names, values.tolist()))
    if any(name not in by_name for name in ALLEGRO_V5_DRIVER_JOINT_NAMES):
        return None
    return np.asarray(
        [by_name[name] for name in ALLEGRO_V5_DRIVER_JOINT_NAMES],
        dtype=np.float64,
    )


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
        allegro_teleop_diagnostic_path=None,
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
        self._allegro_teleop_diagnostic_logger = (
            _AllegroTeleopDiagnosticLogger(allegro_teleop_diagnostic_path)
            if allegro_teleop_diagnostic_path is not None
            else None
        )
        self.hand_action_provider = hand_action_provider
        self.hand_teleoperation_enabled = True
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
        diagnostic_logger = getattr(self, "_allegro_teleop_diagnostic_logger", None)
        if diagnostic_logger is not None:
            try:
                diagnostic_logger.flush()
            except Exception as exc:
                print(f"Failed to save Allegro teleop diagnostic: {exc}")
            self._allegro_teleop_diagnostic_logger = None
        # ROS hand controllers share the process-wide rclpy context initialized
        # by the arm controller.  Destroy them before ending the arm, because
        # the arm may own and shut down that context.
        if self.hand is not None:
            self.hand.end()
        if self.hand_left is not None:
            self.hand_left.end()
        if self.hand_right is not None:
            self.hand_right.end()
        if self.arm is not None:
            self.arm.end()
        if self.arm_left is not None:
            self.arm_left.end()
        if self.arm_right is not None:
            self.arm_right.end()
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

    def set_hand_teleoperation_enabled(self, enabled):
        """Enable or suppress new hand targets from the teleoperation device."""

        self.hand_teleoperation_enabled = bool(enabled)

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
        hand_name = getattr(self, "hand_name", None)
        hand_name_left = getattr(self, "hand_name_left", None)
        hand_name_right = getattr(self, "hand_name_right", None)
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

        def move_hands_if_due(hand_commands, *, teleop_data=None, state=None):
            """Send a coherent hand target set at the configured UI-rate cap."""
            if not getattr(self, "hand_teleoperation_enabled", True):
                return
            # The UI samples feedback and updates its target on a single 30 Hz
            # command tick.  Do the rate check before acquiring the Allegro
            # feedback lock, so CaptureSession's 10 ms loop cannot contend with
            # the controller's 100 Hz command publisher between actual targets.
            limiter = getattr(self, "_hand_command_limiter", None)
            if limiter is not None and not limiter.is_due(time.monotonic()):
                return
            valid_commands = []
            for hand_controller, action, hand_name in hand_commands:
                if hand_controller is None or action is None:
                    continue
                if _is_allegro_ui_aligned_hand(hand_name):
                    # The UI waits for real feedback before it replaces the
                    # driver's feedback-initialized hold target.  Preserve
                    # that safety boundary in capture as well.
                    try:
                        feedback = hand_controller.get_data()
                    except Exception:
                        continue
                    if _allegro_v5_feedback_hold_target(feedback) is None:
                        continue
                    controller_target = retargeter_action_to_live_controller_qpos(
                        action, hand_name
                    )
                    valid_commands.append(
                        (hand_controller, action, controller_target, hand_name, feedback)
                    )
                    continue
                valid_commands.append((hand_controller, action, action, hand_name, None))
            if not valid_commands:
                return
            for hand_controller, retargeter_action, controller_target, command_hand_name, feedback in valid_commands:
                hand_controller.move(controller_target)
                diagnostic_logger = getattr(self, "_allegro_teleop_diagnostic_logger", None)
                if diagnostic_logger is not None and _is_allegro_ui_aligned_hand(command_hand_name):
                    diagnostic_logger.record(
                        teleop_data=teleop_data,
                        state=state,
                        hand_name=command_hand_name,
                        retargeter_action=retargeter_action,
                        controller_target=controller_target,
                        feedback=feedback,
                    )

        arm_deadman_was_enabled = None
        right_hand_hold_target = None

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
            if session_events is not None:
                if self.save_path is None and session_events["save"].is_set():
                    return "start"
                if self.save_path is not None and session_events["stop"].is_set():
                    chime.info(sync=True)
                    return "stop"

            if loop_callback is not None:
                try:
                    loop_callback(self)
                except Exception as exc:
                    print(f"teleop loop_callback failed: {exc}")
                    loop_callback = None

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

                if state == 0:
                    right_hand_hold_target = None
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
                    move_hands_if_due(
                        ((self.hand, hand_action, hand_name),),
                        teleop_data=data,
                        state=state,
                    )

                    if arm_enabled and wrist_pose is not None:
                        move_arm_if_safe(self.hand_side, self.arm, wrist_pose)
                    arm_deadman_was_enabled = arm_enabled

                if state == 1:   
                    self.retargetor.stop()
                    if _is_allegro_ui_aligned_hand(hand_name):
                        if right_hand_hold_target is None and self.hand is not None:
                            right_hand_hold_target = _allegro_v5_feedback_hold_target(
                                self.hand.get_data()
                            )
                        move_hands_if_due(
                            ((
                                self.hand,
                                right_hand_hold_target,
                                hand_name,
                            ),),
                            teleop_data=data,
                            state=state,
                        )
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

                if state == 0:
                    wrist_pose_left, wrist_pose_right, hand_action_left, hand_action_right = self.retargetor.get_action(data)

                    move_arm_if_safe("Left", self.arm_left, wrist_pose_left)
                    move_arm_if_safe("Right", self.arm_right, wrist_pose_right)
                    move_hands_if_due(
                        (
                            (
                                self.hand_left,
                                hand_action_left,
                                hand_name_left,
                            ),
                            (
                                self.hand_right,
                                hand_action_right,
                                hand_name_right,
                            ),
                        ),
                        teleop_data=data,
                        state=state,
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
