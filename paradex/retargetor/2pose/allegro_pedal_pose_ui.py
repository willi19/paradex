#!/usr/bin/env python3
"""Quest/pedal interpolation between two edited Allegro target poses.

Quest mode maps right-controller A to grasp and B to open. With
``--with-quest-xarm``, the right grip pose drives xArm relative teleoperation
only while Grip is held. With ``--with-xarm``, VIVE supplies the arm pose and
Quest Grip is the default deadman; ``--input-source=pedal`` retains the legacy
middle-pedal deadman. ``--simulate`` updates only the Allegro URDF mesh unless
an explicit xArm mode is selected.
"""

from __future__ import annotations

import argparse
import atexit
import re
import shutil
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from threading import Event, Lock, Thread
from typing import Any, Mapping

import numpy as np

from paradex.io.robot_controller import get_arm, get_hand
from paradex.io.streamdeck_pedal import LeftRightPedalState
from paradex.dataset_acqusition.capture import CaptureSession, _PoseCommandLimiter
from paradex.retargetor.allegro_alignment import (
    ALLEGRO_RETARGETER_JOINT_NAMES,
    ALLEGRO_URDF_JOINT_NAMES,
    feedback_to_urdf_qpos,
    retargeter_action_to_urdf_qpos,
    urdf_qpos_to_retargeter_action,
)
from paradex.retargetor.hand_alignment_common import read_json, sha256_file, write_json
from paradex.utils.keyboard_listener import listen_keyboard, stop_listening
from paradex.utils.path import shared_dir
from paradex.visualization.visualizer.viser import ViserViewer

if __package__:
    from .quest_teleop import (
        QuestTeleopGate,
        QuestGripTeleopStateAdapter,
        openxr_pose_matrix,
        quest_delta_to_xarm_target,
        quest_grip_deadman_pressed,
    )
else:
    from quest_teleop import (
        QuestTeleopGate,
        QuestGripTeleopStateAdapter,
        openxr_pose_matrix,
        quest_delta_to_xarm_target,
        quest_grip_deadman_pressed,
    )


COMMAND_RATE_HZ = 100.0
UI_RATE_HZ = 20.0
QUEST_POSE_MAX_AGE_S = 0.25
TACTILE_MAX_AGE_S = 0.25
HAND_SLEW_MARGIN = 1.2
HAND_MIN_SLEW_RAD_S = 0.1
# The official V5 driver zeros invalid tactile values outside [0, 5000].
TACTILE_DISPLAY_MAX = 5000
# A single ROS tactile sample must not stop or restart a finger. At the 100 Hz
# control rate these require roughly 30 ms of sustained contact state.
TACTILE_CONTACT_DEBOUNCE_SAMPLES = 3
TACTILE_RELEASE_DEBOUNCE_SAMPLES = 3
# Allegro v5 driver/action order is four contiguous 4-DoF finger blocks.
ALLEGRO_FINGERS = ("index", "middle", "ring", "thumb")
ALLEGRO_FINGER_ACTION_SLICES = {
    finger: slice(index * 4, (index + 1) * 4)
    for index, finger in enumerate(ALLEGRO_FINGERS)
}
DEFAULT_SESSION = Path(shared_dir) / "retargeter_alignment" / "allegro" / "allegro_alignment"
DEFAULT_SOURCE_POSE_A = DEFAULT_SESSION / "000000"
DEFAULT_SOURCE_POSE_B = DEFAULT_SESSION / "000001"
# Keep pedal endpoints in the regular alignment session so they can be edited
# and browsed with the Allegro alignment experiment just like any capture.
DEFAULT_POSE_A = DEFAULT_SESSION / "000008"
DEFAULT_POSE_B = DEFAULT_SESSION / "000009"
DEFAULT_QUEST_OPENXR_LAUNCHER = (
    Path(__file__).resolve().parents[4] / "quest-openxr" / "scripts" / "run-input.sh"
)
# Backward-compatible import name used by the capture entry point.
DEFAULT_QUEST_OPENXR_BIN = DEFAULT_QUEST_OPENXR_LAUNCHER
QUEST_ANALOG_PATTERN = re.compile(
    r"^analog right trigger=(?P<trigger>[0-9]+(?:\.[0-9]+)?) "
    r"squeeze=(?P<grip>[0-9]+(?:\.[0-9]+)?)"
)
QUEST_BUTTON_PATTERN = re.compile(
    r"^button right\.(?P<button>primary|secondary) = (?P<state>pressed|released)$"
)
QUEST_DEBUG_BUTTON_PATTERN = re.compile(
    r"^debug right primary\(active=\d+, value=(?P<primary>[01])\) "
    r"secondary\(active=\d+, value=(?P<secondary>[01])\)$"
)
_POSE_NUMBER = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?"
QUEST_POSE_PATTERN = re.compile(
    rf"^pose right\.grip position=\((?P<px>{_POSE_NUMBER}), "
    rf"(?P<py>{_POSE_NUMBER}), (?P<pz>{_POSE_NUMBER})\) rotation=\("
    rf"(?P<qx>{_POSE_NUMBER}), (?P<qy>{_POSE_NUMBER}), "
    rf"(?P<qz>{_POSE_NUMBER}), (?P<qw>{_POSE_NUMBER})\)$"
)
QUEST_POSE_UNAVAILABLE_PATTERN = re.compile(r"^pose right\.grip unavailable$")


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def interpolate_allegro_pose(
    pose_a: Mapping[str, float], pose_b: Mapping[str, float], parameter: float
) -> dict[str, float]:
    """Linearly blend semantic Allegro URDF joints with a clamped parameter."""
    parameter = float(np.clip(parameter, 0.0, 1.0))
    return {
        name: (1.0 - parameter) * float(pose_a[name]) + parameter * float(pose_b[name])
        for name in ALLEGRO_URDF_JOINT_NAMES
    }


class QuestOpenXRInput:
    """Read the right Touch controller analog values from quest-openxr.

    The reference probe binds trigger to ``/input/trigger/value`` and grip to
    ``/input/squeeze/value``.  Its debug stream emits both values continuously
    without requiring a new OpenXR Python dependency in this robot process.
    """

    def __init__(self, launcher: Path, *, headless: bool = False):
        launcher = launcher.expanduser().resolve()
        if not launcher.is_file():
            raise FileNotFoundError(
                "Quest OpenXR input launcher was not found: "
                f"{launcher}. Build quest-openxr first or pass --quest-openxr-bin."
            )
        self._lock = Lock()
        self._trigger = 0.0
        self._grip = 0.0
        self._primary = False
        self._secondary = False
        self._pose: np.ndarray | None = None
        self._updated_at: float | None = None
        self._grip_updated_at: float | None = None
        self._pose_updated_at: float | None = None
        self._last_error: str | None = None
        command = [str(launcher), "--debug", "--stream-input", "--poses"]
        if headless:
            command.append("--headless")
        self._process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        self._thread = Thread(target=self._read_output, daemon=True)
        self._thread.start()

    def _read_output(self) -> None:
        assert self._process.stdout is not None
        for line in self._process.stdout:
            text = line.strip()
            now = time.monotonic()
            analog = QUEST_ANALOG_PATTERN.match(text)
            button = QUEST_BUTTON_PATTERN.match(text)
            debug_buttons = QUEST_DEBUG_BUTTON_PATTERN.match(text)
            pose = QUEST_POSE_PATTERN.match(text)
            if analog is not None:
                with self._lock:
                    self._trigger = float(analog.group("trigger"))
                    self._grip = float(analog.group("grip"))
                    self._updated_at = now
                    self._grip_updated_at = now
            elif button is not None:
                pressed = button.group("state") == "pressed"
                with self._lock:
                    if button.group("button") == "primary":
                        self._primary = pressed
                    else:
                        self._secondary = pressed
                    self._updated_at = now
            elif debug_buttons is not None:
                with self._lock:
                    self._primary = debug_buttons.group("primary") == "1"
                    self._secondary = debug_buttons.group("secondary") == "1"
                    self._updated_at = now
            elif pose is not None:
                try:
                    pose_matrix = openxr_pose_matrix(
                        [float(pose.group(name)) for name in ("px", "py", "pz")],
                        [float(pose.group(name)) for name in ("qx", "qy", "qz", "qw")],
                    )
                except ValueError:
                    continue
                with self._lock:
                    self._pose = pose_matrix
                    self._pose_updated_at = now
            elif QUEST_POSE_UNAVAILABLE_PATTERN.match(text) is not None:
                with self._lock:
                    self._pose = None
                    self._pose_updated_at = None
        if self._process.poll() not in (None, 0):
            with self._lock:
                self._last_error = f"quest-openxr exited with code {self._process.returncode}"

    def get_values(self) -> tuple[float, float, float | None, str | None]:
        """Return ``(trigger, grip, updated_at, error)`` from the right controller."""
        with self._lock:
            return self._trigger, self._grip, self._updated_at, self._last_error

    def get_snapshot(self) -> dict[str, Any]:
        """Return a coherent right-controller input and pose snapshot."""
        with self._lock:
            return {
                "trigger": self._trigger,
                "grip": self._grip,
                "primary": self._primary,
                "secondary": self._secondary,
                "pose": None if self._pose is None else self._pose.copy(),
                "updated_at": self._updated_at,
                "grip_updated_at": self._grip_updated_at,
                "pose_updated_at": self._pose_updated_at,
                "error": self._last_error,
            }

    def close(self) -> None:
        if self._process.poll() is None:
            self._process.terminate()
            try:
                self._process.wait(timeout=2.0)
            except subprocess.TimeoutExpired:
                self._process.kill()
                self._process.wait(timeout=2.0)
        self._thread.join(timeout=2.0)


def allegro_tactile_finger_levels(
    tactile: Any,
) -> dict[str, float] | None:
    """Return one conservative contact level per Allegro finger.

    The v5 tactile ROS topic is an unlabelled ``Int32MultiArray``.  Its
    channels are therefore split into four contiguous, equally-sized sensor
    blocks in the same index/middle/ring/thumb order as the driver.  A finger
    level is the largest absolute sensor value in its block.  Malformed or
    incomplete packets are intentionally rejected instead of guessing.
    """
    if tactile is None:
        return None
    values = np.asarray(tactile, dtype=float).reshape(-1)
    if values.size < len(ALLEGRO_FINGERS) or not np.all(np.isfinite(values)):
        return None
    blocks = np.array_split(np.abs(values), len(ALLEGRO_FINGERS))
    if any(block.size == 0 for block in blocks):
        return None
    return {
        finger: float(np.max(block))
        for finger, block in zip(ALLEGRO_FINGERS, blocks)
    }


def hold_contacted_allegro_fingers(
    desired_action: Any,
    feedback_action: Any,
    latched_fingers: Mapping[str, bool],
    latched_targets: Mapping[str, Any] | None = None,
) -> np.ndarray:
    """Keep latched finger blocks at their first-contact target values."""
    desired = np.asarray(desired_action, dtype=float).copy()
    feedback = np.asarray(feedback_action, dtype=float)
    if desired.shape != (16,) or feedback.shape != (16,):
        raise ValueError("Allegro tactile hold requires exactly 16 joint values")
    for finger, action_slice in ALLEGRO_FINGER_ACTION_SLICES.items():
        if latched_fingers.get(finger, False):
            held_target = None if latched_targets is None else latched_targets.get(finger)
            if held_target is None:
                desired[action_slice] = feedback[action_slice]
                continue
            held_target = np.asarray(held_target, dtype=float)
            if held_target.shape != (4,) or not np.all(np.isfinite(held_target)):
                raise ValueError(f"Invalid tactile hold target for Allegro {finger}")
            desired[action_slice] = held_target
    return desired


def _load_pose_sample(sample_dir: Path) -> tuple[Path, dict[str, float]]:
    sample_dir = sample_dir.resolve()
    metadata = read_json(sample_dir / "metadata.json")
    if metadata.get("schema_version") != 1 or metadata.get("robot", {}).get("model") != "allegro":
        raise ValueError(f"Unsupported Allegro target sample: {sample_dir}")
    urdf_path = Path(metadata["robot"]["urdf_path"])
    if not urdf_path.is_file() or sha256_file(urdf_path) != metadata["robot"]["urdf_sha256"]:
        raise ValueError(f"Capture-time Allegro URDF is missing or changed: {urdf_path}")
    target = read_json(sample_dir / "target_robot.json")
    values = target.get("urdf_hand_qpos_rad")
    if not isinstance(values, dict) or set(values) != set(ALLEGRO_URDF_JOINT_NAMES):
        raise ValueError(f"Invalid semantic Allegro target pose: {sample_dir}")
    pose = {name: float(values[name]) for name in ALLEGRO_URDF_JOINT_NAMES}
    if not np.all(np.isfinite(tuple(pose.values()))):
        raise ValueError(f"Non-finite Allegro target pose: {sample_dir}")
    return urdf_path.resolve(), pose


def ensure_editable_pose_copy(destination: Path, source: Path) -> Path:
    """Create an editable endpoint without ever modifying its source capture."""
    destination = destination.resolve()
    source = source.resolve()
    if (destination / "metadata.json").is_file() and (destination / "target_robot.json").is_file():
        return destination
    if not (source / "metadata.json").is_file() or not (source / "target_robot.json").is_file():
        raise FileNotFoundError(f"Allegro source pose is incomplete: {source}")
    destination.mkdir(parents=True, exist_ok=True)
    for name in ("metadata.json", "target_robot.json", "alignment_record.json"):
        origin = source / name
        if origin.is_file():
            shutil.copy2(origin, destination / name)
    target = read_json(destination / "target_robot.json")
    target["pedal_pose_source"] = str(source)
    target["pedal_pose_created_at"] = _now()
    write_json(destination / "target_robot.json", target)
    return destination


def _save_pose_sample(sample_dir: Path, pose: Mapping[str, float]) -> None:
    target_path = sample_dir / "target_robot.json"
    target = read_json(target_path)
    target["urdf_hand_qpos_rad"] = {
        name: float(pose[name]) for name in ALLEGRO_URDF_JOINT_NAMES
    }
    target["retargeter_action_rad"] = urdf_qpos_to_retargeter_action(target["urdf_hand_qpos_rad"])
    target["edited"] = True
    target["updated_at"] = _now()
    write_json(target_path, target)

    record_path = sample_dir / "alignment_record.json"
    if record_path.is_file():
        record = read_json(record_path)
        record["target_robot"] = target
        write_json(record_path, record)


class XArmVivePedalTeleop:
    """Run VIVE xArm teleoperation with a Quest Grip or pedal deadman.

    ``CaptureSession`` owns only the physical xArm and uses the same VIVE pose
    limiter as ``capture_robot.py``. The studio continues to own the Allegro
    controller. The selected studio input supplies both Allegro interpolation
    controls and the xArm deadman without enabling MANUS hand teleoperation.
    """

    def __init__(self, studio: "AllegroPedalPoseStudio", servo_api: str):
        self.studio = studio
        self.events = {
            "exit": studio.exit_event,
            "save": Event(),
            "stop": Event(),
        }
        self.session: CaptureSession | None = None
        self.thread: Thread | None = None
        try:
            self.session = CaptureSession(
                camera=False,
                arm="xarm",
                hand=None,
                teleop="vive",
                hand_side="right",
                events=self.events,
                arm_kwargs={"servo_api": servo_api},
                use_vive=True,
                use_manus=False,
                require_left_control=False,
                arm_command_enabled_provider=(
                    self.studio.xarm_deadman_pressed
                    if self.studio.input_source == "pedal"
                    else None
                ),
            )
            if self.studio.input_source == "quest":
                self.session.teleop_device = QuestGripTeleopStateAdapter(
                    self.session.teleop_device,
                    self.studio.quest_grip_teleop_state,
                )
        except Exception:
            if self.session is not None:
                self.session.end()
            raise

    def start(self) -> None:
        if self.thread is not None:
            return
        self.thread = Thread(target=self._run, daemon=True)
        self.thread.start()
        print(
            "[allegro-vive] xArm VIVE teleoperation ready; hold "
            f"{self.studio.xarm_deadman_name} to move (MANUS disabled)."
        )

    def _run(self) -> None:
        try:
            assert self.session is not None
            self.session.teleop(
                session_events=self.events,
                state_policy="keyboard_control",
            )
        except Exception as exc:
            print(f"[allegro-pedal] xArm teleoperation stopped: {exc}")
            self.studio._set_status("xARM TELEOP ERROR: HAND HOLDING")
            self.studio.command_enabled.clear()

    def close(self) -> None:
        self.events["exit"].set()
        if self.thread is not None:
            self.thread.join(timeout=2.0)
            self.thread = None
        if self.session is not None:
            self.session.end()
            self.session = None


class XArmQuestTeleop:
    """Drive xArm from the right Quest grip pose while Grip is held."""

    def __init__(
        self,
        studio: "AllegroPedalPoseStudio",
        servo_api: str,
        *,
        grip_threshold: float,
        translation_scale: float,
    ):
        if studio.quest_input is None:
            raise ValueError("Quest xArm teleoperation requires Quest input")
        if not 0.0 <= grip_threshold <= 1.0:
            raise ValueError("--quest-grip-threshold must be between 0 and 1")
        if translation_scale <= 0.0 or not np.isfinite(translation_scale):
            raise ValueError("--quest-arm-translation-scale must be positive and finite")
        self.studio = studio
        self.quest_input = studio.quest_input
        self.grip_threshold = float(grip_threshold)
        self.translation_scale = float(translation_scale)
        self.arm = get_arm("xarm", servo_api=servo_api)
        self.thread: Thread | None = None
        self._active = False
        self._initial_controller_pose: np.ndarray | None = None
        self._initial_xarm_pose: np.ndarray | None = None
        self._limiter: _PoseCommandLimiter | None = None
        self._gate = QuestTeleopGate()

    def start(self) -> None:
        if self.thread is not None:
            return
        self.thread = Thread(target=self._run, daemon=True)
        self.thread.start()
        print("[allegro-quest] xArm Quest teleoperation ready; hold right Grip to move.")

    def _hold(self, status: str) -> None:
        self._active = False
        self._initial_controller_pose = None
        self._initial_xarm_pose = None
        self._limiter = None
        self.studio.arm_status.value = status

    def _start_from_current_pose(self, controller_pose: np.ndarray) -> bool:
        arm_pose = np.asarray(self.arm.get_data()["position"], dtype=float)
        if arm_pose.shape != (4, 4) or not np.all(np.isfinite(arm_pose)):
            self._hold("QUEST xARM: WAITING FOR VALID ARM POSE")
            return False
        self._initial_controller_pose = controller_pose.copy()
        self._initial_xarm_pose = arm_pose.copy()
        self._limiter = _PoseCommandLimiter(arm_pose, time.monotonic())
        self._active = True
        self.studio.arm_status.value = "QUEST xARM: ACTIVE (GRIP HELD)"
        return True

    def _run(self) -> None:
        try:
            while not self.studio.exit_event.is_set():
                snapshot = self.quest_input.get_snapshot()
                now = time.monotonic()
                pose = snapshot["pose"]
                pose_time = snapshot["pose_updated_at"]
                grip_time = snapshot["grip_updated_at"]
                pose_fresh = (
                    pose is not None
                    and pose_time is not None
                    and now - pose_time <= QUEST_POSE_MAX_AGE_S
                )
                grip_fresh = (
                    grip_time is not None
                    and now - grip_time <= QUEST_POSE_MAX_AGE_S
                )
                grip_held = snapshot["grip"] >= self.grip_threshold
                gate_state = self._gate.update(
                    pose_fresh=pose_fresh,
                    grip_fresh=grip_fresh,
                    grip_held=grip_held,
                )

                if gate_state == "pose_stale":
                    self._hold("QUEST xARM: HOLD (POSE STALE)")
                elif gate_state == "grip_stale":
                    self._hold("QUEST xARM: HOLD (GRIP INPUT STALE)")
                elif gate_state == "release_to_rearm":
                    self._hold("QUEST xARM: RELEASE GRIP TO REARM")
                elif gate_state == "grip_released":
                    if self._active:
                        self._hold("QUEST xARM: HOLD (GRIP RELEASED)")
                else:
                    assert pose is not None
                    if gate_state == "start" and not self._start_from_current_pose(pose):
                        self._gate.retry_activation()
                        self.studio.exit_event.wait(1.0 / COMMAND_RATE_HZ)
                        continue
                    assert self._initial_controller_pose is not None
                    assert self._initial_xarm_pose is not None
                    assert self._limiter is not None
                    candidate = quest_delta_to_xarm_target(
                        self._initial_controller_pose,
                        pose,
                        self._initial_xarm_pose,
                        translation_scale=self.translation_scale,
                    )
                    filtered, translation_delta, rotation_delta_deg = self._limiter.filter(
                        candidate,
                        now,
                    )
                    if filtered is not None:
                        self.arm.move(filtered)
                    elif self._limiter.rejected_count == 1:
                        self.studio.arm_status.value = (
                            "QUEST xARM: REJECTED JUMP "
                            f"({translation_delta * 1000.0:.1f} mm, "
                            f"{rotation_delta_deg:.1f} deg)"
                        )
                self.studio.exit_event.wait(1.0 / COMMAND_RATE_HZ)
        except Exception as exc:
            self.studio.arm_status.value = f"QUEST xARM ERROR: {exc}"
            print(f"[allegro-quest] xArm teleoperation stopped: {exc}")

    def close(self) -> None:
        if self.thread is not None:
            self.thread.join(timeout=2.0)
            self.thread = None
        self.arm.end()


class AllegroPedalPoseStudio:
    def __init__(
        self,
        *,
        pose_a_sample: Path,
        pose_b_sample: Path,
        simulate: bool,
        observe_only: bool,
        input_source: str,
        input_rate: float,
        quest_grip_threshold: float = 0.5,
        quest_openxr_bin: Path = DEFAULT_QUEST_OPENXR_BIN,
        quest_headless: bool = False,
        tactile_contact_stop: bool = True,
        tactile_threshold: float = 200.0,
        ring_tactile_threshold: float = 150.0,
        external_hand: bool = False,
    ):
        self.pose_a_sample = pose_a_sample.resolve()
        self.pose_b_sample = pose_b_sample.resolve()
        self.urdf_path, self.pose_a = _load_pose_sample(self.pose_a_sample)
        urdf_b, self.pose_b = _load_pose_sample(self.pose_b_sample)
        if urdf_b != self.urdf_path:
            raise ValueError("Pose A and B use different Allegro URDFs")
        if input_source not in {"quest", "pedal", "slider"}:
            raise ValueError("--input-source must be quest, pedal, or slider")
        if input_rate <= 0.0 or not np.isfinite(input_rate):
            raise ValueError("--input-rate must be a positive finite value")
        if not 0.0 <= quest_grip_threshold <= 1.0:
            raise ValueError("--quest-grip-threshold must be between 0 and 1")
        if tactile_threshold < 0.0 or not np.isfinite(tactile_threshold):
            raise ValueError("--tactile-threshold must be a non-negative finite value")
        if ring_tactile_threshold < 0.0 or not np.isfinite(ring_tactile_threshold):
            raise ValueError(
                "--ring-tactile-threshold must be a non-negative finite value"
            )
        if simulate and observe_only:
            raise ValueError("--simulate and --observe-only cannot be combined")

        self.simulate = simulate
        self.observe_only = observe_only
        self.external_hand = bool(external_hand)
        self.input_source = input_source
        self.input_rate = float(input_rate)
        self.quest_grip_threshold = float(quest_grip_threshold)
        self.tactile_contact_stop = bool(tactile_contact_stop)
        self.tactile_threshold = float(tactile_threshold)
        self.ring_tactile_threshold = float(ring_tactile_threshold)
        self.parameter = 0.0
        # A tactile latch freezes the first-contact target.  Reopening only
        # releases it after the global parameter returns to that contact point.
        self.moving_toward_b = False
        self.contact_latched = {finger: False for finger in ALLEGRO_FINGERS}
        self.contact_latch_parameter = {finger: None for finger in ALLEGRO_FINGERS}
        self.contact_hold_action = {finger: None for finger in ALLEGRO_FINGERS}
        self.contact_latch_level = {finger: None for finger in ALLEGRO_FINGERS}
        self.contact_above_threshold_count = {finger: 0 for finger in ALLEGRO_FINGERS}
        self.contact_below_threshold_count = {finger: 0 for finger in ALLEGRO_FINGERS}
        self.editing_endpoint = "A"
        self.command_enabled = Event()
        if not observe_only:
            self.command_enabled.set()
        self.exit_event = Event()
        self.pedal = None
        self.quest_input = None
        self.hand = None
        self._hand_command_controller = None
        self._hand_slew_configured = False
        self._syncing = False
        self.last_time = None
        self._next_ui_update_time = 0.0

        self.viewer = ViserViewer(scene_title="Allegro two-pose interpolation", show_player=False)
        self.viewer.add_robot("allegro", str(self.urdf_path), include_arm_meshes=True)
        self.viser_robot = self.viewer.robot_dict["allegro"]
        self.robot = self.viser_robot.urdf
        self.joint_names = tuple(self.robot.get_joint_names())
        self.joint_limits = self.robot.get_joint_limits()
        missing = set(ALLEGRO_URDF_JOINT_NAMES) - set(self.joint_names)
        if missing:
            raise ValueError(f"Allegro URDF is missing joints: {sorted(missing)}")
        # Pedal endpoints are edited directly against this capture-time URDF.
        # Clipping to its per-joint hardware limits preserves the exact edited
        # direction.  The retargeter's seven-pose calibration envelope is not
        # a physical limit and would otherwise alter valid endpoint values.
        self.action_lower = np.asarray(
            [self.joint_limits[name][0] for name in ALLEGRO_RETARGETER_JOINT_NAMES],
            dtype=float,
        )
        self.action_upper = np.asarray(
            [self.joint_limits[name][1] for name in ALLEGRO_RETARGETER_JOINT_NAMES],
            dtype=float,
        )

        self._build_gui()
        self._update_interpolated_pose()
        if input_source == "pedal":
            self.pedal = LeftRightPedalState()
            atexit.register(self.pedal.close)
        elif input_source == "quest":
            self.quest_input = QuestOpenXRInput(
                quest_openxr_bin,
                headless=quest_headless,
            )
            atexit.register(self.quest_input.close)
        if not simulate and not self.external_hand:
            self.hand = get_hand(
                "allegro_v5",
                hand_side="right",
                # Keep the raw UI observable even when contact stopping is
                # temporarily disabled.
                tactile=True,
                command_enabled=not observe_only,
            )
            self.attach_hand_controller(self.hand)
        if self.external_hand:
            self._set_status("WAITING: ALLEGRO FEEDBACK")
        else:
            self._set_status("SIMULATION: URDF ONLY" if simulate else "WAITING: ALLEGRO FEEDBACK")

    def _build_gui(self) -> None:
        gui = self.viewer.server.gui
        with gui.add_folder("Pose interpolation"):
            self.command_status = gui.add_text("Command state", initial_value="STARTING", disabled=True)
            self.input_status = gui.add_text("Control input", initial_value="STARTING", disabled=True)
            self.arm_status = gui.add_text("xArm teleoperation", initial_value="DISABLED", disabled=True)
            self.actual_status = gui.add_text("Actual Allegro qpos", initial_value="WAITING", disabled=True)
            self.tactile_raw_status = gui.add_text("Tactile raw values", initial_value="WAITING", disabled=True)
            self.tactile_value_sliders = {}
            for finger in ALLEGRO_FINGERS:
                display_name = "ring/pinky" if finger == "ring" else finger
                label = f"Tactile {display_name} raw"
                self.tactile_value_sliders[finger] = gui.add_slider(
                    label,
                    min=0,
                    max=TACTILE_DISPLAY_MAX,
                    step=1,
                    initial_value=0,
                    disabled=True,
                )
            self.tactile_status = gui.add_text("Tactile contact", initial_value="DISABLED", disabled=True)
            self.parameter_text = gui.add_text("Interpolation parameter", initial_value="0.000", disabled=True)
            self.parameter_slider = gui.add_slider("Parameter: A  ←  →  B", min=0.0, max=1.0, step=0.001, initial_value=0.0)
            self.tactile_threshold_slider = gui.add_slider(
                "Tactile stop threshold", min=0.0,
                max=max(10000.0, self.tactile_threshold), step=1.0,
                initial_value=self.tactile_threshold,
                disabled=not self.tactile_contact_stop,
            )
            self.ring_tactile_threshold_slider = gui.add_slider(
                "Ring tactile stop threshold", min=0.0,
                max=max(10000.0, self.ring_tactile_threshold), step=1.0,
                initial_value=self.ring_tactile_threshold,
                disabled=not self.tactile_contact_stop,
            )
            disable_commands = self.observe_only or (self.simulate and not self.external_hand)
            self.enable_button = gui.add_button("Enable pose interpolation", disabled=disable_commands)
            self.pause_button = gui.add_button("Pause hand (hold feedback)", disabled=disable_commands)

        with gui.add_folder("Endpoint editor"):
            self.endpoint_selector = gui.add_dropdown(
                "Editing target", options=("A", "B"), initial_value="A"
            )
            self.endpoint_source = gui.add_text(
                "Target file", initial_value=str(self.pose_a_sample), disabled=True
            )
            self.save_endpoint_button = gui.add_button("Save edited endpoint")
            self.reload_endpoint_button = gui.add_button("Reload endpoint from disk")
            self.endpoint_sliders = {}
            for name in ALLEGRO_URDF_JOINT_NAMES:
                lower, upper = self.joint_limits[name]
                slider = gui.add_slider(
                    name, min=float(lower), max=float(upper), step=0.001,
                    initial_value=float(self.pose_a[name]),
                )
                self.endpoint_sliders[name] = slider

                @slider.on_update
                def _(_event, joint_name=name, handle=slider):
                    if self._syncing:
                        return
                    self._editable_pose()[joint_name] = float(handle.value)
                    self._update_interpolated_pose()

        @self.parameter_slider.on_update
        def _(_event):
            if not self._syncing:
                self._set_parameter(float(self.parameter_slider.value))
                self._update_interpolated_pose()

        @self.tactile_threshold_slider.on_update
        def _(_event):
            self.tactile_threshold = float(self.tactile_threshold_slider.value)

        @self.ring_tactile_threshold_slider.on_update
        def _(_event):
            self.ring_tactile_threshold = float(
                self.ring_tactile_threshold_slider.value
            )

        @self.endpoint_selector.on_update
        def _(_event):
            self.editing_endpoint = str(self.endpoint_selector.value)
            self._sync_endpoint_sliders()

        @self.enable_button.on_click
        def _(_event):
            self.command_enabled.set()
            self._set_status("INPUT → ALLEGRO")

        @self.pause_button.on_click
        def _(_event):
            self.command_enabled.clear()
            self._set_status("HOLDING CURRENT FEEDBACK")

        @self.save_endpoint_button.on_click
        def _(_event):
            self.save_endpoint()

        @self.reload_endpoint_button.on_click
        def _(_event):
            self.reload_endpoint()

    def _editable_pose(self) -> dict[str, float]:
        return self.pose_a if self.editing_endpoint == "A" else self.pose_b

    def _editable_sample(self) -> Path:
        return self.pose_a_sample if self.editing_endpoint == "A" else self.pose_b_sample

    def _set_status(self, value: str) -> None:
        self.command_status.value = value

    def _sync_endpoint_sliders(self) -> None:
        pose = self._editable_pose()
        self.endpoint_source.value = str(self._editable_sample())
        self._syncing = True
        try:
            for name, slider in self.endpoint_sliders.items():
                lower, upper = self.joint_limits[name]
                slider.value = float(np.clip(pose[name], lower, upper))
        finally:
            self._syncing = False

    def _safe_interpolated_pose(self) -> dict[str, float]:
        pose = interpolate_allegro_pose(self.pose_a, self.pose_b, self.parameter)
        return {
            name: float(np.clip(pose[name], *self.joint_limits[name]))
            for name in ALLEGRO_URDF_JOINT_NAMES
        }

    def _safe_interpolated_action(self) -> np.ndarray:
        action = np.asarray(
            urdf_qpos_to_retargeter_action(self._safe_interpolated_pose()), dtype=float
        )
        return np.clip(action, self.action_lower, self.action_upper)

    def _set_parameter(self, value: float) -> None:
        """Set blend parameter; unlock each finger only after its stop point."""
        new_value = float(np.clip(value, 0.0, 1.0))
        if new_value < self.parameter - 1e-6:
            self.moving_toward_b = False
            for finger, latch_parameter in self.contact_latch_parameter.items():
                if latch_parameter is not None and new_value <= latch_parameter + 1e-6:
                    self._release_contact_latch(finger)
        elif new_value > self.parameter + 1e-6:
            self.moving_toward_b = True
        self.parameter = new_value

    def _release_contact_latch(self, finger: str) -> None:
        self.contact_latched[finger] = False
        self.contact_latch_parameter[finger] = None
        self.contact_hold_action[finger] = None
        self.contact_latch_level[finger] = None
        self.contact_above_threshold_count[finger] = 0
        self.contact_below_threshold_count[finger] = 0

    def _update_interpolated_pose(self) -> None:
        self.parameter = float(np.clip(self.parameter, 0.0, 1.0))
        self._syncing = True
        try:
            self.parameter_slider.value = self.parameter
        finally:
            self._syncing = False
        self.parameter_text.value = f"{self.parameter:.3f}"
        pose = self._safe_interpolated_pose()
        self.viser_robot.update_cfg({name: pose[name] for name in ALLEGRO_URDF_JOINT_NAMES})

    def _update_input_parameter(self, now: float, *, update_ui: bool = True) -> None:
        """Apply the selected input source to the interpolation parameter.

        Quest A closes toward B (grasp), while B opens back toward A. Grip is
        reserved as the Quest xArm deadman switch.
        """
        direction = 0.0
        if self.input_source == "pedal":
            assert self.pedal is not None
            direction = float(self.pedal.get_direction())
            outer_status = {1.0: "LEFT PEDAL: GRASP", -1.0: "RIGHT PEDAL: OPEN", 0.0: "HOLD"}[direction]
            xarm_status = "ACTIVE" if self.xarm_deadman_pressed() else "HOLD"
            if update_ui:
                self.input_status.value = (
                    f"PEDAL: {outer_status}; CENTER/xArm: {xarm_status}"
                )
        elif self.input_source == "quest":
            assert self.quest_input is not None
            snapshot = self.quest_input.get_snapshot()
            updated_at = snapshot["updated_at"]
            error = snapshot["error"]
            if error is not None:
                if update_ui:
                    self.input_status.value = f"QUEST ERROR: {error}"
            elif updated_at is None:
                if update_ui:
                    self.input_status.value = "QUEST: WAITING FOR RIGHT CONTROLLER"
            elif now - updated_at > 1.5:
                if update_ui:
                    self.input_status.value = "QUEST: INPUT STALE; HOLD"
            else:
                direction = float(snapshot["primary"]) - float(snapshot["secondary"])
                if update_ui:
                    self.input_status.value = (
                        "QUEST RIGHT: "
                        f"A/grasp={'DOWN' if snapshot['primary'] else 'UP'}, "
                        f"B/open={'DOWN' if snapshot['secondary'] else 'UP'}, "
                        f"Grip/xArm={snapshot['grip']:.2f} "
                        f"({'ACTIVE' if self.xarm_deadman_pressed(now=now) else 'HOLD'})"
                    )
        else:
            if update_ui:
                self.input_status.value = "SLIDER: use Parameter: A ← → B"

        if self.last_time is not None and direction:
            self._set_parameter(self.parameter + direction * self.input_rate * (now - self.last_time))
        self.last_time = now
        if update_ui:
            self._update_interpolated_pose()

    @property
    def xarm_deadman_name(self) -> str:
        return "right Quest Grip" if self.input_source == "quest" else "middle pedal"

    def xarm_deadman_pressed(self, *, now: float | None = None) -> bool:
        """Return whether fresh Quest Grip or the middle pedal is active."""
        if self.input_source == "pedal":
            return self.pedal is not None and self.pedal.get_state() == 0
        if self.input_source != "quest" or self.quest_input is None:
            return False
        snapshot = self.quest_input.get_snapshot()
        return quest_grip_deadman_pressed(
            grip=float(snapshot["grip"]),
            updated_at=snapshot["grip_updated_at"],
            now=time.monotonic() if now is None else now,
            threshold=self.quest_grip_threshold,
            max_age=QUEST_POSE_MAX_AGE_S,
        )

    def quest_grip_teleop_state(self) -> int:
        """Map fresh pressed Grip to active state 0, otherwise hold state 1."""
        now = time.monotonic()
        active = self.xarm_deadman_pressed(now=now)
        if self.external_hand and not active:
            # CaptureSession does not request a hand target in state 1. Keep
            # the interpolation clock current so the next active frame cannot
            # integrate the entire hold duration as one large target jump.
            self.last_time = now
        return 0 if active else 1

    def attach_hand_controller(self, controller) -> None:
        """Enable publisher-level smoothing once valid hand feedback arrives."""
        self._hand_command_controller = controller
        self._hand_slew_configured = False

    def _configure_hand_slew(self, feedback: Mapping[str, Any]) -> None:
        if self._hand_slew_configured or self._hand_command_controller is None:
            return
        setter = getattr(self._hand_command_controller, "set_command_slew_rate", None)
        if setter is None:
            self._hand_slew_configured = True
            return
        endpoint_delta = np.abs(
            np.asarray(urdf_qpos_to_retargeter_action(self.pose_b), dtype=float)
            - np.asarray(urdf_qpos_to_retargeter_action(self.pose_a), dtype=float)
        )
        max_velocity = np.maximum(
            endpoint_delta * self.input_rate * HAND_SLEW_MARGIN,
            HAND_MIN_SLEW_RAD_S,
        )
        setter(max_velocity, initial_action=np.asarray(feedback["qpos"], dtype=float))
        self._hand_slew_configured = True

    def _ui_update_due(self, now: float) -> bool:
        next_update = getattr(self, "_next_ui_update_time", 0.0)
        if now < next_update:
            return False
        self._next_ui_update_time = now + 1.0 / UI_RATE_HZ
        return True

    def _tactile_is_fresh(self, feedback: Mapping[str, Any]) -> bool:
        tactile_time = feedback.get("tactile_time")
        return (
            tactile_time is not None
            and np.isfinite(tactile_time)
            and time.perf_counter() - float(tactile_time) <= TACTILE_MAX_AGE_S
        )

    def _update_tactile_raw_status(self, feedback: Mapping[str, Any]) -> None:
        tactile = feedback.get("tactile")
        if tactile is None:
            self.tactile_raw_status.value = "WAITING: no tactile packet"
            return
        values = np.asarray(tactile).reshape(-1)
        tactile_time = feedback.get("tactile_time")
        age_text = "unknown age"
        if tactile_time is not None and np.isfinite(tactile_time):
            age_text = f"age={time.perf_counter() - float(tactile_time):.3f}s"
        self.tactile_raw_status.value = (
            f"{np.array2string(values, separator=', ')} ({age_text})"
        )
        if values.size != len(ALLEGRO_FINGERS):
            self.tactile_raw_status.value += " (expected 4 tactile values)"
            return
        for finger, value in zip(ALLEGRO_FINGERS, values):
            self.tactile_value_sliders[finger].value = int(
                np.clip(float(value), 0, TACTILE_DISPLAY_MAX)
            )

    def _apply_tactile_contact_stop(
        self,
        desired: np.ndarray,
        feedback: Mapping[str, Any],
        *,
        update_ui: bool = True,
    ) -> np.ndarray:
        """Latch contacted fingers and resume them after contact is released."""
        def set_tactile_status(value: str) -> None:
            if update_ui:
                self.tactile_status.value = value

        feedback_action = np.asarray(feedback["qpos"], dtype=float)
        if not self.tactile_contact_stop:
            set_tactile_status("OFF")
            return desired

        tactile_is_fresh = self._tactile_is_fresh(feedback)
        levels = (
            allegro_tactile_finger_levels(feedback.get("tactile"))
            if tactile_is_fresh
            else None
        )
        released = []
        if levels is not None:
            for finger, level in levels.items():
                threshold = (
                    self.ring_tactile_threshold
                    if finger == "ring"
                    else self.tactile_threshold
                )
                if self.contact_latched[finger] and level < threshold:
                    self.contact_below_threshold_count[finger] += 1
                    if (
                        self.contact_below_threshold_count[finger]
                        >= TACTILE_RELEASE_DEBOUNCE_SAMPLES
                    ):
                        self._release_contact_latch(finger)
                        released.append(finger)
                else:
                    self.contact_below_threshold_count[finger] = 0

        if not self.moving_toward_b:
            stopped = [finger for finger in ALLEGRO_FINGERS if self.contact_latched[finger]]
            if not stopped:
                released_text = f"; RELEASED: {', '.join(released)}" if released else ""
                set_tactile_status("OPENING: all fingers moving" + released_text)
                return desired
            release_points = ", ".join(
                f"{finger}@{self.contact_latch_parameter[finger]:.3f}"
                f" (latched={self.contact_latch_level[finger]:.0f})"
                for finger in stopped
            )
            set_tactile_status(
                "OPENING: HOLD UNTIL PARAMETER REACHES " + release_points
            )
            return hold_contacted_allegro_fingers(
                desired,
                feedback_action,
                self.contact_latched,
                self.contact_hold_action,
            )
        if not tactile_is_fresh:
            set_tactile_status("NO FRESH TACTILE: HOLDING ALL FINGERS")
            return feedback_action.copy()
        if levels is None:
            set_tactile_status("INVALID TACTILE: HOLDING ALL FINGERS")
            return feedback_action.copy()
        arming = []
        for finger, level in levels.items():
            threshold = (
                self.ring_tactile_threshold
                if finger == "ring"
                else self.tactile_threshold
            )
            if level >= threshold:
                self.contact_above_threshold_count[finger] += 1
            else:
                self.contact_above_threshold_count[finger] = 0
            if (
                self.contact_above_threshold_count[finger] >= TACTILE_CONTACT_DEBOUNCE_SAMPLES
                and not self.contact_latched[finger]
            ):
                self.contact_latched[finger] = True
                self.contact_below_threshold_count[finger] = 0
                self.contact_latch_parameter[finger] = self.parameter
                self.contact_latch_level[finger] = level
                action_slice = ALLEGRO_FINGER_ACTION_SLICES[finger]
                # Do not keep replacing this target with later feedback: the
                # latter follows object-induced deflection and makes a
                # supposedly stopped finger drift after contact.
                self.contact_hold_action[finger] = feedback_action[action_slice].copy()
            if not self.contact_latched[finger] and self.contact_above_threshold_count[finger]:
                arming.append(
                    f"{finger} {self.contact_above_threshold_count[finger]}/"
                    f"{TACTILE_CONTACT_DEBOUNCE_SAMPLES}"
                )
        stopped = [finger for finger in ALLEGRO_FINGERS if self.contact_latched[finger]]
        levels_text = ", ".join(f"{finger}={levels[finger]:.0f}" for finger in ALLEGRO_FINGERS)
        held_text = (
            "; HOLD: " + ", ".join(
                f"{finger}@{self.contact_latch_parameter[finger]:.3f}"
                f" (latched={self.contact_latch_level[finger]:.0f})"
                for finger in stopped
            )
            if stopped else ""
        )
        arming_text = "; ARMING: " + ", ".join(arming) if arming else ""
        released_text = "; RELEASED: " + ", ".join(released) if released else ""
        suffix = held_text or "; MOVING"
        set_tactile_status(levels_text + suffix + arming_text + released_text)
        return hold_contacted_allegro_fingers(
            desired,
            feedback_action,
            self.contact_latched,
            self.contact_hold_action,
        )

    def command_action(self, feedback: Mapping[str, Any] | None) -> np.ndarray | None:
        """Return a safe command, or current feedback while the hand is paused."""
        now = time.monotonic()
        update_ui = self._ui_update_due(now)
        self._update_input_parameter(now, update_ui=update_ui)
        if feedback is None:
            return None
        if not feedback.get("is_connected", False):
            if update_ui:
                self.actual_status.value = (
                    f"WAITING: {feedback.get('state_topic', '/joint_states')}"
                )
            return None
        self._configure_hand_slew(feedback)
        if update_ui:
            actual = feedback_to_urdf_qpos(feedback["qpos"], feedback["joint_names"])
            self.actual_status.value = np.array2string(
                np.asarray([actual[name] for name in ALLEGRO_URDF_JOINT_NAMES]),
                precision=3,
                separator=",",
            )
            self._update_tactile_raw_status(feedback)
        if self.observe_only:
            if update_ui:
                self._set_status("OBSERVING ALLEGRO FEEDBACK (NO COMMANDS)")
            return None
        if not self.command_enabled.is_set():
            if update_ui:
                self._set_status("HOLDING CURRENT FEEDBACK")
                self.tactile_status.value = "COMMAND PAUSED"
            return np.asarray(feedback["qpos"], dtype=float).copy()
        command = self._safe_interpolated_action()
        command = self._apply_tactile_contact_stop(
            command,
            feedback,
            update_ui=update_ui,
        )
        if update_ui:
            preview = retargeter_action_to_urdf_qpos(command)
            self.viser_robot.update_cfg(
                {name: preview[name] for name in ALLEGRO_URDF_JOINT_NAMES}
            )
            self._set_status("INPUT → ALLEGRO")
        return command

    def _update_hand(self) -> None:
        if self.hand is None:
            return
        command = self.command_action(self.hand.get_data())
        if command is not None:
            self.hand.move(command)

    def save_endpoint(self) -> None:
        _save_pose_sample(self._editable_sample(), self._editable_pose())
        print(f"[allegro-pedal] saved {self.editing_endpoint}: {self._editable_sample()}")

    def reload_endpoint(self) -> None:
        _urdf, pose = _load_pose_sample(self._editable_sample())
        if self.editing_endpoint == "A":
            self.pose_a = pose
        else:
            self.pose_b = pose
        self._sync_endpoint_sliders()
        self._update_interpolated_pose()
        print(f"[allegro-pedal] reloaded {self.editing_endpoint}: {self._editable_sample()}")

    def run(self) -> None:
        listen_keyboard({"q": self.exit_event})
        print("[allegro-quest] Open the Viser URL. q + Enter exits.")
        period = 1.0 / COMMAND_RATE_HZ
        next_tick = time.monotonic()
        try:
            while not self.exit_event.is_set():
                self._update_hand()
                next_tick += period
                now = time.monotonic()
                if next_tick <= now:
                    next_tick = now
                    continue
                self.exit_event.wait(timeout=next_tick - now)
        finally:
            self.close()

    def close(self) -> None:
        self.exit_event.set()
        stop_listening()
        if self.hand is not None:
            self.hand.end()
            self.hand = None
        if self.pedal is not None:
            self.pedal.close()
            self.pedal = None
        if self.quest_input is not None:
            self.quest_input.close()
            self.quest_input = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pose-a", type=Path, default=DEFAULT_POSE_A)
    parser.add_argument("--pose-b", type=Path, default=DEFAULT_POSE_B)
    parser.add_argument("--source-pose-a", type=Path, default=DEFAULT_SOURCE_POSE_A)
    parser.add_argument("--source-pose-b", type=Path, default=DEFAULT_SOURCE_POSE_B)
    parser.add_argument(
        "--input-source", choices=("quest", "pedal", "slider"), default="quest",
        help="pose interpolation control source (default: quest)",
    )
    parser.add_argument(
        "--input-rate", "--pedal-rate", dest="input_rate", type=float, default=0.7,
        help="parameter change per second at full input (legacy alias: --pedal-rate)",
    )
    parser.add_argument(
        "--quest-openxr-bin", type=Path, default=DEFAULT_QUEST_OPENXR_BIN,
        help="path to quest-openxr run-input.sh launcher (legacy name retained)",
    )
    parser.add_argument(
        "--quest-headless", action="store_true",
        help="use a headless OpenXR session; focused streaming mode is the default",
    )
    parser.add_argument(
        "--no-pedal", action="store_true",
        help="legacy alias for --input-source=slider",
    )
    parser.add_argument(
        "--tactile-threshold", type=float, default=200.0,
        help="per-finger raw tactile maximum at which a closing finger latches (default: 200)",
    )
    parser.add_argument(
        "--ring-tactile-threshold", type=float, default=150.0,
        help="ring-finger raw tactile maximum at which it latches (default: 150)",
    )
    parser.add_argument(
        "--no-tactile-contact-stop", action="store_true",
        help="disable tactile finger holds and interpolate all fingers normally",
    )
    parser.add_argument(
        "--with-xarm", action="store_true",
        help=(
            "teleoperate xArm from right VIVE while the selected deadman is held: "
            "Quest Grip by default, or middle pedal with --input-source=pedal"
        ),
    )
    parser.add_argument(
        "--with-quest-xarm", action="store_true",
        help="teleoperate xArm from right Quest pose while Grip is held",
    )
    parser.add_argument(
        "--quest-grip-threshold", type=float, default=0.5,
        help="Grip value that enables Quest xArm teleoperation (default: 0.5)",
    )
    parser.add_argument(
        "--quest-arm-translation-scale", type=float, default=1.0,
        help="Quest controller translation multiplier for xArm (default: 1.0)",
    )
    parser.add_argument(
        "--xarm-servo-api", choices=("cartesian_aa", "angle_j"),
        default="cartesian_aa",
        help="xArm controller command API used by either xArm teleoperation mode",
    )
    behavior = parser.add_mutually_exclusive_group()
    behavior.add_argument("--simulate", action="store_true", help="URDF mesh only; do not create an Allegro driver")
    behavior.add_argument("--observe-only", action="store_true", help="show feedback but never send Allegro commands")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.with_xarm and args.with_quest_xarm:
        raise ValueError("--with-xarm and --with-quest-xarm are mutually exclusive")
    if (args.with_xarm or args.with_quest_xarm) and args.observe_only:
        raise ValueError("xArm teleoperation cannot be combined with --observe-only")
    if args.no_pedal:
        if args.input_source != "quest":
            raise ValueError("--no-pedal cannot be combined with --input-source")
        args.input_source = "slider"
    if args.with_xarm and args.input_source == "slider":
        raise ValueError(
            "--with-xarm requires --input-source=quest (Grip deadman) or pedal"
        )
    if args.with_quest_xarm and args.input_source != "quest":
        raise ValueError("--with-quest-xarm requires --input-source=quest")
    pose_a = ensure_editable_pose_copy(args.pose_a, args.source_pose_a)
    pose_b = ensure_editable_pose_copy(args.pose_b, args.source_pose_b)
    studio = AllegroPedalPoseStudio(
        pose_a_sample=pose_a,
        pose_b_sample=pose_b,
        simulate=args.simulate,
        observe_only=args.observe_only,
        input_source=args.input_source,
        input_rate=args.input_rate,
        quest_grip_threshold=args.quest_grip_threshold,
        quest_openxr_bin=args.quest_openxr_bin,
        quest_headless=args.quest_headless,
        tactile_contact_stop=not args.no_tactile_contact_stop,
        tactile_threshold=args.tactile_threshold,
        ring_tactile_threshold=args.ring_tactile_threshold,
        external_hand=False,
    )
    xarm_teleop = None
    try:
        if args.with_xarm:
            xarm_teleop = XArmVivePedalTeleop(studio, args.xarm_servo_api)
            xarm_teleop.start()
        elif args.with_quest_xarm:
            xarm_teleop = XArmQuestTeleop(
                studio,
                args.xarm_servo_api,
                grip_threshold=args.quest_grip_threshold,
                translation_scale=args.quest_arm_translation_scale,
            )
            xarm_teleop.start()
        studio.run()
    except KeyboardInterrupt:
        print("[allegro-pedal] stopped.")
    finally:
        if xarm_teleop is not None:
            xarm_teleop.close()


if __name__ == "__main__":
    main()
