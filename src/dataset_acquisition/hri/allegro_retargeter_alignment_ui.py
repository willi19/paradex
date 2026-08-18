#!/usr/bin/env python3
"""Capture and edit paired MANUS -> Allegro retargeter-alignment states.

Live mode records a MANUS right-hand frame together with the real Allegro
feedback qpos.  Press ``c`` then Enter in the launching terminal, or click
**Capture paired state** in the browser.  Opening a sample always starts from
the captured real pose; edited targets are stored separately and can be loaded
with **Load saved edited target**.

The UI starts by holding the current physical right-hand pose. **Resume**
switches to live MANUS retargeting; **Pause** freezes the latest feedback pose
as the persistent command target. Sliders edit the displayed target; in live
mode **Move right hand to sliders (hold)** explicitly sends that target.
With ``--observe-only``, it instead subscribes to and displays the real Allegro
feedback without ever issuing a hand command.
With ``--simulate``, it does not create an Allegro driver at all; the mesh
shows the URDF pose implied by the live MANUS retargeter output.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from paradex.io.robot_controller import get_hand
from paradex.io.teleop.vive.receiver import ViveManusROSReceiver
from paradex.retargetor.allegro_alignment import (
    ALLEGRO_FEEDBACK_JOINT_NAMES,
    ALLEGRO_RETARGETER_JOINT_NAMES,
    ALLEGRO_URDF_JOINT_NAMES,
    feedback_to_urdf_qpos,
    retargeter_action_to_live_controller_qpos,
    retargeter_action_to_urdf_qpos,
    urdf_qpos_from_hand_qpos,
    urdf_qpos_to_retargeter_action,
)
from paradex.retargetor.hand_regargetor import (
    allegro,
    allegro_v5,
    clip_allegro_v5_safe_action,
)
from paradex.retargetor.allegro_v5_wonik import AllegroV5WonikManusRetargeter
from paradex.retargetor.allegro_v5_anyteleop import AllegroV5AnyTeleopRetargeter
from paradex.retargetor.hand_alignment_common import (
    HandStateContractError,
    read_json,
    serialize_manus_frame,
    sha256_file,
    write_json,
)
from paradex.utils.keyboard_listener import listen_keyboard, stop_listening
from paradex.utils.path import rsc_path, shared_dir
from paradex.visualization.visualizer.viser import ViserViewer


SCHEMA_VERSION = 1
# Use the hand-only model: live capture has no xArm controller dependency.
DEFAULT_URDF = Path(rsc_path) / "robot" / "allegro.urdf"
COMMAND_RATE_HZ = 30.0
DEFAULT_MANUS_TOPICS = ("/manus_glove_0", "/manus_glove_1")
_V5_HAND_NAMES = frozenset(("allegro_v5", "allegro_v5_wonik"))
RETARGETER_MODES = ("direct", "anyteleop")
_MANUS_ERGONOMIC_FIELDS = (
    "ThumbMCPStretch", "ThumbPIPStretch", "ThumbDIPStretch", "ThumbMCPSpread",
    "IndexMCPStretch", "IndexPIPStretch", "IndexDIPStretch", "IndexSpread",
    "MiddleMCPStretch", "MiddlePIPStretch", "MiddleDIPStretch", "MiddleSpread",
    "RingMCPStretch", "RingPIPStretch", "RingDIPStretch", "RingSpread",
    "PinkyMCPStretch", "PinkyPIPStretch", "PinkyDIPStretch", "PinkySpread",
)


def _make_retargeter(hand_name: str, mode: str):
    """Build a UI-only retargeter without changing the capture pipeline."""
    if mode not in RETARGETER_MODES:
        raise ValueError(f"Unsupported Allegro alignment retargeter: {mode}")
    if mode == "anyteleop":
        if hand_name not in _V5_HAND_NAMES:
            raise ValueError("AnyTeleop geometric retargeting requires --hand allegro_v5")
        return AllegroV5AnyTeleopRetargeter()
    if hand_name == "allegro_v5_wonik":
        return AllegroV5WonikManusRetargeter()
    return allegro_v5 if hand_name == "allegro_v5" else allegro


def _live_retargeter_kwargs(hand_name: str, mode: str, ergonomics: dict) -> dict:
    """Pass MANUS ergonomic angles through the direct v5 path only.

    This is deliberately the same input contract as ``Retargetor.get_action``
    in the live capture path.  The AnyTeleop implementation owns its separate
    geometric input model and must not receive these keyword arguments.
    """
    if mode == "direct" and hand_name in _V5_HAND_NAMES:
        return {"ergonomics": ergonomics}
    return {}


def _retargeter_function_name(hand_name: str, mode: str) -> str:
    if mode == "anyteleop":
        return "paradex.retargetor.allegro_v5_anyteleop.AllegroV5AnyTeleopRetargeter"
    return (
        "paradex.retargetor.hand_regargetor.allegro_v5"
        if hand_name == "allegro_v5"
        else (
            "paradex.retargetor.allegro_v5_wonik.AllegroV5WonikManusRetargeter"
            if hand_name == "allegro_v5_wonik"
            else "paradex.retargetor.hand_regargetor.allegro"
        )
    )


def _clip_v5_action_for_hand(hand_name: str, action: np.ndarray) -> np.ndarray:
    if hand_name not in _V5_HAND_NAMES:
        raise ValueError(f"Not an Allegro v5 hand: {hand_name}")
    return clip_allegro_v5_safe_action(action)


def _retargeter_action_to_preview_qpos(
    hand_name: str, action: np.ndarray
) -> dict[str, float]:
    """Convert a retargeter output into the hand-only URDF preview pose."""
    action = np.asarray(action, dtype=np.float64).reshape(-1)
    if hand_name in _V5_HAND_NAMES:
        return retargeter_action_to_urdf_qpos(action)
    if action.shape != (len(ALLEGRO_URDF_JOINT_NAMES),):
        raise ValueError(f"Expected {len(ALLEGRO_URDF_JOINT_NAMES)} Allegro joints, got {action.shape}")
    return dict(zip(ALLEGRO_URDF_JOINT_NAMES, action))


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sample_name(root: Path) -> Path:
    existing = [int(path.name) for path in root.iterdir() if path.is_dir() and path.name.isdigit()]
    return root / f"{(max(existing, default=-1) + 1):06d}"


def _alignment_samples(root: Path) -> list[Path]:
    """Return only numbered, directly-owned capture directories in order."""
    return sorted(
        (path for path in root.iterdir() if path.is_dir() and path.name.isdigit()),
        key=lambda path: int(path.name),
    )


def _delete_alignment_sample(sample_path: Path, root: Path) -> None:
    """Delete one explicitly selected numbered capture, never its session root."""
    sample_path = sample_path.resolve()
    root = root.resolve()
    if sample_path.parent != root or not sample_path.name.isdigit():
        raise ValueError(f"Refusing to delete a capture outside {root}: {sample_path}")
    shutil.rmtree(sample_path)


def _normalize_feedback(values: Any, names: Any) -> tuple[np.ndarray, tuple[str, ...]]:
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    names = ALLEGRO_FEEDBACK_JOINT_NAMES if not names else tuple(str(name) for name in names)
    if values.shape != (len(names),):
        raise HandStateContractError(
            f"Allegro feedback has {values.shape[0]} values for {len(names)} names"
        )
    feedback_to_urdf_qpos(values, names)
    return values, tuple(names)


def _load_sample(sample_dir: Path) -> dict[str, Any]:
    metadata = read_json(sample_dir / "metadata.json")
    if int(metadata.get("schema_version", -1)) != SCHEMA_VERSION:
        raise ValueError(f"Unsupported alignment sample schema: {metadata.get('schema_version')}")
    urdf_path = Path(metadata["robot"]["urdf_path"])
    if not urdf_path.exists():
        raise FileNotFoundError(f"Saved URDF no longer exists: {urdf_path}")
    if sha256_file(urdf_path) != metadata["robot"]["urdf_sha256"]:
        raise ValueError(
            "Saved URDF bytes differ from the capture-time URDF; refusing to "
            "claim an exact reconstruction. Restore the recorded URDF version."
        )
    return {
        "metadata": metadata,
        "feedback": read_json(sample_dir / "robot_feedback.json"),
        "target": read_json(sample_dir / "target_robot.json"),
        "urdf_path": urdf_path,
    }


class AllegroRetargeterAlignmentStudio:
    def __init__(
        self,
        *,
        urdf_path: Path,
        output_root: Path | None = None,
        source_sample: Path | None = None,
        live: bool = False,
        hand_control: bool = False,
        use_vive: bool = False,
        observe_only: bool = False,
        simulate: bool = False,
        manus_topics: tuple[str, ...] = DEFAULT_MANUS_TOPICS,
        hand_name: str = "allegro_v5",
        retargeter_mode: str = "direct",
    ):
        self.urdf_path = urdf_path.resolve()
        self.output_root = output_root
        self.source_sample = source_sample.resolve() if source_sample is not None else None
        self.sample_root = self.source_sample.parent if self.source_sample is not None else None
        self.live = live
        self.hand_control = bool(hand_control)
        self.use_vive = use_vive
        self.observe_only = observe_only
        self.simulate = simulate
        if self.observe_only and self.simulate:
            raise ValueError("observe_only and simulate cannot both be enabled")
        self.manus_topics = tuple(manus_topics)
        self.hand_name = hand_name
        self.retargeter_mode = retargeter_mode
        self.retargeter = _make_retargeter(hand_name, retargeter_mode)
        self.lock = threading.Lock()
        self._syncing_sliders = False
        self.capture_event = threading.Event()
        self.exit_event = threading.Event()
        self.right_retargeting_enabled = threading.Event()
        self.slider_follow_enabled = threading.Event()
        self.hold_controller_target: np.ndarray | None = None
        self.actual_hand_qpos: dict[str, float] | None = None
        # False: sliders/mesh mirror physical feedback.  True: sliders/mesh
        # represent an operator-prepared target (including slider hold).
        self.slider_target_active = False
        self.teleop = None
        self.hand = None
        self.latest_manus_frame: dict[str, np.ndarray] | None = None
        self.latest_manus_action: np.ndarray | None = None
        self.saved_manus_ergonomics: dict[str, float] = {}

        # A capture is a single state, not a trajectory: omit Viser's generic
        # playback/video/scene-control panels entirely.
        self.viewer = ViserViewer(
            scene_title="MANUS → Allegro alignment", show_player=False
        )
        # This is a hand-only URDF.  The ``include_arm_meshes=False`` option
        # filters meshes using F1-specific link names and would hide every
        # Allegro mesh, so keep all meshes visible here.
        self.viewer.add_robot("allegro", str(self.urdf_path), include_arm_meshes=True)
        # ``RobotModule`` stores the kinematic state, whereas
        # ``ViserRobotModule`` additionally pushes transforms to browser mesh
        # frames.  Keep both references and always update through the latter.
        self.viser_robot = self.viewer.robot_dict["allegro"]
        self.robot = self.viser_robot.urdf
        self.urdf_joint_names = tuple(self.robot.get_joint_names())
        self.joint_limits = self.robot.get_joint_limits()
        # Some Allegro joints (notably thumb_base) do not allow zero.  Keep
        # the pre-feedback preview within the URDF limits so Viser can create
        # its sliders before the first physical state arrives.
        self.hand_qpos = {
            name: float(np.clip(0.0, *self.joint_limits[name]))
            for name in ALLEGRO_URDF_JOINT_NAMES
        }
        self.captured_hand_qpos: dict[str, float] | None = None
        self.out_of_limit_joints: tuple[str, ...] = ()

        if source_sample is not None:
            loaded = _load_sample(source_sample)
            self.urdf_path = loaded["urdf_path"]
            # Derive the preview from named raw controller feedback.  This
            # also repairs early v5 samples whose cached URDF map used the
            # wrong index/middle/ring/thumb driver block order.
            feedback = loaded["feedback"]
            captured_qpos = feedback_to_urdf_qpos(
                feedback["controller_qpos_rad"],
                feedback["controller_joint_names"],
            )
            self.captured_hand_qpos = dict(captured_qpos)
            self.hand_qpos = {
                name: float(captured_qpos[name]) for name in ALLEGRO_URDF_JOINT_NAMES
            }
            self.saved_manus_ergonomics = self._read_saved_manus_ergonomics(
                source_sample
            )

        self._build_gui()
        self._refresh_pose_notice()
        self._update_robot()

        if live:
            self.teleop = ViveManusROSReceiver(
                hand_side="right",
                require_left_control=False,
                use_vive=self.use_vive,
                manus_topics=self.manus_topics,
            )
            if self.simulate:
                self._set_command_status("SIMULATING MANUS → ALLEGRO URDF (NO DRIVER)")
                print("[alignment] Simulation mode: no Allegro driver or ROS hand topic is opened.")
            else:
                # Use the exact factory path used by CaptureSession/capture_robot:
                # Allegro v5 resolves ``hand_side='right'`` to
                # /right/allegroHand_0/joint_states.
                self.hand = get_hand(self.hand_name, hand_side="right")
            if self.observe_only:
                self._set_command_status("OBSERVING ALLEGRO FEEDBACK (NO COMMANDS)")
                print("[alignment] Right-hand mode: observation only; no commands will be sent.")
            elif not self.simulate:
                self._set_command_status("WAITING FOR ALLEGRO FEEDBACK")
                print("[alignment] Right-hand command mode: HOLD current pose.")
        elif self.hand_control:
            # Open-mode hand control deliberately creates no teleop receiver:
            # a saved target can be inspected and applied without MANUS/VIVE.
            self.hand = get_hand(self.hand_name, hand_side="right")
            self._set_command_status("WAITING FOR ALLEGRO FEEDBACK")
            print("[alignment] Open-mode hand control: MANUS/VIVE are disabled; holding feedback pose.")

    def _build_gui(self) -> None:
        gui = self.viewer.server.gui
        self.capture_button = None
        self.pause_button = None
        self.resume_button = None
        self.apply_slider_button = None
        self.follow_sliders_button = None
        with gui.add_folder("Capture controls" if self.live else "Saved pose"):
            self.command_status = gui.add_text(
                "Right-hand command state",
                initial_value=(
                    "OBSERVING ALLEGRO FEEDBACK (NO COMMANDS)"
                    if self.live and self.simulate
                    else "OBSERVING ALLEGRO FEEDBACK (NO COMMANDS)"
                    if self.live and self.observe_only
                    else "WAITING FOR ALLEGRO FEEDBACK"
                    if self.live
                    else "CAPTURE-TIME FEEDBACK POSE"
                ),
                disabled=True,
            )
            self.pose_notice = gui.add_text(
                "Saved-pose notice", initial_value="", disabled=True
            )
            if self.live:
                self.retargeter_selector = gui.add_dropdown(
                    "Live retargeter",
                    options=RETARGETER_MODES,
                    initial_value=self.retargeter_mode,
                )
                self.capture_button = gui.add_button(
                    "Capture paired state", disabled=self.simulate
                )
                self.pause_button = gui.add_button(
                    "Pause right hand (hold pose)",
                    disabled=self.observe_only or self.simulate,
                )
                self.resume_button = gui.add_button(
                    "Resume MANUS → right hand",
                    disabled=self.observe_only or self.simulate,
                )
                self.apply_slider_button = gui.add_button(
                    "Apply sliders to right hand now (hold)",
                    disabled=self.observe_only or self.simulate,
                )
                self.follow_sliders_button = gui.add_button(
                    "Toggle continuous slider control",
                    disabled=self.observe_only or self.simulate,
                )
            elif self.hand_control:
                self.pause_button = gui.add_button(
                    "Pause right hand (hold feedback)",
                    disabled=self.observe_only,
                )
                self.apply_slider_button = gui.add_button(
                    "Apply sliders to right hand now (hold)",
                    disabled=self.observe_only,
                )
                self.follow_sliders_button = gui.add_button(
                    "Toggle continuous slider control",
                    disabled=self.observe_only,
                )
            self.save_target_button = gui.add_button(
                "Save edited robot target", disabled=self.source_sample is None
            )
            self.load_target_button = gui.add_button(
                "Load saved edited target", disabled=self.source_sample is None
            )
        if not self.live:
            self.retargeter_selector = None

        self.previous_sample_button = None
        self.next_sample_button = None
        self.delete_sample_button = None
        self.sample_position = None
        if not self.live:
            with gui.add_folder("Capture browser"):
                self.sample_position = gui.add_text(
                    "Current capture", initial_value="", disabled=True
                )
                self.previous_sample_button = gui.add_button("Previous capture")
                self.next_sample_button = gui.add_button("Next capture")
                self.delete_sample_button = gui.add_button("Delete current capture")

            self.saved_manus_sliders = {}
            with gui.add_folder("Captured MANUS ergonomics (degrees, read only)"):
                for field in _MANUS_ERGONOMIC_FIELDS:
                    value = self.saved_manus_ergonomics.get(field, 0.0)
                    self.saved_manus_sliders[field] = gui.add_slider(
                        field,
                        min=-80.0,
                        max=80.0,
                        step=0.1,
                        initial_value=float(np.clip(value, -80.0, 80.0)),
                        disabled=True,
                    )
        else:
            self.saved_manus_sliders = {}

        if self.live or self.hand_control:
            with gui.add_folder(
                "Live MANUS input" if self.live else "Live Allegro hand feedback"
            ):
                self.allegro_status = gui.add_text(
                    "Allegro feedback", initial_value="NOT STARTED", disabled=True
                )
                self.command_target_status = gui.add_text(
                    "Command target / max error",
                    initial_value="No command target",
                    disabled=True,
                )
                self.actual_feedback_status = gui.add_text(
                    (
                        "Simulated Allegro qpos (URDF order)"
                        if self.simulate
                        else "Actual Allegro qpos (URDF order)"
                    ),
                    initial_value=("WAITING FOR MANUS" if self.simulate else "WAITING FOR FEEDBACK"),
                    disabled=True,
                )
                if self.live:
                    self.manus_status = gui.add_text(
                        "Right MANUS state", initial_value="WAITING", disabled=True
                    )
                    self.manus_topics_status = gui.add_text(
                        "Subscribed MANUS topics",
                        initial_value=", ".join(self.manus_topics),
                        disabled=True,
                    )
                    self.manus_transport = gui.add_text(
                        "MANUS transport", initial_value="No messages yet", disabled=True
                    )
                    self.manus_values = gui.add_text(
                        "Wrist + fingertips [m]", initial_value="", disabled=True
                    )
                    self.manus_action = gui.add_text(
                        "Retargeter action [rad]", initial_value="", disabled=True
                    )

        self.sliders = {}
        with gui.add_folder("Target Allegro hand (URDF radians)"):
            for name in ALLEGRO_URDF_JOINT_NAMES:
                lower, upper = self.joint_limits[name]
                slider = gui.add_slider(
                    name,
                    min=float(lower),
                    max=float(upper),
                    step=0.001,
                    # Encoders can report a small limit overshoot.  Keep the
                    # raw value for URDF reconstruction, but Viser sliders
                    # cannot be constructed outside their declared bounds.
                    initial_value=float(np.clip(self.hand_qpos[name], lower, upper)),
                    disabled=self.observe_only or self.simulate,
                )
                self.sliders[name] = slider

                @slider.on_update
                def _(_event, joint_name=name, handle=slider):
                    if self._syncing_sliders:
                        return
                    with self.lock:
                        self.hand_qpos[joint_name] = float(handle.value)
                        self.slider_target_active = True
                    self._update_robot()
                    if self.slider_follow_enabled.is_set():
                        self.apply_slider_target_now(announce=False)

        if self.live:
            @self.retargeter_selector.on_update
            def _(_event):
                try:
                    self.set_retargeter_mode(str(self.retargeter_selector.value))
                except Exception as exc:
                    self.retargeter_selector.value = self.retargeter_mode
                    print(f"[alignment] retargeter switch rejected: {exc}", file=sys.stderr)

            @self.capture_button.on_click
            def _(_event):
                self.capture_event.set()

            @self.pause_button.on_click
            def _(_event):
                self.pause_right_hand()

            @self.resume_button.on_click
            def _(_event):
                self.resume_right_hand()

            @self.apply_slider_button.on_click
            def _(_event):
                self.apply_slider_target_now()

            @self.follow_sliders_button.on_click
            def _(_event):
                self.toggle_continuous_slider_control()

        elif self.hand_control:
            @self.pause_button.on_click
            def _(_event):
                self.pause_right_hand()

            @self.apply_slider_button.on_click
            def _(_event):
                self.apply_slider_target_now()

            @self.follow_sliders_button.on_click
            def _(_event):
                self.toggle_continuous_slider_control()

        @self.save_target_button.on_click
        def _(_event):
            try:
                self.save_edited_target()
            except Exception as exc:
                print(f"[alignment] unable to save target: {exc}", file=sys.stderr)

        @self.load_target_button.on_click
        def _(_event):
            self.load_saved_target()

        if not self.live:
            @self.previous_sample_button.on_click
            def _(_event):
                self._select_relative_sample(-1)

            @self.next_sample_button.on_click
            def _(_event):
                self._select_relative_sample(1)

            @self.delete_sample_button.on_click
            def _(_event):
                self._delete_current_sample()

            self._refresh_sample_browser()

    def set_retargeter_mode(self, mode: str) -> None:
        """Switch the live UI experiment without touching capture_robot.py."""
        mode = str(mode)
        if mode == self.retargeter_mode:
            return
        self.retargeter = _make_retargeter(self.hand_name, mode)
        self.retargeter_mode = mode
        self.latest_manus_action = None
        if self.live:
            self._set_command_status(f"RETARGETER SWITCHED: {mode.upper()}")
        print(f"[alignment] live retargeter: {mode}")

    def _update_robot(self) -> None:
        with self.lock:
            qpos = urdf_qpos_from_hand_qpos(self.hand_qpos, self.urdf_joint_names)
        self.viser_robot.update_cfg(qpos)

    def _refresh_pose_notice(self) -> None:
        self.out_of_limit_joints = tuple(
            name
            for name in ALLEGRO_URDF_JOINT_NAMES
            if not self.joint_limits[name][0] <= self.hand_qpos[name] <= self.joint_limits[name][1]
        )
        if self.out_of_limit_joints:
            self.pose_notice.value = (
                "Raw captured feedback is outside URDF limits for "
                + ", ".join(self.out_of_limit_joints)
                + "; robot preview uses raw values, sliders show nearest limit."
            )
        else:
            self.pose_notice.value = ""

    def set_hand_qpos(self, values: Mapping[str, float]) -> None:
        with self.lock:
            for name in ALLEGRO_URDF_JOINT_NAMES:
                self.hand_qpos[name] = float(values[name])
        self._syncing_sliders = True
        try:
            for name in ALLEGRO_URDF_JOINT_NAMES:
                lower, upper = self.joint_limits[name]
                self.sliders[name].value = float(
                    np.clip(values[name], lower, upper)
                )
        finally:
            self._syncing_sliders = False
        self._refresh_pose_notice()
        self._update_robot()

    def load_saved_target(self) -> None:
        """Load the saved target and keep it visible against live feedback."""
        if self.source_sample is None:
            raise RuntimeError("Open or capture a sample before loading a target.")
        target = read_json(self.source_sample / "target_robot.json")
        # Old captures cached v5 target_robot.json using the wrong finger
        # block order.  An unedited target is just a duplicate of capture
        # feedback, so reconstruct it from the retained raw named feedback.
        if not target.get("edited", False) and self.captured_hand_qpos is not None:
            values = self.captured_hand_qpos
            status = "CAPTURE-TIME FEEDBACK POSE"
            message = "[alignment] restored capture-time raw feedback pose."
        else:
            values = target["urdf_hand_qpos_rad"]
            status = "EDITED TARGET POSE"
            message = "[alignment] loaded saved edited target pose."
        # In --drive-hand mode feedback still arrives at 100 Hz.  Mark this
        # as an operator target before updating sliders so feedback cannot
        # overwrite the newly loaded pose before it is applied.
        with self.lock:
            self.slider_target_active = True
        self.set_hand_qpos(values)
        self._set_command_status(status)
        print(message)

    def _set_command_status(self, value: str) -> None:
        self.command_status.value = value

    def _show_command_target(self, target_qpos: Mapping[str, float]) -> None:
        """Show command/feedback tracking in the same semantic order as sliders."""
        if not (self.live or self.hand_control):
            return
        target = np.asarray(
            [target_qpos[name] for name in ALLEGRO_URDF_JOINT_NAMES], dtype=np.float64
        )
        with self.lock:
            actual_qpos = None if self.actual_hand_qpos is None else dict(self.actual_hand_qpos)
        if actual_qpos is None:
            error_text = "actual feedback pending"
        else:
            actual = np.asarray(
                [actual_qpos[name] for name in ALLEGRO_URDF_JOINT_NAMES],
                dtype=np.float64,
            )
            error_text = f"max actual error={np.max(np.abs(target - actual)):.3f} rad"
        self.command_target_status.value = (
            "target=" + np.array2string(target, precision=3, separator=",")
            + " | " + error_text
        )

    def _refresh_sample_browser(self) -> list[Path]:
        """Update Open-mode navigation controls from the current session root."""
        if self.live or self.sample_root is None or self.sample_position is None:
            return []
        samples = _alignment_samples(self.sample_root)
        if self.source_sample is None or self.source_sample not in samples:
            self.sample_position.value = "No capture selected"
            self.previous_sample_button.disabled = True
            self.next_sample_button.disabled = True
            self.delete_sample_button.disabled = True
            return samples
        index = samples.index(self.source_sample)
        self.sample_position.value = f"{self.source_sample.name} ({index + 1}/{len(samples)})"
        self.previous_sample_button.disabled = index == 0
        self.next_sample_button.disabled = index == len(samples) - 1
        self.delete_sample_button.disabled = False
        return samples

    @staticmethod
    def _read_saved_manus_ergonomics(sample_path: Path) -> dict[str, float]:
        """Load finite, named MANUS ergonomic angles from one capture."""
        raw = read_json(sample_path / "manus_ergonomics.json")
        return {
            field: float(raw[field])
            for field in _MANUS_ERGONOMIC_FIELDS
            if field in raw and np.isfinite(float(raw[field]))
        }

    def _refresh_saved_manus_ergonomics(self) -> None:
        """Make open-mode read-only sliders follow the selected capture."""
        for field, slider in self.saved_manus_sliders.items():
            slider.value = float(
                np.clip(self.saved_manus_ergonomics.get(field, 0.0), -80.0, 80.0)
            )

    def _load_selected_sample(self, sample_path: Path) -> None:
        """Replace the displayed single pose without recreating the Viser server."""
        loaded = _load_sample(sample_path)
        if loaded["urdf_path"].resolve() != self.urdf_path:
            raise ValueError(
                "Selected capture uses a different URDF; open it in a new UI session."
            )
        feedback = loaded["feedback"]
        captured_qpos = feedback_to_urdf_qpos(
            feedback["controller_qpos_rad"], feedback["controller_joint_names"]
        )
        self.source_sample = sample_path
        self.captured_hand_qpos = dict(captured_qpos)
        self.saved_manus_ergonomics = self._read_saved_manus_ergonomics(sample_path)
        self._refresh_saved_manus_ergonomics()
        self.set_hand_qpos(captured_qpos)
        self.save_target_button.disabled = False
        self.load_target_button.disabled = False
        self._set_command_status("CAPTURE-TIME FEEDBACK POSE")
        self._refresh_sample_browser()
        print(f"[alignment] opened capture: {sample_path}")

    def _select_relative_sample(self, direction: int) -> None:
        samples = self._refresh_sample_browser()
        if self.source_sample is None or self.source_sample not in samples:
            return
        index = samples.index(self.source_sample) + int(direction)
        if 0 <= index < len(samples):
            try:
                self._load_selected_sample(samples[index])
            except Exception as exc:
                print(f"[alignment] unable to open capture: {exc}", file=sys.stderr)

    def _delete_current_sample(self) -> None:
        if self.source_sample is None or self.sample_root is None:
            return
        samples = _alignment_samples(self.sample_root)
        if self.source_sample not in samples:
            self._refresh_sample_browser()
            return
        current_index = samples.index(self.source_sample)
        deleted = self.source_sample
        try:
            _delete_alignment_sample(deleted, self.sample_root)
        except Exception as exc:
            print(f"[alignment] unable to delete capture: {exc}", file=sys.stderr)
            return
        print(f"[alignment] deleted capture: {deleted}")

        remaining = _alignment_samples(self.sample_root)
        if remaining:
            self._load_selected_sample(remaining[min(current_index, len(remaining) - 1)])
            return

        self.source_sample = None
        self.captured_hand_qpos = None
        self.saved_manus_ergonomics = {}
        self._refresh_saved_manus_ergonomics()
        self.save_target_button.disabled = True
        self.load_target_button.disabled = True
        self._set_command_status("NO CAPTURES REMAIN")
        self._refresh_sample_browser()

    def _show_manus_input(self, frame: Mapping[str, np.ndarray], action: np.ndarray) -> None:
        """Show the live MANUS values that are fed into the retargeter."""
        names = ("wrist", "thumb_distal", "index_distal", "middle_distal", "ring_distal", "pinky_distal")
        positions = {
            name: np.asarray(frame[name], dtype=np.float64)[:3, 3].round(4).tolist()
            for name in names
            if name in frame
        }
        self.manus_status.value = "RECEIVING" + (" + VIVE" if self.use_vive else " (MANUS-only)")
        self.manus_values.value = json.dumps(positions, separators=(",", ":"))
        self.manus_action.value = np.array2string(
            np.asarray(action, dtype=np.float64), precision=3, separator=","
        )

    def _poll_manus_input(self) -> None:
        """Refresh the UI from MANUS even while Allegro is offline or holding."""
        if not self.live or self.teleop is None:
            return
        diagnostics = self.teleop.get_diagnostics()
        right = diagnostics["Right"]
        topic_counts = diagnostics["topic_message_counts"]
        self.manus_topics_status.value = ", ".join(
            f"{topic}: {topic_counts.get(topic, 0)} msg" for topic in diagnostics["manus_topics"]
        )
        age = right["last_age_s"]
        age_text = "never" if age is None else f"{age:.3f}s ago"
        self.manus_transport.value = (
            f"right valid={right['valid_messages']}, invalid={right['invalid_messages']}, "
            f"last={age_text}, topic={right['last_topic'] or '-'}, "
            f"glove_id={right['glove_id'] if right['glove_id'] is not None else '-'}"
        )

        manus = self.teleop.get_data()
        frame = manus.get("Right")
        self.latest_manus_frame = None
        self.latest_manus_action = None
        if frame is None:
            if right["last_error"]:
                self.manus_status.value = f"INVALID RIGHT MANUS: {right['last_error']}"
            elif right["valid_messages"]:
                self.manus_status.value = f"RIGHT MANUS STALE ({age_text})"
            elif diagnostics["unknown_side_messages"]:
                self.manus_status.value = (
                    "MANUS SIDE INVALID: "
                    f"{diagnostics['last_unknown_side']!r}"
                )
            else:
                self.manus_status.value = "WAITING FOR RIGHT MANUS"
            return

        ergonomics = manus.get("ergonomics", {}).get("Right") or {}
        kwargs = _live_retargeter_kwargs(
            self.hand_name, self.retargeter_mode, ergonomics
        )
        action = self.retargeter(frame, **kwargs)
        if action is None:
            self.manus_status.value = "MANUS FRAME INVALID FOR RETARGETER"
            return
        self.latest_manus_frame = frame
        self.latest_manus_action = np.asarray(action, dtype=np.float64)
        self._show_manus_input(frame, self.latest_manus_action)
        if self.simulate:
            qpos = _retargeter_action_to_preview_qpos(self.hand_name, self.latest_manus_action)
            self.actual_feedback_status.value = np.array2string(
                np.asarray([qpos[name] for name in ALLEGRO_URDF_JOINT_NAMES]),
                precision=3,
                separator=",",
            )
            self.allegro_status.value = "SIMULATION: MANUS retargeter → URDF (NO ALLEGRO CONNECTION)"
            self._set_command_status("SIMULATING MANUS → ALLEGRO URDF (NO DRIVER)")
            self.set_hand_qpos(qpos)

    def _show_allegro_connection(self, feedback: Mapping[str, Any]) -> bool:
        """Display actual feedback availability; this is the connection criterion."""
        connected = bool(feedback.get("is_connected", False)) and feedback.get("qpos") is not None
        if not connected:
            topic = feedback.get("state_topic", "/joint_states")
            self.allegro_status.value = f"WAITING: no valid feedback on {topic}"
            return False
        values, names = _normalize_feedback(
            feedback.get("qpos"), feedback.get("joint_names")
        )
        topic = feedback.get("state_topic", "/joint_states")
        self.allegro_status.value = (
            f"CONNECTED: {len(names)} Allegro joints on {topic}"
        )
        self._sync_preview_from_feedback(values, names)
        return True

    def _sync_preview_from_feedback(self, values: np.ndarray, names: tuple[str, ...]) -> None:
        """Reflect feedback unless an operator is actively holding a slider target."""
        qpos = feedback_to_urdf_qpos(values, names)
        with self.lock:
            self.actual_hand_qpos = dict(qpos)
            slider_target_active = self.slider_target_active
        self.actual_feedback_status.value = np.array2string(
            np.asarray([qpos[name] for name in ALLEGRO_URDF_JOINT_NAMES]),
            precision=3,
            separator=",",
        )
        if not slider_target_active:
            # This updates both the target sliders and the displayed mesh to
            # the current physical feedback while MANUS/live tracking runs.
            self.set_hand_qpos(qpos)

    def pause_right_hand(self) -> None:
        """Freeze the current physical feedback as the persistent hold target."""
        if self.observe_only or self.simulate:
            self._set_command_status("OBSERVING ALLEGRO FEEDBACK (NO COMMANDS)")
            return
        self.right_retargeting_enabled.clear()
        self.slider_follow_enabled.clear()
        self.hold_controller_target = None
        self.slider_target_active = False
        try:
            self._ensure_hold_target()
        except Exception as exc:
            print(f"[alignment] waiting for Allegro feedback before holding: {exc}", file=sys.stderr)
        print("[alignment] Right-hand commands paused: holding current feedback pose.")
        self._set_command_status("HOLDING CURRENT FEEDBACK")

    def resume_right_hand(self) -> None:
        if not self.live or self.observe_only or self.simulate:
            return
        self.right_retargeting_enabled.set()
        self.slider_follow_enabled.clear()
        self.slider_target_active = False
        self._set_command_status("MANUS → RIGHT HAND")
        print("[alignment] Right-hand commands resumed from live MANUS retargeting.")

    def hold_slider_target(self, *, announce: bool = True) -> None:
        """Command the browser slider pose and keep holding it physically."""
        if not (self.live or self.hand_control) or self.observe_only or self.simulate:
            return
        self.right_retargeting_enabled.clear()
        with self.lock:
            self.slider_target_active = True
            if self.hand_name in _V5_HAND_NAMES:
                # The v5 driver order is index/middle/ring/thumb, and manual
                # slider commands must obey the same approved safety range as
                # live MANUS retargeting.
                target_action = urdf_qpos_to_retargeter_action(self.hand_qpos)
                self.hold_controller_target = _clip_v5_action_for_hand(
                    self.hand_name, target_action
                )
                command_qpos = retargeter_action_to_urdf_qpos(self.hold_controller_target)
                command_kind = "SAFE SLIDER TARGET"
            else:
                self.hold_controller_target = np.asarray(
                    [self.hand_qpos[name] for name in ALLEGRO_URDF_JOINT_NAMES],
                    dtype=np.float64,
                )
                command_qpos = dict(zip(ALLEGRO_URDF_JOINT_NAMES, self.hold_controller_target))
                command_kind = "SLIDER TARGET"
        self._show_command_target(command_qpos)
        self._set_command_status(f"HOLDING {command_kind}")
        if announce:
            print(
                "[alignment] slider command qpos="
                f"{command_qpos}; controller={self.hold_controller_target.tolist()}"
            )

    def apply_slider_target_now(self, *, announce: bool = True) -> None:
        """Set a safe slider hold target and publish it in this UI callback."""
        self.hold_slider_target(announce=announce)
        if self.hand is None or self.observe_only or self.simulate:
            return
        try:
            self._send_right_hand_command()
        except Exception as exc:
            print(f"[alignment] immediate slider command skipped: {exc}", file=sys.stderr)

    def toggle_continuous_slider_control(self) -> None:
        """Toggle direct physical following of browser slider updates."""
        if not (self.live or self.hand_control) or self.observe_only or self.simulate:
            return
        if self.slider_follow_enabled.is_set():
            self.slider_follow_enabled.clear()
            self._set_command_status("CONTINUOUS SLIDER CONTROL OFF: HOLDING LAST TARGET")
            print("[alignment] continuous slider control disabled; holding last target.")
            return
        self.slider_follow_enabled.set()
        self.apply_slider_target_now()
        self._set_command_status("CONTINUOUS SLIDER CONTROL ON")
        print("[alignment] continuous slider control enabled.")

    def _ensure_hold_target(self) -> bool:
        if self.hold_controller_target is not None:
            return True
        if self.hand is None:
            return False
        feedback = self.hand.get_data()
        if not feedback.get("is_connected", False) or feedback.get("qpos") is None:
            return False
        values, names = _normalize_feedback(
            feedback.get("qpos"), feedback.get("joint_names")
        )
        # The controller itself expects its logical joint order, which is the
        # same named order as the feedback vector returned above.
        self.hold_controller_target = values.copy()
        self._sync_preview_from_feedback(values, names)
        self._set_command_status("HOLDING CURRENT FEEDBACK")
        return True

    def _send_right_hand_command(self) -> None:
        if not (self.live or self.hand_control) or self.hand is None:
            return
        feedback = self.hand.get_data()
        if not self._show_allegro_connection(feedback):
            self._set_command_status(
                "WAITING FOR ALLEGRO FEEDBACK (OBSERVE ONLY)"
                if self.observe_only
                else "WAITING FOR ALLEGRO FEEDBACK"
            )
            return
        if self.observe_only:
            # ``get_data`` is the ROS feedback subscription.  Return before
            # resolving a controller target so this path can never call
            # ``hand.move``.
            self._set_command_status("OBSERVING ALLEGRO FEEDBACK (NO COMMANDS)")
            self.command_target_status.value = "Observation only: no command target"
            return
        values, _ = _normalize_feedback(
            feedback.get("qpos"), feedback.get("joint_names")
        )
        if not self.right_retargeting_enabled.is_set() and self.hold_controller_target is None:
            self.hold_controller_target = values.copy()
            self._set_command_status("HOLDING CURRENT FEEDBACK")
        if self.right_retargeting_enabled.is_set():
            if self.latest_manus_action is None:
                self._set_command_status("WAITING FOR VALID RIGHT MANUS")
                return
            # Match CaptureSession exactly.  The v5 data-collection path
            # calls ``allegro_v5(frame)`` then sends that 16-vector directly
            # to its controller.  Reordering it here swaps the finger blocks.
            controller_target = retargeter_action_to_live_controller_qpos(
                self.latest_manus_action, self.hand_name
            )
            self._show_command_target(
                retargeter_action_to_urdf_qpos(self.latest_manus_action)
                if self.hand_name in _V5_HAND_NAMES
                else dict(zip(ALLEGRO_URDF_JOINT_NAMES, controller_target))
            )
            self._set_command_status("MANUS → RIGHT HAND")
        else:
            if not self._ensure_hold_target():
                return
            controller_target = self.hold_controller_target
            self._show_command_target(
                retargeter_action_to_urdf_qpos(controller_target)
                if self.hand_name in _V5_HAND_NAMES
                else dict(zip(ALLEGRO_URDF_JOINT_NAMES, controller_target))
            )
        self.hand.move(np.asarray(controller_target, dtype=np.float64))

    def _read_live_pair(
        self,
    ) -> tuple[dict[str, np.ndarray], dict[str, float], np.ndarray, tuple[str, ...], Any, dict[str, float]]:
        assert self.teleop is not None and self.hand is not None
        manus = self.teleop.get_data()
        frame = manus.get("Right")
        if frame is None:
            raise RuntimeError("No fresh MANUS/VIVE right-hand frame is available.")
        feedback = self.hand.get_data()
        values, names = _normalize_feedback(feedback.get("qpos"), feedback.get("joint_names"))
        ergonomics = manus.get("ergonomics", {}).get("Right") or {}
        action = feedback.get("action")
        if action is not None:
            action = np.asarray(action, dtype=np.float64).reshape(-1)
            if action.shape != (16,) or not np.all(np.isfinite(action)):
                action = None
            else:
                action = action.tolist()
        source_times = {
            "manus_frame_time": float(manus["time"]),
            "robot_feedback_time": float(feedback["time"]),
            "capture_time": datetime.now(timezone.utc).timestamp(),
        }
        return frame, ergonomics, values, names, action, source_times

    def _target_payload(self, *, edited: bool) -> dict[str, Any]:
        with self.lock:
            qpos = dict(self.hand_qpos)
        return {
            "urdf_hand_joint_names": list(ALLEGRO_URDF_JOINT_NAMES),
            "urdf_hand_qpos_rad": qpos,
            "retargeter_action_joint_names": list(ALLEGRO_RETARGETER_JOINT_NAMES),
            "retargeter_action_rad": urdf_qpos_to_retargeter_action(qpos),
            "edited": edited,
            "updated_at": _now(),
        }

    def capture_snapshot(self) -> Path:
        if not self.live or self.output_root is None:
            raise RuntimeError("Capture is available only in live mode.")
        if self.simulate:
            raise RuntimeError("Simulation has no real Allegro feedback to capture.")
        manus_frame, ergonomics, values, names, action, source_times = self._read_live_pair()
        hand_qpos = feedback_to_urdf_qpos(values, names)
        sample_dir = _sample_name(self.output_root)
        sample_dir.mkdir(parents=True, exist_ok=False)
        full_qpos = urdf_qpos_from_hand_qpos(hand_qpos, self.urdf_joint_names)

        robot = {
            "model": "allegro",
            "feedback_encoding": "radian",
            "feedback_joint_names": list(names),
            "urdf_path": str(self.urdf_path),
            "urdf_sha256": sha256_file(self.urdf_path),
            "urdf_actuated_joint_names": list(self.urdf_joint_names),
        }
        metadata = {
            "schema_version": SCHEMA_VERSION,
            "created_at": _now(),
            "source_times": source_times,
            "capture_mode": "live_manus_allegro",
            "use_vive": self.use_vive,
            "observe_only": self.observe_only,
            "robot": robot,
            "retargeter": {
                "mode": self.retargeter_mode,
                "function": _retargeter_function_name(
                    self.hand_name, self.retargeter_mode
                ),
                "input_contract": (
                    "named MANUS 4x4 transforms in manus_frame.json; direct v5 "
                    "also consumes named ergonomic angles in manus_ergonomics.json"
                ),
                "output_joint_names": list(ALLEGRO_RETARGETER_JOINT_NAMES),
                "target_contract": "named URDF-radian joints in target_robot.json",
            },
        }
        feedback_payload = {
            "controller_joint_names": list(names),
            "controller_qpos_rad": values.tolist(),
            "controller_target_action_rad": action,
            "urdf_hand_qpos_rad": hand_qpos,
            "urdf_full_qpos_rad": full_qpos.tolist(),
        }
        target = {
            "urdf_hand_joint_names": list(ALLEGRO_URDF_JOINT_NAMES),
            "urdf_hand_qpos_rad": hand_qpos,
            "retargeter_action_joint_names": list(ALLEGRO_RETARGETER_JOINT_NAMES),
            "retargeter_action_rad": urdf_qpos_to_retargeter_action(hand_qpos),
            "edited": False,
            "updated_at": _now(),
        }
        write_json(sample_dir / "metadata.json", metadata)
        write_json(sample_dir / "manus_frame.json", serialize_manus_frame(manus_frame))
        write_json(sample_dir / "manus_ergonomics.json", dict(ergonomics))
        write_json(sample_dir / "robot_feedback.json", feedback_payload)
        write_json(sample_dir / "target_robot.json", target)
        write_json(
            sample_dir / "alignment_record.json",
            {
                "metadata": metadata,
                "manus_frame": serialize_manus_frame(manus_frame),
                "manus_ergonomics": dict(ergonomics),
                "robot_feedback": feedback_payload,
                "target_robot": target,
            },
        )
        self.source_sample = sample_dir
        self.set_hand_qpos(hand_qpos)
        self.save_target_button.disabled = False
        self.load_target_button.disabled = False
        print(f"[alignment] saved paired sample: {sample_dir}")
        return sample_dir

    def save_edited_target(self) -> None:
        if self.source_sample is None:
            raise RuntimeError("Open or capture a sample before saving an edited target.")
        target = self._target_payload(edited=True)
        write_json(self.source_sample / "target_robot.json", target)
        record = read_json(self.source_sample / "alignment_record.json")
        record["target_robot"] = target
        write_json(self.source_sample / "alignment_record.json", record)
        print(f"[alignment] updated target: {self.source_sample / 'target_robot.json'}")

    def run(self) -> None:
        if self.live or self.hand_control:
            listen_keyboard(
                {"q": self.exit_event}
                if self.simulate or not self.live
                else {"c": self.capture_event, "q": self.exit_event}
            )
            print(
                "[alignment] Browser UI is ready. Terminal keys: q=exit."
                if self.simulate or not self.live
                else "[alignment] Browser UI is ready. Terminal keys: c=save pair, q=exit."
            )
            if self.observe_only:
                print("[alignment] Observe-only is active: Allegro commands are disabled.")
        else:
            print(
                "[alignment] Reconstructed the capture-time real hand pose. "
                "Adjust sliders and click Save edited robot target."
            )
        print("[alignment] Open the Viser URL above. Press Ctrl+C to exit.")
        try:
            while not self.exit_event.is_set():
                if self.capture_event.is_set():
                    self.capture_event.clear()
                    try:
                        self.capture_snapshot()
                    except Exception as exc:
                        print(f"[alignment] capture skipped: {exc}", file=sys.stderr)
                try:
                    self._poll_manus_input()
                except Exception as exc:
                    print(f"[alignment] MANUS input skipped: {exc}", file=sys.stderr)
                try:
                    self._send_right_hand_command()
                except Exception as exc:
                    print(f"[alignment] right-hand command skipped: {exc}", file=sys.stderr)
                self.exit_event.wait(timeout=1.0 / COMMAND_RATE_HZ)
        finally:
            self.close()

    def close(self) -> None:
        stop_listening()
        if self.hand is not None:
            self.hand.end()
            self.hand = None
        if self.teleop is not None:
            self.teleop.end()
            self.teleop = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--live", action="store_true", help="capture live MANUS + real Allegro pairs")
    mode.add_argument("--open", type=Path, help="open an existing alignment sample directory")
    parser.add_argument(
        "--drive-hand",
        action="store_true",
        help=(
            "with --open, connect to the physical right Allegro hand without "
            "MANUS/VIVE; click 'Apply sliders to right hand now (hold)' to apply "
            "the loaded or edited target"
        ),
    )
    parser.add_argument(
        "--name",
        default=datetime.now().strftime("%Y%m%d_%H%M%S"),
        help="live session name below shared_data/retargeter_alignment/allegro",
    )
    parser.add_argument("--urdf", type=Path, default=DEFAULT_URDF, help="Allegro hand URDF")
    parser.add_argument(
        "--hand",
        choices=("allegro_v5", "allegro_v5_wonik", "allegro"),
        default="allegro_v5",
        help=(
            "robot driver/retargeter, matching capture_robot.py --hand; "
            "default allegro_v5 uses direct-angle retargeting plus conservative "
            "ergonomic anchor correction; "
            "allegro_v5_wonik uses Wonik's Manus ergonomics rule-based "
            "mapping; both use the same /right/allegroHand_0/joint_states driver"
        ),
    )
    parser.add_argument(
        "--retargeter",
        choices=RETARGETER_MODES,
        default="direct",
        help=(
            "UI experiment only: direct keeps the existing Allegro mapping; "
            "anyteleop enables the geometric fingertip optimizer for Allegro V5."
        ),
    )
    parser.add_argument(
        "--use-vive",
        action="store_true",
        help="reparent MANUS frames using the right VIVE tracker (default: MANUS-only)",
    )
    behavior = parser.add_mutually_exclusive_group()
    behavior.add_argument(
        "--observe-only",
        action="store_true",
        help=(
            "subscribe to and display real Allegro feedback, but never send "
            "an Allegro command; valid with --live or --open --drive-hand"
        ),
    )
    behavior.add_argument(
        "--simulate",
        action="store_true",
        help=(
            "run MANUS retargeting in the URDF viewer without constructing an "
            "Allegro driver or subscribing to an Allegro hand topic; valid with --live only"
        ),
    )
    parser.add_argument(
        "--manus-topic",
        dest="manus_topics",
        action="append",
        default=None,
        metavar="TOPIC",
        help=(
            "MANUS ROS topic to subscribe to; repeat for multiple gloves "
            f"(default: {', '.join(DEFAULT_MANUS_TOPICS)})"
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.drive_hand and not args.open:
        raise SystemExit("--drive-hand requires --open SAMPLE_DIR.")
    if args.simulate and not args.live:
        raise SystemExit("--simulate is valid with --live only.")
    if args.observe_only and not (args.live or args.drive_hand):
        raise SystemExit("--observe-only requires --live or --open --drive-hand.")
    if args.live:
        output_root = Path(shared_dir) / "retargeter_alignment" / "allegro" / args.name
        output_root.mkdir(parents=True, exist_ok=True)
        studio = AllegroRetargeterAlignmentStudio(
            urdf_path=args.urdf,
            output_root=output_root,
            live=True,
            use_vive=args.use_vive,
            observe_only=args.observe_only,
            simulate=args.simulate,
            manus_topics=tuple(args.manus_topics or DEFAULT_MANUS_TOPICS),
            hand_name=args.hand,
            retargeter_mode=args.retargeter,
        )
    else:
        sample_dir = args.open.resolve()
        loaded = _load_sample(sample_dir)
        studio = AllegroRetargeterAlignmentStudio(
            urdf_path=loaded["urdf_path"],
            source_sample=sample_dir,
            live=False,
            hand_control=args.drive_hand,
            observe_only=args.observe_only,
            hand_name=args.hand,
        )
    try:
        studio.run()
    except KeyboardInterrupt:
        # ``run`` performs its own resource cleanup in ``finally``; avoid a
        # misleading traceback for the normal Ctrl+C shutdown path.
        print("[alignment] stopped.")


if __name__ == "__main__":
    main()
