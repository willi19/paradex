#!/usr/bin/env python3
"""Capture, inspect, and simulate paired MANUS -> Wuji hand poses.

Live mode records a MANUS right/left-hand frame and the matching measured Wuji
20-joint state.  Open mode reconstructs that recorded state from the recorded
URDF, then allows its target pose to be edited and saved.  ``--simulate``
requires only MANUS: it neither creates a Wuji controller nor opens a Wuji ROS
topic, and previews the retargeter output directly on the URDF mesh.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import threading
from datetime import datetime, timezone
from functools import partial
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from paradex.io.robot_controller import get_hand
from paradex.io.teleop.vive.receiver import ViveManusROSReceiver
from paradex.retargetor.hand_alignment_common import (
    read_json,
    serialize_manus_frame,
    sha256_file,
    write_json,
)
from paradex.retargetor.hand_regargetor import wuji, wuji_direct, wuji_hybrid
from paradex.utils.keyboard_listener import listen_keyboard, stop_listening
from paradex.utils.path import shared_dir
from paradex.visualization.visualizer.viser import ViserViewer


DOF = 20
COMMAND_RATE_HZ = 30.0
DEFAULT_MANUS_TOPICS = ("/manus_glove_0", "/manus_glove_1")
RETARGETERS = {"wuji": wuji, "wuji_direct": wuji_direct, "wuji_hybrid": wuji_hybrid}
REPO_ROOT = Path(__file__).resolve().parents[3]
_VENDORED_URDF_ROOT = (
    REPO_ROOT
    / "thirdparty"
    / "wuji-retargeting"
    / "wuji_retargeting"
    / "wuji-description"
    / "hand"
    / "body"
    / "urdf"
)


def _default_urdf_root() -> Path:
    """Prefer a complete Wuji description package over the code-only vendor copy."""
    candidates = (
        Path(os.environ["WUJI_DESCRIPTION_PATH"]) / "hand" / "body" / "urdf"
        if "WUJI_DESCRIPTION_PATH" in os.environ
        else None,
        Path.home()
        / "wuji_ws"
        / "src"
        / "wuji-hand-teleop"
        / "src"
        / "wuji-retargeting"
        / "wuji_retargeting"
        / "wuji-description"
        / "hand"
        / "body"
        / "urdf",
        _VENDORED_URDF_ROOT,
    )
    for root in candidates:
        if root is None:
            continue
        mesh = root.parent / "meshes" / "right" / "right_palm_link.STL"
        if (root / "right.urdf").is_file() and mesh.is_file():
            return root
    return _VENDORED_URDF_ROOT


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _wuji_joint_names(side: str) -> tuple[str, ...]:
    return tuple(
        f"{side}_finger{finger}_joint{joint}"
        for finger in range(1, 6)
        for joint in range(1, 5)
    )


def _as_qpos(values: Any) -> np.ndarray:
    qpos = np.asarray(values, dtype=np.float64).reshape(-1)
    if qpos.shape != (DOF,) or not np.all(np.isfinite(qpos)):
        raise ValueError(f"Wuji qpos must be {DOF} finite values, got {qpos.shape}")
    return qpos


def _make_retargeter(name: str, side: str, scale: float):
    return partial(RETARGETERS[name], is_right=side == "right", scale=scale)


def _alignment_samples(root: Path) -> list[Path]:
    """Return only numbered capture directories directly under one session."""
    return sorted(
        (path for path in root.iterdir() if path.is_dir() and path.name.isdigit()),
        key=lambda path: int(path.name),
    )


def _delete_alignment_sample(sample_path: Path, root: Path) -> None:
    """Delete exactly the selected capture, never a parent/session directory."""
    sample_path = sample_path.resolve()
    root = root.resolve()
    if sample_path.parent != root or not sample_path.name.isdigit():
        raise ValueError(f"Refusing to delete a capture outside {root}: {sample_path}")
    shutil.rmtree(sample_path)


def _resolve_target_source_sample(source_sample: Path, target_from: str | None) -> Path | None:
    """Resolve a numbered target source within the opened capture session."""
    if target_from is None:
        return None
    if not target_from.isdigit():
        raise ValueError("--target-from must be a numbered capture, for example 000010")
    target_sample = (source_sample.resolve().parent / target_from).resolve()
    if target_sample.parent != source_sample.resolve().parent or not target_sample.is_dir():
        raise ValueError(f"Target source capture does not exist: {target_sample}")
    return target_sample


class WujiRetargeterAlignmentStudio:
    def __init__(
        self,
        *,
        urdf_path: Path,
        side: str,
        retargeter_name: str,
        hand_scale: float,
        output_root: Path | None = None,
        source_sample: Path | None = None,
        target_source_sample: Path | None = None,
        live: bool = False,
        use_vive: bool = False,
        observe_only: bool = False,
        simulate: bool = False,
        manus_topics: tuple[str, ...] = DEFAULT_MANUS_TOPICS,
    ):
        if observe_only and simulate:
            raise ValueError("--observe-only and --simulate cannot be combined")
        self.urdf_path = urdf_path.resolve()
        self.side = side
        self.retargeter_name = retargeter_name
        self.retargeter = _make_retargeter(retargeter_name, side, hand_scale)
        self.hand_scale = hand_scale
        self.output_root = output_root
        self.source_sample = source_sample.resolve() if source_sample else None
        self.sample_root = self.source_sample.parent if self.source_sample else None
        self.target_source_sample = (
            target_source_sample.resolve() if target_source_sample else None
        )
        self.live = live
        self.use_vive = use_vive
        self.observe_only = observe_only
        self.simulate = simulate
        self.manus_topics = tuple(manus_topics)
        self.lock = threading.Lock()
        self.capture_event = threading.Event()
        self.exit_event = threading.Event()
        self.command_live = False
        self.slider_target_active = False
        self.hold_target: np.ndarray | None = None
        self.latest_manus_frame: dict[str, np.ndarray] | None = None
        self.latest_action: np.ndarray | None = None
        self.actual_qpos: np.ndarray | None = None
        self.teleop = None
        self.hand = None

        self.viewer = ViserViewer(scene_title="MANUS → Wuji alignment", show_player=False)
        self.viewer.add_robot("wuji", str(self.urdf_path), include_arm_meshes=True)
        self.viser_robot = self.viewer.robot_dict["wuji"]
        self.robot = self.viser_robot.urdf
        self.joint_names = tuple(self.robot.get_joint_names())
        expected_names = _wuji_joint_names(side)
        if self.joint_names != expected_names:
            raise ValueError(
                "Wuji URDF joint order does not match the controller order: "
                f"expected {expected_names}, got {self.joint_names}"
            )
        self.joint_limits = self.robot.get_joint_limits()
        self.qpos = np.asarray(
            [np.clip(0.0, *self.joint_limits[name]) for name in self.joint_names],
            dtype=np.float64,
        )

        if self.source_sample is not None:
            loaded = self._load_sample(self.source_sample)
            self.urdf_path = loaded["urdf_path"]
            self.qpos = _as_qpos(loaded["feedback"]["controller_qpos_rad"])
            if self.target_source_sample is not None:
                self.qpos = self._target_qpos_from_sample(self.target_source_sample)

        self._build_gui()
        self._set_qpos(self.qpos)
        if self.source_sample is not None:
            self._set_command_status(
                f"EDITING FROM {self.target_source_sample.name} TARGET"
                if self.target_source_sample is not None
                else "CAPTURE-TIME FEEDBACK POSE"
            )
        if self.live:
            self.teleop = ViveManusROSReceiver(
                hand_side=self.side,
                require_left_control=False,
                use_vive=self.use_vive,
                manus_topics=self.manus_topics,
            )
            if self.simulate:
                self._set_command_status("SIMULATING MANUS → WUJI URDF (NO DRIVER)")
            else:
                self.hand = get_hand(
                    self.retargeter_name,
                    hand_side=self.side,
                    command_enabled=not self.observe_only,
                )
                self._set_command_status(
                    "OBSERVING WUJI FEEDBACK (NO COMMANDS)"
                    if self.observe_only
                    else "WAITING FOR WUJI FEEDBACK"
                )

    def _build_gui(self) -> None:
        gui = self.viewer.server.gui
        with gui.add_folder("Capture controls" if self.live else "Saved pose"):
            self.command_status = gui.add_text("Wuji command state", initial_value="STARTING", disabled=True)
            self.capture_button = None
            self.pause_button = None
            self.resume_button = None
            self.apply_slider_button = None
            if self.live:
                self.capture_button = gui.add_button("Capture paired state", disabled=self.simulate)
                self.pause_button = gui.add_button("Pause Wuji hand (hold pose)", disabled=self.observe_only or self.simulate)
                self.resume_button = gui.add_button("Resume MANUS → Wuji hand", disabled=self.observe_only or self.simulate)
                self.apply_slider_button = gui.add_button("Move Wuji hand to sliders (hold)", disabled=self.observe_only or self.simulate)
            self.save_target_button = gui.add_button("Save edited robot target", disabled=self.source_sample is None)
            self.load_target_button = gui.add_button("Load saved edited target", disabled=self.source_sample is None)

        self.target_source_selector = None
        self.load_target_source_button = None
        if not self.live and self.sample_root is not None:
            options = tuple(path.name for path in _alignment_samples(self.sample_root))
            initial_target = (
                self.target_source_sample.name
                if self.target_source_sample is not None
                else self.source_sample.name
            )
            with gui.add_folder("Start editing from another target"):
                self.target_source_selector = gui.add_dropdown(
                    "Target source capture",
                    options=options,
                    initial_value=initial_target,
                )
                self.load_target_source_button = gui.add_button("Use selected target pose")

        self.previous_sample_button = None
        self.next_sample_button = None
        self.delete_sample_button = None
        self.sample_position = None
        if not self.live:
            with gui.add_folder("Capture browser"):
                self.sample_position = gui.add_text("Current capture", initial_value="", disabled=True)
                self.previous_sample_button = gui.add_button("Previous capture")
                self.next_sample_button = gui.add_button("Next capture")
                self.delete_sample_button = gui.add_button("Delete current capture")

        if self.live:
            with gui.add_folder("Live input / feedback"):
                self.wuji_status = gui.add_text("Wuji feedback", initial_value="NOT STARTED", disabled=True)
                self.actual_status = gui.add_text(
                    "Simulated Wuji qpos" if self.simulate else "Actual Wuji qpos",
                    initial_value="WAITING",
                    disabled=True,
                )
                self.manus_status = gui.add_text("MANUS state", initial_value="WAITING", disabled=True)
                self.manus_action = gui.add_text("Retargeter action [rad]", initial_value="", disabled=True)

        self.sliders = {}
        with gui.add_folder("Wuji hand target (URDF radians)"):
            for index, name in enumerate(self.joint_names):
                lower, upper = self.joint_limits[name]
                slider = gui.add_slider(
                    name,
                    min=float(lower),
                    max=float(upper),
                    step=0.001,
                    initial_value=float(self.qpos[index]),
                    disabled=self.observe_only or self.simulate,
                )
                self.sliders[name] = slider

                @slider.on_update
                def _(_event, joint_index=index, handle=slider):
                    if getattr(self, "_syncing_sliders", False):
                        return
                    self.qpos[joint_index] = float(handle.value)
                    self.slider_target_active = True
                    self._update_robot()

        if self.live:
            @self.capture_button.on_click
            def _(_event):
                self.capture_event.set()

            @self.pause_button.on_click
            def _(_event):
                self.pause()

            @self.resume_button.on_click
            def _(_event):
                self.resume()

            @self.apply_slider_button.on_click
            def _(_event):
                self.hold_sliders()

        @self.save_target_button.on_click
        def _(_event):
            self.save_edited_target()

        @self.load_target_button.on_click
        def _(_event):
            self.load_saved_target()

        if self.load_target_source_button is not None:
            @self.load_target_source_button.on_click
            def _(_event):
                self.load_selected_target_source()

        if not self.live:
            @self.previous_sample_button.on_click
            def _(_event):
                self._select_relative_sample(-1)

            @self.next_sample_button.on_click
            def _(_event):
                self._select_relative_sample(1)

            @self.delete_sample_button.on_click
            def _(_event):
                self.delete_current_sample()

            self._refresh_sample_browser()

    def _set_command_status(self, value: str) -> None:
        self.command_status.value = value

    def _update_robot(self) -> None:
        self.viser_robot.update_cfg(self.qpos)

    def _set_qpos(self, values: Any) -> None:
        self.qpos = _as_qpos(values)
        self._syncing_sliders = True
        try:
            for value, name in zip(self.qpos, self.joint_names):
                lower, upper = self.joint_limits[name]
                self.sliders[name].value = float(np.clip(value, lower, upper))
        finally:
            self._syncing_sliders = False
        self._update_robot()

    def _load_sample(self, sample_dir: Path) -> dict[str, Any]:
        metadata = read_json(sample_dir / "metadata.json")
        if metadata.get("schema_version") != 1 or metadata.get("robot", {}).get("model") != "wuji":
            raise ValueError(f"Unsupported Wuji alignment sample: {sample_dir}")
        urdf_path = Path(metadata["robot"]["urdf_path"])
        if not urdf_path.exists() or sha256_file(urdf_path) != metadata["robot"]["urdf_sha256"]:
            raise ValueError("Capture-time Wuji URDF is missing or has changed")
        return {"metadata": metadata, "feedback": read_json(sample_dir / "robot_feedback.json"), "target": read_json(sample_dir / "target_robot.json"), "urdf_path": urdf_path}

    def _target_qpos_from_sample(self, target_sample: Path) -> np.ndarray:
        """Read a target pose only when it belongs to this exact Wuji model."""
        loaded = self._load_sample(target_sample)
        if loaded["urdf_path"].resolve() != self.urdf_path:
            raise ValueError("Target source uses a different URDF")
        target = loaded["target"]
        if tuple(target.get("urdf_joint_names", ())) != self.joint_names:
            raise ValueError("Target source joint order does not match the opened Wuji hand")
        return _as_qpos(target["urdf_qpos_rad"])

    def _refresh_sample_browser(self) -> list[Path]:
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

    def _load_selected_sample(self, sample_path: Path) -> None:
        loaded = self._load_sample(sample_path)
        if loaded["urdf_path"].resolve() != self.urdf_path:
            raise ValueError("Selected capture uses a different URDF; open it in a new UI session.")
        self.source_sample = sample_path.resolve()
        self.qpos = _as_qpos(loaded["feedback"]["controller_qpos_rad"])
        self._set_qpos(self.qpos)
        self.save_target_button.disabled = False
        self.load_target_button.disabled = False
        self._set_command_status("CAPTURE-TIME FEEDBACK POSE")
        self._refresh_sample_browser()
        print(f"[wuji-alignment] loaded capture: {self.source_sample}")

    def _select_relative_sample(self, offset: int) -> None:
        samples = self._refresh_sample_browser()
        if self.source_sample not in samples:
            return
        index = samples.index(self.source_sample) + offset
        if 0 <= index < len(samples):
            self._load_selected_sample(samples[index])

    def delete_current_sample(self) -> None:
        if self.live or self.source_sample is None or self.sample_root is None:
            return
        samples = self._refresh_sample_browser()
        if self.source_sample not in samples:
            return
        current_index = samples.index(self.source_sample)
        deleted = self.source_sample
        _delete_alignment_sample(deleted, self.sample_root)
        print(f"[wuji-alignment] deleted capture: {deleted}")
        remaining = _alignment_samples(self.sample_root)
        if remaining:
            self._load_selected_sample(remaining[min(current_index, len(remaining) - 1)])
            return
        self.source_sample = None
        self.save_target_button.disabled = True
        self.load_target_button.disabled = True
        self._set_command_status("NO CAPTURES REMAIN")
        self._refresh_sample_browser()

    def _poll_manus(self) -> None:
        if self.teleop is None:
            return
        data = self.teleop.get_data()
        frame = data.get("Right" if self.side == "right" else "Left")
        self.latest_manus_frame = None
        self.latest_action = None
        if frame is None:
            self.manus_status.value = f"WAITING FOR {self.side.upper()} MANUS"
            return
        action = self.retargeter(frame)
        if action is None:
            self.manus_status.value = "MANUS FRAME INVALID FOR WUJI RETARGETER"
            return
        self.latest_manus_frame = frame
        self.latest_action = _as_qpos(action)
        self.manus_status.value = "RECEIVING" + (" + VIVE" if self.use_vive else " (MANUS-only)")
        self.manus_action.value = np.array2string(self.latest_action, precision=3, separator=",")
        if self.simulate:
            self.actual_status.value = self.manus_action.value
            self.wuji_status.value = "SIMULATION: MANUS retargeter → URDF (NO WUJI CONNECTION)"
            self._set_qpos(self.latest_action)

    def _update_hand(self) -> None:
        if self.hand is None:
            return
        feedback = self.hand.get_data()
        if not feedback.get("is_connected", False):
            self.wuji_status.value = f"WAITING: no feedback on {feedback.get('state_topic', '/joint_states')}"
            return
        actual = _as_qpos(feedback["qpos"])
        self.actual_qpos = actual
        self.wuji_status.value = f"CONNECTED: {DOF} Wuji joints on {feedback.get('state_topic', '/joint_states')}"
        self.actual_status.value = np.array2string(actual, precision=3, separator=",")
        if not self.slider_target_active:
            self._set_qpos(actual)
        if self.observe_only:
            self._set_command_status("OBSERVING WUJI FEEDBACK (NO COMMANDS)")
            return
        if self.command_live:
            if self.latest_action is None:
                self._set_command_status("WAITING FOR VALID MANUS")
                return
            target = self.latest_action
            self._set_command_status("MANUS → WUJI HAND")
        else:
            if self.hold_target is None:
                self.hold_target = actual.copy()
            target = self.hold_target
            self._set_command_status("HOLDING CURRENT FEEDBACK")
        self.hand.move(target)

    def pause(self) -> None:
        if self.observe_only or self.simulate:
            return
        self.command_live = False
        self.slider_target_active = False
        self.hold_target = None if self.actual_qpos is None else self.actual_qpos.copy()

    def resume(self) -> None:
        if not self.observe_only and not self.simulate:
            self.command_live = True
            self.slider_target_active = False
            self._set_command_status("MANUS → WUJI HAND")

    def hold_sliders(self) -> None:
        if self.observe_only or self.simulate:
            return
        self.command_live = False
        self.slider_target_active = True
        self.hold_target = self.qpos.copy()
        self._set_command_status("HOLDING SLIDER TARGET")

    def capture_snapshot(self) -> Path:
        if self.simulate or self.hand is None or self.output_root is None:
            raise RuntimeError("Paired capture requires live measured Wuji feedback")
        if self.latest_manus_frame is None:
            raise RuntimeError("No valid MANUS frame available")
        feedback = self.hand.get_data()
        if not feedback.get("is_connected", False):
            raise RuntimeError("No valid Wuji feedback available")
        qpos = _as_qpos(feedback["qpos"])
        sample_dir = self.output_root / f"{max([-1, *[int(p.name) for p in self.output_root.iterdir() if p.is_dir() and p.name.isdigit()]]) + 1:06d}"
        sample_dir.mkdir(parents=True, exist_ok=False)
        metadata = {
            "schema_version": 1,
            "created_at": _now(),
            "capture_mode": "live_manus_wuji",
            "use_vive": self.use_vive,
            "robot": {"model": "wuji", "side": self.side, "urdf_path": str(self.urdf_path), "urdf_sha256": sha256_file(self.urdf_path), "controller_joint_names": list(self.joint_names)},
            "retargeter": {"name": self.retargeter_name, "hand_scale": self.hand_scale, "input_contract": "named MANUS 4x4 transforms in manus_frame.json", "output_joint_names": list(self.joint_names)},
        }
        target = {"urdf_joint_names": list(self.joint_names), "urdf_qpos_rad": qpos.tolist(), "edited": False, "updated_at": _now()}
        feedback_payload = {"controller_joint_names": list(self.joint_names), "controller_qpos_rad": qpos.tolist(), "controller_target_action_rad": _as_qpos(feedback["action"]).tolist()}
        write_json(sample_dir / "metadata.json", metadata)
        write_json(sample_dir / "manus_frame.json", serialize_manus_frame(self.latest_manus_frame))
        write_json(sample_dir / "robot_feedback.json", feedback_payload)
        write_json(sample_dir / "target_robot.json", target)
        self.source_sample = sample_dir
        self.save_target_button.disabled = False
        self.load_target_button.disabled = False
        print(f"[wuji-alignment] saved paired sample: {sample_dir}")
        return sample_dir

    def save_edited_target(self) -> None:
        if self.source_sample is None:
            return
        write_json(self.source_sample / "target_robot.json", {"urdf_joint_names": list(self.joint_names), "urdf_qpos_rad": self.qpos.tolist(), "edited": True, "updated_at": _now()})
        print(f"[wuji-alignment] updated target: {self.source_sample / 'target_robot.json'}")

    def load_saved_target(self) -> None:
        if self.source_sample is None:
            return
        target = read_json(self.source_sample / "target_robot.json")
        self._set_qpos(target["urdf_qpos_rad"])

    def load_selected_target_source(self) -> None:
        """Start editing the opened capture from another capture's target pose."""
        if self.target_source_selector is None or self.sample_root is None:
            return
        target_sample = (self.sample_root / str(self.target_source_selector.value)).resolve()
        self._set_qpos(self._target_qpos_from_sample(target_sample))
        self.target_source_sample = target_sample
        self._set_command_status(f"EDITING FROM {target_sample.name} TARGET")
        print(f"[wuji-alignment] editing {self.source_sample.name} from target {target_sample}")

    def run(self) -> None:
        if self.live:
            listen_keyboard({"q": self.exit_event} if self.simulate else {"c": self.capture_event, "q": self.exit_event})
            print("[wuji-alignment] Browser UI is ready. Terminal keys: q=exit." if self.simulate else "[wuji-alignment] Browser UI is ready. Terminal keys: c=save pair, q=exit.")
        try:
            while not self.exit_event.is_set():
                if self.capture_event.is_set():
                    self.capture_event.clear()
                    try:
                        self.capture_snapshot()
                    except Exception as exc:
                        print(f"[wuji-alignment] capture skipped: {exc}", file=sys.stderr)
                try:
                    self._poll_manus()
                    self._update_hand()
                except Exception as exc:
                    print(f"[wuji-alignment] live update skipped: {exc}", file=sys.stderr)
                self.exit_event.wait(timeout=1.0 / COMMAND_RATE_HZ)
        finally:
            self.close()

    def close(self) -> None:
        stop_listening()
        if self.hand is not None:
            self.hand.end()
        if self.teleop is not None:
            self.teleop.end()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--live", action="store_true")
    mode.add_argument("--open", type=Path)
    parser.add_argument("--name", default=datetime.now().strftime("%Y%m%d_%H%M%S"))
    parser.add_argument("--hand-side", choices=("right", "left"), default="right")
    parser.add_argument(
        "--urdf",
        type=Path,
        help=(
            "Wuji left/right URDF; defaults to a complete local Wuji description "
            "package when available"
        ),
    )
    parser.add_argument("--retargeter", choices=tuple(RETARGETERS), default="wuji_direct")
    parser.add_argument(
        "--target-from",
        help="Open mode: start editing from this numbered capture's saved target, e.g. 000010",
    )
    parser.add_argument("--scale", type=float, default=1.15)
    parser.add_argument("--use-vive", action="store_true")
    behavior = parser.add_mutually_exclusive_group()
    behavior.add_argument("--observe-only", action="store_true")
    behavior.add_argument("--simulate", action="store_true")
    parser.add_argument("--manus-topic", dest="manus_topics", action="append")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if (args.observe_only or args.simulate) and not args.live:
        raise SystemExit("--observe-only and --simulate are valid with --live only")
    if args.target_from and args.live:
        raise SystemExit("--target-from is valid with --open only")
    urdf_path = args.urdf or _default_urdf_root() / f"{args.hand_side}.urdf"
    if args.live:
        output_root = Path(shared_dir) / "retargeter_alignment" / "wuji" / args.name
        output_root.mkdir(parents=True, exist_ok=True)
        studio = WujiRetargeterAlignmentStudio(
            urdf_path=urdf_path,
            side=args.hand_side,
            retargeter_name=args.retargeter,
            hand_scale=args.scale,
            output_root=output_root,
            live=True,
            use_vive=args.use_vive,
            observe_only=args.observe_only,
            simulate=args.simulate,
            manus_topics=tuple(args.manus_topics or DEFAULT_MANUS_TOPICS),
        )
    else:
        target_source_sample = _resolve_target_source_sample(args.open, args.target_from)
        metadata = read_json(args.open / "metadata.json")
        robot = metadata["robot"]
        retargeter = metadata["retargeter"]
        studio = WujiRetargeterAlignmentStudio(
            urdf_path=Path(robot["urdf_path"]),
            side=robot["side"],
            retargeter_name=retargeter["name"],
            hand_scale=float(retargeter["hand_scale"]),
            source_sample=args.open,
            target_source_sample=target_source_sample,
        )
    try:
        studio.run()
    except KeyboardInterrupt:
        print("[wuji-alignment] stopped.")


if __name__ == "__main__":
    main()
