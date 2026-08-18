#!/usr/bin/env python3
"""Capture and edit paired MANUS -> Inspire F1 retargeter-alignment states.

Live mode subscribes to the MANUS/VIVE receiver and the real F1 hand only;
it never creates, connects to, or reads an xArm controller. Press
``c`` followed by Enter in the launching terminal (the same interaction as
``capture_robot.py``), or click **Capture paired state** in the browser UI, to
write one self-contained alignment sample.  Open a saved sample later with
``--open`` to reproduce its robot-hand pose from the recorded URDF + qpos and
edit the target pose using six URDF-radian sliders.

The UI starts by holding the current physical right-hand pose.  **Resume**
switches the right hand to live MANUS retargeting; **Pause** captures the
latest feedback as a hold target and continues commanding that pose.  Slider
editing remains preview-only.
"""

from __future__ import annotations

import argparse
import sys
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from paradex.io.robot_controller import get_hand
from paradex.io.teleop.vive.receiver import ViveManusROSReceiver
from paradex.retargetor.inspire_f1_alignment import (
    F1_RIGHT_WIRE_JOINT_NAMES,
    HandStateContractError,
    master_qpos_to_raw_f1,
    raw_f1_to_master_qpos,
    read_json,
    retargeter_action_to_raw_f1,
    serialize_manus_frame,
    sha256_file,
    urdf_qpos_from_master_qpos,
    write_json,
)
from paradex.retargetor.hand_regargetor import inspire_f1
from paradex.utils.keyboard_listener import listen_keyboard, stop_listening
from paradex.utils.path import rsc_path, shared_dir
from paradex.visualization.visualizer.viser import ViserViewer


SCHEMA_VERSION = 1
DEFAULT_URDF = Path(rsc_path) / "robot" / "xarm_inspire_f1_right.urdf"
COMMAND_RATE_HZ = 30.0


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sample_name(root: Path) -> Path:
    existing = [int(path.name) for path in root.iterdir() if path.is_dir() and path.name.isdigit()]
    return root / f"{(max(existing, default=-1) + 1):06d}"


def _normalize_feedback(
    values: Any, names: Any,
) -> tuple[np.ndarray, tuple[str, ...]]:
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    if names is None or len(names) == 0:
        names = F1_RIGHT_WIRE_JOINT_NAMES
    names = tuple(str(name) for name in names)
    if values.shape != (len(names),):
        raise HandStateContractError(
            f"F1 feedback has {values.shape[0]} values for {len(names)} names"
        )
    # This UI implements the F1 contract documented by capture_object6d:
    # /right/joint_states publishes the six raw wire motor values.  Refuse a
    # radian-looking signal rather than save a plausible but wrong GLB pose.
    if values.size and float(np.max(np.abs(values))) < 100.0:
        raise HandStateContractError(
            "F1 feedback appears to be radians, but this UI expects the raw "
            "0..1800 wire contract. Verify the ROS driver before capturing."
        )
    raw_f1_to_master_qpos(values, names)  # validates names, length, and finiteness
    return values, names


def _load_sample(sample_dir: Path) -> dict[str, Any]:
    metadata = read_json(sample_dir / "metadata.json")
    if int(metadata.get("schema_version", -1)) != SCHEMA_VERSION:
        raise ValueError(f"Unsupported alignment sample schema: {metadata.get('schema_version')}")
    urdf_path = Path(metadata["robot"]["urdf_path"])
    if not urdf_path.exists():
        raise FileNotFoundError(f"Saved URDF no longer exists: {urdf_path}")
    expected_hash = metadata["robot"]["urdf_sha256"]
    actual_hash = sha256_file(urdf_path)
    if actual_hash != expected_hash:
        raise ValueError(
            "Saved URDF bytes differ from the capture-time URDF; refusing to "
            "claim an exact reconstruction. Restore the recorded URDF version."
        )
    target = read_json(sample_dir / "target_robot.json")
    feedback = read_json(sample_dir / "robot_feedback.json")
    return {
        "metadata": metadata,
        "target": target,
        "feedback": feedback,
        "urdf_path": urdf_path,
    }


class RetargeterAlignmentStudio:
    def __init__(
        self,
        *,
        urdf_path: Path,
        output_root: Path | None = None,
        source_sample: Path | None = None,
        live: bool = False,
    ):
        self.urdf_path = urdf_path.resolve()
        self.output_root = output_root
        self.source_sample = source_sample
        self.live = live
        self.lock = threading.Lock()
        self._syncing_sliders = False
        self.capture_event = threading.Event()
        self.exit_event = threading.Event()
        self.right_retargeting_enabled = threading.Event()
        self.hold_raw_target: dict[str, float] | None = None
        self.teleop = None
        self.hand = None

        self.viewer = ViserViewer(scene_title="MANUS → Inspire F1 alignment")
        self.viewer.add_robot(
            "inspire_f1",
            str(self.urdf_path),
            include_arm_meshes=False,
        )
        self.robot = self.viewer.robot_dict["inspire_f1"].urdf
        self.urdf_joint_names = tuple(self.robot.get_joint_names())
        self.joint_limits = self.robot.get_joint_limits()
        self.master_qpos = {name: 0.0 for name in F1_RIGHT_WIRE_JOINT_NAMES}

        if source_sample is not None:
            loaded = _load_sample(source_sample)
            self.urdf_path = loaded["urdf_path"]
            captured_master = loaded["feedback"]["master_qpos_rad"]
            # Always start from the recorded real hand state.  A separately
            # edited target is available through the GUI, but must not replace
            # the evidence needed to reproduce the capture.
            self.master_qpos = {
                name: float(captured_master[name])
                for name in F1_RIGHT_WIRE_JOINT_NAMES
            }

        self._build_gui()
        self._update_robot()

        if live:
            self.teleop = ViveManusROSReceiver(
                hand_side="right", require_left_control=False
            )
            self.hand = get_hand("inspire_f1", hand_side="right")
            print("[alignment] Right-hand command mode: HOLD current pose.")

    def _build_gui(self) -> None:
        gui = self.viewer.server.gui
        with gui.add_folder("Retargeter alignment"):
            self.capture_button = gui.add_button("Capture paired state", disabled=not self.live)
            self.pause_button = gui.add_button("Pause right hand (hold pose)", disabled=not self.live)
            self.resume_button = gui.add_button("Resume MANUS → right hand", disabled=not self.live)
            self.save_target_button = gui.add_button("Save edited robot target", disabled=self.source_sample is None)
            self.load_target_button = gui.add_button("Load saved target pose", disabled=self.source_sample is None)

        self.sliders = {}
        with gui.add_folder("Target robot hand (URDF radians)"):
            for name in F1_RIGHT_WIRE_JOINT_NAMES:
                lower, upper = self.joint_limits[name]
                slider = gui.add_slider(
                    name,
                    min=float(lower),
                    max=float(upper),
                    step=0.001,
                    initial_value=float(self.master_qpos[name]),
                )
                self.sliders[name] = slider

                @slider.on_update
                def _(_event, joint_name=name, handle=slider):
                    if self._syncing_sliders:
                        return
                    with self.lock:
                        self.master_qpos[joint_name] = float(handle.value)
                    self._update_robot()

        @self.capture_button.on_click
        def _(_event):
            self.capture_event.set()

        @self.pause_button.on_click
        def _(_event):
            self.pause_right_hand()

        @self.resume_button.on_click
        def _(_event):
            self.resume_right_hand()

        @self.save_target_button.on_click
        def _(_event):
            try:
                self.save_edited_target()
            except Exception as exc:
                print(f"[alignment] unable to save target: {exc}", file=sys.stderr)

        @self.load_target_button.on_click
        def _(_event):
            if self.source_sample is None:
                return
            target = read_json(self.source_sample / "target_robot.json")
            self.set_master_qpos(target["master_qpos_rad"])

    def _update_robot(self) -> None:
        with self.lock:
            qpos = urdf_qpos_from_master_qpos(self.master_qpos, self.urdf_joint_names)
        self.robot.update_cfg(qpos)

    def set_master_qpos(self, values: Mapping[str, float]) -> None:
        with self.lock:
            for name in F1_RIGHT_WIRE_JOINT_NAMES:
                self.master_qpos[name] = float(values[name])
        self._syncing_sliders = True
        try:
            for name in F1_RIGHT_WIRE_JOINT_NAMES:
                self.sliders[name].value = float(values[name])
        finally:
            self._syncing_sliders = False
        self._update_robot()

    def pause_right_hand(self) -> None:
        """Freeze the current physical feedback as the persistent hold target."""
        self.right_retargeting_enabled.clear()
        self.hold_raw_target = None
        try:
            self._ensure_hold_target()
        except Exception as exc:
            print(f"[alignment] waiting for F1 feedback before holding: {exc}", file=sys.stderr)
        print("[alignment] Right-hand commands paused: holding current feedback pose.")

    def resume_right_hand(self) -> None:
        if not self.live:
            return
        self.right_retargeting_enabled.set()
        print("[alignment] Right-hand commands resumed from live MANUS retargeting.")

    def _ensure_hold_target(self) -> bool:
        if self.hold_raw_target is not None:
            return True
        if self.hand is None:
            return False
        feedback = self.hand.get_data()
        if feedback.get("qpos") is None:
            return False
        raw_values, state_names = _normalize_feedback(
            feedback.get("qpos"), feedback.get("joint_names")
        )
        raw_by_name = dict(zip(state_names, raw_values.tolist()))
        self.hold_raw_target = {
            name: float(raw_by_name[name]) for name in F1_RIGHT_WIRE_JOINT_NAMES
        }
        return True

    def _send_right_hand_command(self) -> None:
        if not self.live or self.hand is None:
            return
        if self.right_retargeting_enabled.is_set():
            assert self.teleop is not None
            manus = self.teleop.get_data()
            frame = manus.get("Right")
            if frame is None:
                return
            action = inspire_f1(frame, is_right=True)
            if action is None:
                return
            raw_target = retargeter_action_to_raw_f1(action)
        else:
            if not self._ensure_hold_target():
                return
            raw_target = self.hold_raw_target
        self.hand.publish_raw_wire_command(
            [raw_target[name] for name in F1_RIGHT_WIRE_JOINT_NAMES],
            F1_RIGHT_WIRE_JOINT_NAMES,
            side="right",
        )

    def _read_live_pair(
        self,
    ) -> tuple[dict[str, np.ndarray], dict[str, float], np.ndarray, tuple[str, ...], dict[str, float]]:
        assert self.teleop is not None and self.hand is not None
        manus = self.teleop.get_data()
        frame = manus.get("Right")
        if frame is None:
            raise RuntimeError("No fresh MANUS/VIVE right-hand frame is available.")
        ergonomics = manus.get("ergonomics", {}).get("Right") or {}
        feedback = self.hand.get_data()
        raw_values, state_names = _normalize_feedback(
            feedback.get("qpos"), feedback.get("joint_names")
        )
        source_times = {
            "manus_frame_time": float(manus["time"]),
            "robot_feedback_time": float(feedback["time"]),
            "capture_time": datetime.now(timezone.utc).timestamp(),
        }
        return frame, ergonomics, raw_values, state_names, source_times

    def capture_snapshot(self) -> Path:
        if not self.live or self.output_root is None:
            raise RuntimeError("Capture is available only in live mode.")
        manus_frame, ergonomics, raw_values, state_names, source_times = self._read_live_pair()
        master_qpos = raw_f1_to_master_qpos(raw_values, state_names)
        sample_dir = _sample_name(self.output_root)
        sample_dir.mkdir(parents=True, exist_ok=False)

        qpos = urdf_qpos_from_master_qpos(master_qpos, self.urdf_joint_names)
        robot_info = {
            "model": "inspire_f1_right",
            "feedback_encoding": "inspire_f1_raw_motor_count",
            "feedback_joint_names": list(state_names),
            "urdf_path": str(self.urdf_path),
            "urdf_sha256": sha256_file(self.urdf_path),
            "urdf_actuated_joint_names": list(self.urdf_joint_names),
        }
        target = {
            "master_joint_names": list(F1_RIGHT_WIRE_JOINT_NAMES),
            "master_qpos_rad": master_qpos,
            "raw_motor_targets": master_qpos_to_raw_f1(master_qpos),
            "edited": False,
            "updated_at": _now(),
        }
        metadata = {
            "schema_version": SCHEMA_VERSION,
            "created_at": _now(),
            "source_times": source_times,
            "capture_mode": "live_manus_inspire_f1",
            "robot": robot_info,
            "retargeter": {
                "function": "paradex.retargetor.hand_regargetor.inspire_f1",
                "input_contract": "named MANUS 4x4 transforms in manus_frame.json",
                "target_contract": "named URDF-radian master joints in target_robot.json",
            },
        }

        write_json(sample_dir / "metadata.json", metadata)
        write_json(sample_dir / "manus_frame.json", serialize_manus_frame(manus_frame))
        write_json(sample_dir / "manus_ergonomics.json", dict(ergonomics))
        write_json(
            sample_dir / "robot_feedback.json",
            {
                "joint_names": list(state_names),
                "raw_motor_values": raw_values.tolist(),
                "master_qpos_rad": master_qpos,
                "urdf_qpos_rad": qpos.tolist(),
            },
        )
        write_json(sample_dir / "target_robot.json", target)
        # One flattened JSON record makes model-assisted retargeter fitting
        # possible without reconstructing a pair from several opaque npy files.
        write_json(
            sample_dir / "alignment_record.json",
            {
                "metadata": metadata,
                "manus_frame": serialize_manus_frame(manus_frame),
                "manus_ergonomics": dict(ergonomics),
                "robot_feedback": read_json(sample_dir / "robot_feedback.json"),
                "target_robot": target,
            },
        )
        self.source_sample = sample_dir
        self.set_master_qpos(master_qpos)
        self.save_target_button.disabled = False
        self.load_target_button.disabled = False
        print(f"[alignment] saved paired sample: {sample_dir}")
        return sample_dir

    def save_edited_target(self) -> None:
        if self.source_sample is None:
            raise RuntimeError("Open or capture a sample before saving an edited target.")
        with self.lock:
            masters = dict(self.master_qpos)
        target = {
            "master_joint_names": list(F1_RIGHT_WIRE_JOINT_NAMES),
            "master_qpos_rad": masters,
            "raw_motor_targets": master_qpos_to_raw_f1(masters),
            "edited": True,
            "updated_at": _now(),
        }
        write_json(self.source_sample / "target_robot.json", target)
        record_path = self.source_sample / "alignment_record.json"
        record = read_json(record_path)
        record["target_robot"] = target
        write_json(record_path, record)
        print(f"[alignment] updated target: {self.source_sample / 'target_robot.json'}")

    def run(self) -> None:
        if self.live:
            listen_keyboard({"c": self.capture_event, "q": self.exit_event})
            print("[alignment] Browser UI is ready. Terminal keys: c=save pair, q=exit.")
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
    mode.add_argument("--live", action="store_true", help="capture live MANUS + real F1 pairs")
    mode.add_argument("--open", type=Path, help="open an existing alignment sample directory")
    parser.add_argument(
        "--name",
        default=datetime.now().strftime("%Y%m%d_%H%M%S"),
        help="live session name below shared_data/retargeter_alignment",
    )
    parser.add_argument("--urdf", type=Path, default=DEFAULT_URDF, help="combined xArm + right F1 URDF")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.live:
        output_root = Path(shared_dir) / "retargeter_alignment" / args.name
        output_root.mkdir(parents=True, exist_ok=True)
        studio = RetargeterAlignmentStudio(
            urdf_path=args.urdf,
            output_root=output_root,
            live=True,
        )
    else:
        sample_dir = args.open.resolve()
        loaded = _load_sample(sample_dir)
        studio = RetargeterAlignmentStudio(
            urdf_path=loaded["urdf_path"],
            source_sample=sample_dir,
            live=False,
        )
    studio.run()


if __name__ == "__main__":
    main()
