#!/usr/bin/env python3
"""Record robot sessions using the Allegro retargeter from the alignment UI.

This is an isolated experiment entry point.  It keeps the VIVE/xArm/camera
and recording flow of ``capture_robot.py``, but explicitly constructs the
right-hand retargeter through
``paradex.retargetor.experiment.allegro_retargeter_alignment_ui``.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from threading import Event
from typing import Optional

import numpy as np

from paradex.dataset_acqusition.capture import CaptureSession
from paradex.retargetor.unimanual import Retargetor
from paradex.utils.file_io import find_latest_index
from paradex.utils.keyboard_listener import listen_keyboard
from paradex.utils.path import shared_dir
from paradex.utils.system import get_pc_list
from paradex.retargetor.experiment.allegro_retargeter_alignment_ui import (
    DEFAULT_URDF,
    RETARGETER_MODES,
    _live_retargeter_kwargs,
    _make_retargeter,
)


ALLEGRO_HAND_NAME = "allegro_v5"


class AlignmentUiAllegroRetargetor(Retargetor):
    """Retarget VIVE/MANUS data through the alignment UI's exact hand path."""

    def __init__(self, *, arm_name: Optional[str], mode: str) -> None:
        super().__init__(
            arm_name=arm_name,
            hand_name=ALLEGRO_HAND_NAME,
            hand_side="Right",
            teleop_name="vive",
        )
        self.retargeter_mode = mode
        self.hand_retargetor = _make_retargeter(ALLEGRO_HAND_NAME, mode)

    def get_action(self, data):
        frame = data.get("Right")
        if frame is None:
            raise ValueError("A fresh right MANUS frame is required.")

        wrist_action = self._compute_wrist_pose("Right", data)
        ergonomics = data.get("ergonomics", {}).get("Right") or {}
        hand_action = self.hand_retargetor(
            frame,
            **_live_retargeter_kwargs(
                ALLEGRO_HAND_NAME,
                self.retargeter_mode,
                ergonomics,
            ),
        )
        return wrist_action, hand_action


def _optional_controller_name(value: Optional[str]) -> Optional[str]:
    if value is None or value.strip().lower() in ("", "none", "null"):
        return None
    return value


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--name", required=True)
    parser.add_argument("--capture_root", default="eccv2026/allegro_v5_alignment_ui")
    parser.add_argument(
        "--arm",
        default="xarm",
        help="Arm controller name; use 'none' for hand-only operation.",
    )
    parser.add_argument(
        "--retargeter",
        choices=RETARGETER_MODES,
        default="direct",
        help="Alignment UI retargeter mode (default: direct).",
    )
    vive = parser.add_mutually_exclusive_group()
    vive.add_argument("--use-vive", dest="use_vive", action="store_true")
    vive.add_argument("--no-vive", dest="use_vive", action="store_false")
    parser.set_defaults(use_vive=True)

    camera = parser.add_mutually_exclusive_group()
    camera.add_argument(
        "--camera",
        dest="camera_mode",
        nargs="?",
        const="capture",
        choices=("capture", "preview"),
        default="capture",
    )
    camera.add_argument(
        "--no-camera",
        dest="camera_mode",
        action="store_const",
        const="off",
    )
    camera.add_argument(
        "--camera-preview",
        dest="camera_mode",
        action="store_const",
        const="preview",
    )
    parser.add_argument("--camera-preview-port", type=int, default=5484)
    parser.add_argument("--camera-preview-refresh-interval", type=float, default=1.0 / 30.0)
    parser.add_argument("--camera-preview-request-timeout", type=float, default=1.5)
    parser.add_argument("--no-timestamp", dest="timestamp", action="store_false")
    parser.set_defaults(timestamp=True)
    parser.add_argument("--tactile", action="store_true")
    parser.add_argument(
        "--visualize-allegro-feedback",
        "--visualize-tactile-realtime",
        dest="visualize_allegro_feedback",
        action="store_true",
        help=(
            "Show the live right-Allegro URDF mesh from ROS2 feedback; tactile "
            "arrows are included when --tactile is enabled."
        ),
    )
    parser.add_argument("--allegro-visualization-rate-hz", type=float, default=100.0)
    parser.add_argument("--allegro-tactile-display-max", type=float, default=1000.0)
    parser.add_argument(
        "--xarm-servo-api",
        choices=("cartesian_aa", "angle_j"),
        default="cartesian_aa",
    )
    parser.add_argument("--allegro-command-rate-hz", type=float, default=30.0)
    args = parser.parse_args()

    args.arm = _optional_controller_name(args.arm)
    positive_values = {
        "--allegro-command-rate-hz": args.allegro_command_rate_hz,
        "--camera-preview-port": args.camera_preview_port,
        "--camera-preview-refresh-interval": args.camera_preview_refresh_interval,
        "--camera-preview-request-timeout": args.camera_preview_request_timeout,
        "--allegro-visualization-rate-hz": args.allegro_visualization_rate_hz,
        "--allegro-tactile-display-max": args.allegro_tactile_display_max,
    }
    for option, value in positive_values.items():
        if value <= 0:
            parser.error(f"{option} must be positive.")
    return args


def main() -> None:
    args = parse_args()
    camera_enabled = args.camera_mode != "off"
    preview_enabled = args.camera_mode == "preview"
    camera_pc_list = get_pc_list()

    save_event = Event()
    stop_event = Event()
    exit_event = Event()
    events = {"save": save_event, "stop": stop_event, "exit": exit_event}
    listen_keyboard({"c": save_event, "s": stop_event, "q": exit_event})
    print("Keyboard control: c=start capture, s=stop capture, q=exit")
    print(
        "Allegro retargeter: alignment UI "
        f"({args.retargeter}); command rate={args.allegro_command_rate_hz:g} Hz"
    )

    session = CaptureSession(
        camera=camera_enabled,
        realsense=False,
        arm=args.arm,
        hand=ALLEGRO_HAND_NAME,
        teleop="vive",
        hand_side="right",
        events=events,
        tactile=args.tactile,
        timestamp=args.timestamp,
        camera_pc_list=camera_pc_list,
        arm_kwargs={"servo_api": args.xarm_servo_api} if args.arm == "xarm" else None,
        hand_command_rate_hz=args.allegro_command_rate_hz,
        use_vive=args.use_vive,
        require_left_control=args.use_vive,
    )
    # CaptureSession still owns devices, recording, command-rate limiting and
    # safety conversion.  Only its action generator is replaced.
    session.retargetor = AlignmentUiAllegroRetargetor(
        arm_name=args.arm,
        mode=args.retargeter,
    )

    allegro_visualizer = None
    if args.visualize_allegro_feedback:
        from paradex.visualization.allegro_realtime import AllegroRealtimeViser

        allegro_visualizer = AllegroRealtimeViser(
            session.hand,
            update_rate_hz=args.allegro_visualization_rate_hz,
            tactile_display_max=args.allegro_tactile_display_max,
            urdf_path=str(DEFAULT_URDF),
            render_feedback_pose=True,
        )
        allegro_visualizer.start()

    preview = None
    if preview_enabled:
        from paradex.io.camera_system.capture_pc_preview import CapturePcPreviewGui

        preview = CapturePcPreviewGui(
            pc_list=camera_pc_list,
            port=args.camera_preview_port,
            refresh_interval=args.camera_preview_refresh_interval,
            request_timeout=args.camera_preview_request_timeout,
            side_panel_provider=getattr(allegro_visualizer, "render_bgr", None),
        )
        preview.start()

    def refresh_preview(_session=None):
        if preview is not None:
            preview.show()

    capture_root = Path(shared_dir) / "capture" / args.capture_root / args.name
    last_idx = int(find_latest_index(str(capture_root)))

    try:
        while not exit_event.is_set():
            state = session.teleop(
                session_events=events,
                state_policy="keyboard_control",
                loop_callback=refresh_preview,
            )
            if state == "exit":
                break
            if state != "start":
                continue

            last_idx += 1
            episode_rel_path = os.path.join(
                "capture", args.capture_root, args.name, str(last_idx)
            )
            episode_abs_path = Path(shared_dir) / episode_rel_path
            stop_event.clear()
            session.start(episode_rel_path)
            print(f"Starting recording: {args.name}, episode {last_idx}")

            state = session.teleop(
                session_events=events,
                state_policy="keyboard_control",
                loop_callback=refresh_preview,
            )
            session.stop()
            print(f"Stopped recording: {args.name}, episode {last_idx}")

            timestamp_path = episode_abs_path / "raw" / "timestamps" / "timestamp.npy"
            if timestamp_path.exists():
                print(f"timestamp.npy length: {len(np.load(timestamp_path))}")
            else:
                print(f"timestamp.npy not found at {timestamp_path}")
            save_event.clear()
            stop_event.clear()
            if state == "exit":
                break
    finally:
        print("Exiting Allegro alignment-retargeter capture.")
        if preview is not None:
            preview.close()
        if allegro_visualizer is not None:
            allegro_visualizer.close()
        if session.save_path is not None:
            session.stop()
        session.end()


if __name__ == "__main__":
    main()
