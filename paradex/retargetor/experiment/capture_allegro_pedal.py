#!/usr/bin/env python3
"""Capture VIVE arm teleoperation with an Allegro hand driven by outer pedals.

The VIVE wrist stream drives the xArm exactly through ``CaptureSession``.
MANUS is deliberately not subscribed to.  The left pedal moves the Allegro
hand from endpoint A toward B; the right pedal moves it back toward A.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from threading import Event

import numpy as np

from paradex.dataset_acqusition.capture import CaptureSession
from paradex.utils.file_io import find_latest_index
from paradex.utils.keyboard_listener import listen_keyboard
from paradex.utils.path import shared_dir
from paradex.retargetor.experiment.allegro_pedal_pose_ui import (
    DEFAULT_POSE_A,
    DEFAULT_POSE_B,
    DEFAULT_SOURCE_POSE_A,
    DEFAULT_SOURCE_POSE_B,
    AllegroPedalPoseStudio,
    ensure_editable_pose_copy,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--name", required=True)
    parser.add_argument("--capture-root", default="eccv2026/allegro_v5_pedal")
    parser.add_argument("--arm", default="xarm")
    parser.add_argument("--camera", dest="camera", action="store_true", default=True)
    parser.add_argument("--no-camera", dest="camera", action="store_false")
    parser.add_argument("--timestamp", dest="timestamp", action="store_true", default=True)
    parser.add_argument("--no-timestamp", dest="timestamp", action="store_false")
    parser.add_argument("--xarm-servo-api", choices=("cartesian_aa", "angle_j"), default="cartesian_aa")
    parser.add_argument("--pose-a", type=Path, default=DEFAULT_POSE_A)
    parser.add_argument("--pose-b", type=Path, default=DEFAULT_POSE_B)
    parser.add_argument("--source-pose-a", type=Path, default=DEFAULT_SOURCE_POSE_A)
    parser.add_argument("--source-pose-b", type=Path, default=DEFAULT_SOURCE_POSE_B)
    parser.add_argument("--pedal-rate", type=float, default=0.35)
    parser.add_argument("--no-pedal", action="store_true", help="Use the UI parameter slider instead of Stream Deck Pedal.")
    parser.add_argument(
        "--tactile-threshold", type=float, default=200.0,
        help="per-finger raw tactile maximum at which a closing finger latches (default: 200)",
    )
    parser.add_argument(
        "--no-tactile-contact-stop", action="store_true",
        help="disable tactile finger holds and interpolate all fingers normally",
    )
    return parser.parse_args()


def wait_for_grasp_result(exit_event: Event, yes_event: Event, no_event: Event) -> str:
    yes_event.clear()
    no_event.clear()
    print("Grasp success? Press y or n, then Enter.")
    while not exit_event.is_set():
        if yes_event.is_set():
            return "y"
        if no_event.is_set():
            return "n"
        time.sleep(0.01)
    return "n"


def main() -> None:
    args = parse_args()
    stop_event = Event()
    save_event = Event()
    exit_event = Event()
    grasp_yes_event = Event()
    grasp_no_event = Event()
    events = {"save": save_event, "stop": stop_event, "exit": exit_event}
    listen_keyboard(
        {
            "c": save_event,
            "s": stop_event,
            "q": exit_event,
            "y": grasp_yes_event,
            "n": grasp_no_event,
        }
    )
    print("Keyboard: c=start capture, s=stop capture, q=exit, y/n=grasp result")

    pose_a = ensure_editable_pose_copy(args.pose_a, args.source_pose_a)
    pose_b = ensure_editable_pose_copy(args.pose_b, args.source_pose_b)
    studio = AllegroPedalPoseStudio(
        pose_a_sample=pose_a,
        pose_b_sample=pose_b,
        simulate=True,
        observe_only=False,
        use_pedal=not args.no_pedal,
        pedal_rate=args.pedal_rate,
        tactile_contact_stop=not args.no_tactile_contact_stop,
        tactile_threshold=args.tactile_threshold,
        external_hand=True,
    )
    studio._set_status("WAITING: ALLEGRO + VIVE FEEDBACK")

    session: CaptureSession | None = None
    try:
        session = CaptureSession(
            camera=args.camera,
            realsense=False,
            arm=args.arm,
            hand="allegro_v5",
            # Capture tactile continuously for the UI even when its stopping
            # rule is temporarily disabled.
            tactile=True,
            teleop="vive",
            hand_side="right",
            events=events,
            timestamp=args.timestamp,
            arm_kwargs={"servo_api": args.xarm_servo_api} if args.arm == "xarm" else None,
            use_vive=True,
            use_manus=False,
            require_left_control=False,
            hand_action_provider=lambda: studio.command_action(session.hand.get_data()),
        )

        last_idx = int(find_latest_index(os.path.join(shared_dir, "capture", args.capture_root, args.name)))
        success_count = 0
        fail_count = 0
        while not exit_event.is_set():
            state = session.teleop(session_events=events, state_policy="keyboard_control")
            if state == "exit":
                break
            if state != "start":
                continue

            last_idx += 1
            episode_rel_path = os.path.join("capture", args.capture_root, args.name, str(last_idx))
            episode_abs_path = os.path.join(shared_dir, episode_rel_path)
            print(f"Starting episode {last_idx}: {episode_rel_path}")
            session.start(episode_rel_path)
            session.teleop(session_events=events, state_policy="keyboard_control")
            session.stop()
            save_event.clear()
            stop_event.clear()
            timestamp_path = os.path.join(
                episode_abs_path, "raw", "timestamps", "timestamp.npy"
            )
            if os.path.isfile(timestamp_path):
                try:
                    print(f"timestamp.npy length: {len(np.load(timestamp_path))}")
                except Exception as exc:
                    print(f"Could not inspect timestamp.npy: {exc}")

            result = wait_for_grasp_result(exit_event, grasp_yes_event, grasp_no_event)
            success_count += result == "y"
            fail_count += result != "y"
            os.makedirs(episode_abs_path, exist_ok=True)
            with open(os.path.join(episode_abs_path, "grasp_result.json"), "w", encoding="utf-8") as stream:
                json.dump({"episode": last_idx, "grasp_success": result == "y"}, stream, indent=2)
            while not exit_event.is_set():
                try:
                    paired_episode = int(input("Paired human episode number: "))
                    break
                except ValueError:
                    print("Enter an integer episode number.")
            else:
                break
            with open(
                os.path.join(episode_abs_path, "paired_human_episode.json"),
                "w",
                encoding="utf-8",
            ) as stream:
                json.dump(
                    {
                        "human hand episode": last_idx,
                        "paired human episode": paired_episode,
                    },
                    stream,
                    indent=2,
                )
            print(f"Success count: {success_count} / Failure count: {fail_count}")
    finally:
        if session is not None:
            session.end()
        studio.close()


if __name__ == "__main__":
    main()
