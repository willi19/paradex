"""Capture extrinsic-calibration frames and the matching xArm pose.

This is a variant of ``src/calibration/extrinsic/capture.py`` with the same
unimanual ``CaptureSession`` teleoperation flow as ``capture_robot.py``.
Press ``c`` in the OpenCV window to save the camera calibration frame and the
xArm's current joint and wrist poses.  Press ``q`` to stop.

The xArm files use the established calibration format:
``<index>_qpos.npy`` contains the first six measured joint angles (radians),
and ``<index>_aa.npy`` contains the 4x4 measured wrist transform.
"""

import argparse
import os
import sys
import time
from pathlib import Path
from threading import Event, Lock, Thread
from typing import Any

import cv2
import numpy as np

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from paradex.calibration.utils import extrinsic_dir
from paradex.image.aruco import draw_charuco
from paradex.image.merge import merge_image
from paradex.io.capture_pc.command_sender import CommandSender
from paradex.io.capture_pc.data_sender import DataCollector
from paradex.io.capture_pc.ssh import run_script
from paradex.utils.system import get_pc_list


EXCLUDED_PCS = {}
DEFAULT_XARM_SAVE_DIR = REPOSITORY_ROOT / "system/current/hecalib/xarm/new"
BOARD_COLORS = [(0, 0, 255), (0, 255, 0)]


def next_pose_index(save_dir: Path) -> int:
    """Return the next unused numeric pose index in ``save_dir``."""
    indices = []
    for path in save_dir.glob("*_qpos.npy"):
        try:
            indices.append(int(path.name[: -len("_qpos.npy")]))
        except ValueError:
            continue
    return max(indices, default=-1) + 1


def save_xarm_pose(arm: Any, save_dir: Path, index: int) -> None:
    """Save measured xArm feedback from the active CaptureSession controller."""
    data = arm.get_data()
    qpos = np.asarray(data.get("qpos"), dtype=np.float64)
    wrist_transform = np.asarray(data.get("position"), dtype=np.float64)
    if qpos.shape != (6,) or not np.all(np.isfinite(qpos)):
        raise RuntimeError("xArm joint feedback is not available yet.")
    if wrist_transform.shape != (4, 4) or not np.all(np.isfinite(wrist_transform)):
        raise RuntimeError("xArm wrist feedback is not available yet.")
    np.save(save_dir / f"{index}_qpos.npy", qpos)
    np.save(save_dir / f"{index}_aa.npy", wrist_transform)
    print(f"Saved xArm pose {index}: {save_dir / f'{index}_qpos.npy'}")
    print(f"Saved xArm wrist transform {index}: {save_dir / f'{index}_aa.npy'}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=["xsens", "vive"], default="vive")
    parser.add_argument("--hand-side", choices=["right", "left"], default="right")
    parser.add_argument(
        "--hand",
        default="inspire_f1",
        help=(
            "Hand controller/retargetor name, matching capture_robot.py. "
            "Use 'none' for arm-only control."
        ),
    )
    parser.add_argument(
        "--xarm-servo-api",
        choices=["cartesian_aa", "angle_j"],
        default="cartesian_aa",
    )
    parser.add_argument("--hand-scale", type=float, default=1.15)
    parser.add_argument("--allegro-command-rate-hz", type=float, default=30.0)
    vive_group = parser.add_mutually_exclusive_group()
    vive_group.add_argument("--use-vive", dest="use_vive", action="store_true")
    vive_group.add_argument("--no-vive", dest="use_vive", action="store_false")
    parser.set_defaults(use_vive=True)
    parser.add_argument(
        "--xarm-save-dir",
        type=Path,
        default=DEFAULT_XARM_SAVE_DIR,
        help=f"Directory for xArm pose files (default: {DEFAULT_XARM_SAVE_DIR})",
    )
    args = parser.parse_args()
    if args.hand.strip().lower() in ("", "none", "null"):
        args.hand = None
    if args.device == "vive" and args.use_vive and args.hand_side != "right":
        parser.error("VIVE unimanual teleoperation currently requires --hand-side right.")
    if args.allegro_command_rate_hz <= 0.0:
        parser.error("--allegro-command-rate-hz must be positive.")

    # Import lazily so ``--help`` remains usable outside the robot ROS environment.
    from paradex.dataset_acqusition.capture import CaptureSession

    xarm_save_dir = args.xarm_save_dir
    xarm_save_dir.mkdir(parents=True, exist_ok=True)
    pose_index = next_pose_index(xarm_save_dir)

    pc_list = [pc for pc in get_pc_list() if pc not in EXCLUDED_PCS]
    filename = time.strftime("%Y%m%d_%H%M%S", time.localtime())
    os.makedirs(os.path.join(extrinsic_dir, filename), exist_ok=True)

    run_script("python src/calibration/extrinsic/client.py", pc_list=pc_list, log=True)
    collector = DataCollector(pc_list=pc_list)
    collector.start()
    commands = CommandSender(pc_list=pc_list)
    save_event = Event()
    stop_event = Event()
    exit_event = Event()
    session_events = {
        "save": save_event,
        "stop": stop_event,
        "exit": exit_event,
    }
    session = CaptureSession(
        camera=False,
        realsense=False,
        arm="xarm",
        hand=args.hand,
        teleop=args.device,
        hand_side=args.hand_side,
        events=session_events,
        tactile=False,
        ip=False,
        hand_kwargs=None,
        timestamp=False,
        camera_pc_list=pc_list,
        arm_kwargs={"servo_api": args.xarm_servo_api},
        hand_scale=args.hand_scale,
        hand_command_rate_hz=(
            args.allegro_command_rate_hz
            if args.hand == "allegro_v5"
            else None
        ),
        allegro_teleop_diagnostic_path=None,
        use_vive=args.use_vive,
        require_left_control=args.use_vive,
    )

    saved_corner_img = {}
    saved_corner_mask = {}
    cur_state = {}
    img_dict = {}
    img_text = {}
    save_num = 0
    camera_state_lock = Lock()
    latest_image_lock = Lock()
    camera_errors = []
    latest_merged_image = [np.full((600, 800, 3), 255, dtype=np.uint8)]

    def update_camera_stream() -> None:
        """Decode and merge camera streams away from the teleoperation loop."""
        try:
            while not exit_event.is_set():
                waiting_save = False
                all_data = collector.get_data()
                with camera_state_lock:
                    for item_name, item_data in all_data.items():
                        if item_data.get("type") == "image":
                            image_bytes = item_data.get("data")
                            frame_id = item_data.get("frame_id", 0)
                            if item_data.get("save_id", 0) < save_num:
                                waiting_save = True
                            if image_bytes:
                                image = cv2.imdecode(
                                    np.frombuffer(image_bytes, np.uint8),
                                    cv2.IMREAD_COLOR,
                                )
                                if image is not None:
                                    img_dict[item_name] = image
                                    img_text[item_name] = str(frame_id)
                        elif item_data.get("type") == "charuco_detection":
                            corners = np.frombuffer(
                                item_data.get("data"), dtype=np.float32
                            ).reshape(-1, 2)
                            serial_num = item_name.split("_")[0]
                            if serial_num not in saved_corner_img:
                                saved_corner_img[serial_num] = np.zeros(
                                    (1536 // 8, 2048 // 8, 3), dtype=np.uint8
                                )
                                saved_corner_mask[serial_num] = np.zeros(
                                    (0, 2), dtype=np.int32
                                )
                            cur_state[serial_num] = (
                                corners,
                                item_data.get("frame_id", 0),
                            )

                    if img_dict:
                        display_dict = {}
                        for serial_num, image in img_dict.items():
                            display_image = image.copy()
                            if serial_num in cur_state:
                                mask = saved_corner_mask[serial_num]
                                display_image[mask[:, 1], mask[:, 0]] = BOARD_COLORS[0]
                                corners, _ = cur_state[serial_num]
                                if corners.size:
                                    draw_charuco(
                                        display_image,
                                        corners,
                                        BOARD_COLORS[1],
                                        1,
                                        -1,
                                    )
                            display_dict[serial_num] = display_image
                        merged_image = merge_image(display_dict, img_text)
                        if waiting_save:
                            cv2.putText(
                                merged_image,
                                "Saving...",
                                (50, 50),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                1,
                                (0, 0, 255),
                                2,
                            )
                    else:
                        merged_image = np.full(
                            (600, 800, 3), 255, dtype=np.uint8
                        )
                        cv2.putText(
                            merged_image,
                            "Waiting for stream...",
                            (50, 300),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            1,
                            (0, 0, 0),
                            2,
                        )
                    with latest_image_lock:
                        latest_merged_image[0] = merged_image
                time.sleep(0.001)
        except BaseException as error:
            camera_errors.append(error)
            exit_event.set()

    def refresh_guis(_session: Any = None) -> None:
        """Match capture_robot.py's lightweight teleop loop callback."""
        with latest_image_lock:
            merged_image = latest_merged_image[0].copy()
        cv2.imshow("Merged Stream", merged_image)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("c"):
            save_event.set()
        elif key == ord("q"):
            exit_event.set()

    camera_thread = Thread(
        target=update_camera_stream,
        name="extrinsic-camera-ui",
        daemon=True,
    )
    camera_thread.start()

    if args.device == "xsens":
        teleop_source = "XSens"
    elif args.use_vive:
        teleop_source = "VIVE + MANUS"
    else:
        teleop_source = "MANUS only"
    print(f"{teleop_source} unimanual teleoperation uses capture_robot.py's main loop.")
    print("JPEG decoding runs separately from teleoperation.")
    print("Press 'c' in the Merged Stream window to capture; press 'q' to quit.")
    try:
        while not exit_event.is_set():
            state = session.teleop(
                session_events=session_events,
                state_policy="keyboard_control",
                loop_callback=refresh_guis,
            )
            if state == "exit":
                break
            if state != "start":
                continue

            try:
                save_xarm_pose(session.arm, xarm_save_dir, pose_index)
            except RuntimeError as error:
                print(f"xArm pose was not saved; camera capture skipped: {error}")
                save_event.clear()
                continue

            capture_idx = time.strftime("%Y%m%d_%H%M%S", time.localtime())
            capture_dir = os.path.join(extrinsic_dir, filename, capture_idx)
            suffix = 1
            while os.path.exists(capture_dir):
                capture_dir = os.path.join(
                    extrinsic_dir, filename, f"{capture_idx}_{suffix:02d}"
                )
                suffix += 1
            os.makedirs(os.path.join(capture_dir, "markers_2d"))
            os.makedirs(os.path.join(capture_dir, "images"))
            commands.send_command("save", True)
            save_num += 1
            pose_index += 1

            with camera_state_lock:
                for serial_num, (corners, _) in cur_state.items():
                    if corners.size:
                        draw_charuco(
                            saved_corner_img[serial_num],
                            corners,
                            BOARD_COLORS[1],
                            1,
                            -1,
                        )
                        ys, xs, _ = np.where(saved_corner_img[serial_num] != 0)
                        saved_corner_mask[serial_num] = np.stack([xs, ys], axis=1)
            save_event.clear()
    finally:
        print("Stopping capture...")
        exit_event.set()
        camera_thread.join(timeout=3.0)
        session.end()
        collector.end()
        commands.end()
        cv2.destroyAllWindows()
        print("Stream stopped.")

    if camera_errors:
        raise camera_errors[0]


if __name__ == "__main__":
    main()
