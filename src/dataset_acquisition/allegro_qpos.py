import argparse
import os
import time
from threading import Event
from typing import List

import numpy as np

from paradex.dataset_acqusition.capture_image import CaptureSession
from paradex.utils.file_io import find_latest_index
from paradex.utils.keyboard_listener import listen_keyboard
from paradex.utils.path import shared_dir

from paradex.calibration.utils import save_current_camparam, save_current_C2R
from paradex.io.camera_system.remote_camera_controller import remote_camera_controller


def build_session_dir(name: str) -> str:
    """
    Create a new session directory under shared_data.

    Layout: <shared_dir>/capture/allegro_qpos/<name>/<index>
    """
    base_dir = os.path.join(shared_dir, "capture", "allegro_qpos", name)
    last_idx = int(find_latest_index(base_dir))
    new_idx = last_idx + 1
    session_dir = os.path.join(base_dir, str(new_idx))
    os.makedirs(session_dir, exist_ok=True)
    return session_dir


def save_snapshots(session_dir: str, timestamps: List[float], xarm_qpos: List[np.ndarray],
                   allegro_qpos: List[np.ndarray]) -> None:
    """Persist captured data to disk."""
    if len(timestamps) == 0:
        print("No snapshots were captured; nothing saved.")
        return

    xarm_arr = np.stack(xarm_qpos, axis=0)
    allegro_arr = np.stack(allegro_qpos, axis=0)
    ts_arr = np.array(timestamps)
    robot_arr = np.concatenate([xarm_arr, allegro_arr], axis=1)

    np.save(os.path.join(session_dir, "timestamp.npy"), ts_arr)
    np.save(os.path.join(session_dir, "xarm_qpos.npy"), xarm_arr)
    np.save(os.path.join(session_dir, "allegro_qpos.npy"), allegro_arr)
    np.save(os.path.join(session_dir, "robot_qpos.npy"), robot_arr)

    print(f"Saved {len(ts_arr)} snapshots to {session_dir}")


def main():
    parser = argparse.ArgumentParser(description="Capture Allegro + XArm qpos on key press.")
    parser.add_argument("--name", required=True, help="Dataset/session name (subdirectory).")
    parser.add_argument("--arm", default="xarm", help="Arm controller name (default: xarm).")
    parser.add_argument("--hand", default="allegro", help="Hand controller name (default: allegro).")
    parser.add_argument("--hand-driver", choices=["ros1", "ros2"], default="ros1",
                        help="Choose Allegro driver implementation: ros2 uses allegro_controller_temp, ros1 uses allegro_controller.")
    parser.add_argument("--tactile", action="store_true", help="Enable tactile hand if supported.")
    parser.add_argument("--ip", action="store_true", help="Use IP connection for hand when applicable.")
    parser.add_argument("--teleop", action="store_true", help="Enable Xsens teleoperation while capturing.")
    args = parser.parse_args()

    capture_event = Event()
    exit_event = Event()

    listen_keyboard({"c": capture_event, "q": exit_event})
    print("Press 'c' to save a snapshot, 'q' to quit.")

    # session_dir = build_session_dir(args.name)
    # print(f"Session directory: {session_dir}")

    teleop_mode = "xsens" if args.teleop else None

    # Use CaptureSession for arm/teleop, but swap in ROS2 allegro controller if requested.
    cs = CaptureSession(arm=args.arm, hand=None if args.hand_driver == "ros2" else args.hand,
                        tactile=args.tactile, ip=args.ip, teleop=teleop_mode)
    xarm = cs.arm

    if args.hand_driver == "ros2":
        from paradex.io.robot_controller.allegro_controller import AllegroController as AllegroControllerROS2
        allegro = AllegroControllerROS2()
    else:
        allegro = cs.hand
    rcc = remote_camera_controller("image_main.py")


    count = int(find_latest_index(os.path.join(shared_dir, "allegro_qpos", args.name))) + 1
    # print("ccccccccccccccc": count)

    try:
        # Wait for Allegro to receive its first joint state (when supported)
        if hasattr(allegro, "connection_event"):
            if not allegro.connection_event.wait(timeout=5.0):
                print("Warning: Allegro joint state not received within 5 seconds. Continuing anyway.")

        while not exit_event.is_set():
            if not capture_event.wait(timeout=0.01):
                continue
            # capture_event.clear()
            
            os.makedirs(os.path.join(shared_dir, "allegro_qpos", args.name, str(count)))
            # os.makedirs(os.path.join(shared_dir, args.name, str(count)))
            
            timestamps: List[float] = []
            xarm_qpos: List[np.ndarray] = []
            allegro_qpos: List[np.ndarray] = []

            xarm_data = xarm.get_data()
            allegro_data = allegro.get_data()
            timestamp = time.time()

            xarm_qpos.append(np.asarray(xarm_data["qpos"], dtype=float))
            allegro_qpos.append(np.asarray(allegro_data["qpos"], dtype=float))
            timestamps.append(timestamp)
            
            save_current_C2R(os.path.join(shared_dir, "allegro_qpos", args.name, str(count)))
            save_current_camparam(os.path.join(shared_dir, "allegro_qpos", args.name, str(count)))

            np.save(os.path.join(shared_dir, "allegro_qpos", args.name, str(count), f"xarm_qpos.npy"), xarm_qpos[0])
            np.save(os.path.join(shared_dir, "allegro_qpos", args.name, str(count), f"allegro_qpos.npy"), allegro_qpos[0])
            
            print("Saved allegro qpos")
            
            # np.save(os.path.join(session_dir, f".npy"), allegro_qpos[-1])


            rcc.start("image", False, f'shared_data/allegro_qpos/{args.name}/{count}/raw')
            rcc.stop()
            # save_snapshots(session_dir, timestamps, xarm_qpos, allegro_qpos)
            
            capture_event.clear()

            count += 1
            print(f"[{count}] snapshot saved at t={timestamp:.3f}")

    finally:
        # save_snapshots(session_dir, timestamps, xarm_qpos, allegro_qpos)
        rcc.end()


if __name__ == "__main__":
    main()
