import argparse
import subprocess
import sys
import time

import numpy as np

from paradex.io.robot_controller.inspire_f1_controller_ros2 import InspireF1ControllerROS2


def wait_for_topics(topics, timeout_s=20.0):
    start = time.time()
    while time.time() - start < timeout_s:
        try:
            out = subprocess.check_output(["ros2", "topic", "list"], text=True)
        except Exception:
            time.sleep(0.5)
            continue
        found = set(out.split())
        if all(t in found for t in topics):
            return True
        time.sleep(0.5)
    return False


def run_control(duration_s, hand_side):
    ctrl = InspireF1ControllerROS2(hand_side=hand_side)
    start_time = time.time()
    try:
        while time.time() - start_time < duration_s:
            phase = (time.time() - start_time) * 2 * np.pi * 0.2
            s = np.sin(phase)
            ranges = np.array(
                [
                    [900, 1740],
                    [900, 1740],
                    [900, 1740],
                    [900, 1740],
                    [1100, 1350],
                    [600, 1800],
                ],
                dtype=np.float64,
            )
            mid = ranges.mean(axis=1)
            amp = (ranges[:, 1] - ranges[:, 0]) / 2.0
            target = mid + amp * s
            ctrl.move(target.astype(np.float64))
            time.sleep(0.01)
    finally:
        ctrl.end()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--hand-side", default="left", choices=["left", "right"])
    parser.add_argument("--master-id", default="0")
    parser.add_argument("--duration", type=float, default=10.0)
    parser.add_argument("--bringup-only", action="store_true")
    parser.add_argument("--ros-ws", default="/home/temp_id/inspire-hand-ros2")
    args = parser.parse_args()

    setup_script = f"{args.ros_ws}/install/setup.bash"
    launch_cmd = (
        f"source {setup_script} && "
        f"ros2 launch inspire_hand_hardware rh56f1_ethercat_bringup.launch.py "
        f"hand_side:={args.hand_side} master_id:={args.master_id}"
    )

    print("[INFO] launching EtherCAT bringup...")
    launch_proc = subprocess.Popen(["bash", "-lc", launch_cmd])
    try:
        topics_ok = wait_for_topics(
            [
                "/joint_states",
                "/dynamic_joint_states",
                "/tactile_sensor_states",
                "/position_controller/commands",
            ],
            timeout_s=30.0,
        )
        if not topics_ok:
            print("[WARN] expected topics not found; continuing anyway.")

        if args.bringup_only:
            print("[INFO] bringup-only mode; press Ctrl+C to stop.")
            while True:
                time.sleep(1.0)
        else:
            run_control(args.duration, args.hand_side)
    finally:
        print("[INFO] shutting down bringup...")
        launch_proc.terminate()
        try:
            launch_proc.wait(timeout=5.0)
        except subprocess.TimeoutExpired:
            launch_proc.kill()


if __name__ == "__main__":
    main()
