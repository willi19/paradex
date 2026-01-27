import argparse
import os
import time
from typing import Tuple

import numpy as np

from paradex.io.robot_controller import get_arm, get_hand

from paradex.dataset_acqusition.capture import CaptureSession
from paradex.utils.path import shared_dir
from paradex.utils.file_io import find_latest_index

from paradex.io.camera_system.timestamp_monitor import TimestampMonitor
from paradex.utils.system import network_info

def load_series(data_dir: str, candidates: Tuple[str, ...]) -> Tuple[np.ndarray, np.ndarray]:
    for name in candidates:
        path = os.path.join(data_dir, name)
        if os.path.exists(path):
            data = np.load(path)
            time_path = os.path.join(data_dir, "time.npy")
            if os.path.exists(time_path):
                t = np.load(time_path)
            else:
                t = np.arange(data.shape[0], dtype=float)
            if len(t) != data.shape[0]:
                n = min(len(t), data.shape[0])
                data = data[:n]
                t = t[:n]
            return data, t
    raise FileNotFoundError(f"No data found in {data_dir} for {candidates}")


def resample_to(times_src: np.ndarray, data_src: np.ndarray, times_dst: np.ndarray) -> np.ndarray:
    if data_src.shape[0] == times_dst.shape[0] and np.allclose(times_src, times_dst):
        return data_src
    order = np.argsort(times_src)
    times_src = times_src[order]
    data_src = data_src[order]
    out = np.zeros((times_dst.shape[0], data_src.shape[1]), dtype=float)
    for j in range(data_src.shape[1]):
        out[:, j] = np.interp(times_dst, times_src, data_src[:, j])
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--arm", type=str, default="xarm")
    parser.add_argument("--hand", type=str, default="inspire")
    parser.add_argument("--object", type=str, default = "smallbowl")
    parser.add_argument("--ep", type=int, default = "1", help="Episode number.")
    parser.add_argument("--teleop-device", choices=["xsens", "occulus"], default=None)
    parser.add_argument("--rate-scale", type=float, default=1.0, help=">1.0 to play faster, <1.0 to slow down.")
    parser.add_argument(
        "--go-to-first-position",
        action="store_true",
        help="Move to the first pose and exit without replay.",
    )
    parser.add_argument("--output-dir", type=str, default=None)
    args = parser.parse_args()

    capture_root = os.path.join("/home/temp_id/shared_data/capture/hri_inspire_left", args.object, str(args.ep))
    data_root = os.path.join(capture_root, "raw")
    arm_dir = os.path.join(data_root, "arm")
    hand_dir = os.path.join(data_root, "hand")

    arm_qpos, arm_time = load_series(arm_dir, ("position.npy", "action_qpos.npy", "action.npy"))
    hand_qpos, hand_time = load_series(hand_dir, ("position.npy", "action.npy"))

    # arm_qpos, arm_time = load_series(arm_dir, ("action_qpos.npy", "position.npy", "action.npy"))
    # hand_qpos, hand_time = load_series(hand_dir, ("action.npy", "position.npy"))

    # Need to revisit
    hand_qpos = resample_to(hand_time, hand_qpos, arm_time)


    arm_ctrl = get_arm(args.arm)

    
    print(arm_qpos)
    # i = i+1

    if args.go_to_first_position:
        hand_ctrl = get_hand(args.hand)
        print("Moving to first pose only...")
        # print("moving the arm...")
        # arm_ctrl.move(arm_qpos[0], is_servo=False)
        # arm_ctrl.stop()
        print("moving the hand...")
        hand_ctrl.move(hand_qpos[0])
        # hand_ctrl.stop()
        # time.sleep(1.5)
        return

    name = args.object
    ep = args.ep

    cs = CaptureSession(
        camera=True,
        arm = args.arm,
        hand= args.hand,
        teleop=args.teleop_device
    )

    cs.timestamp_monitor = TimestampMonitor(**network_info["timestamp"]["param"])

    if args.output_dir is not None:
        save_path = os.path.join("capture", "hri_inspire_left", name, args.output_dir)
    else:
        save_path = os.path.join("capture", "hri_inspire_left", name, str(ep) + '_replay')
        
    cs.start(save_path)
    print("Starting new recording session:", name)
    print("Capturing index:", ep, '_replay`')
    
    print(f"Starting replay: {len(arm_qpos)} frames (arm) @ rate_scale={args.rate_scale}")
    try:
        for i in range(len(arm_qpos)):
            if i == 0:
                start_timestamp = cs.timestamp_monitor.get_data()
                start_time, start_frame_id = start_timestamp['time'], start_timestamp['frame_id']
                print(f"Start time: {start_time}, frame ID: {start_frame_id}")
                
            arm_ctrl.move(arm_qpos[i])
            cs.hand.move(hand_qpos[i])

            if i + 1 < len(arm_time):
                dt = (arm_time[i + 1] - arm_time[i]) / max(args.rate_scale, 1e-6)
                if dt < 0:
                    dt = 0.0
                time.sleep(dt)
                
        end_timestamp = cs.timestamp_monitor.get_data()
        end_time, end_frame_id = end_timestamp['time'], end_timestamp['frame_id']
        
        
        # time.sleep(0.1)
    except KeyboardInterrupt:
        print("Replay interrupted by user.")
    finally:
        try:
            hand_ctrl.end()
        except Exception:
            pass
        try:
            arm_ctrl.end()
        except Exception:
            pass
        print("Replay finished.")

    # arm_ctrl.stop()
    # hand_ctrl.stop()

    cs.stop()
    
    print("Stopped recording session:", name)
    print(f"Start time: {start_time}, frame ID: {start_frame_id}")
    print(f"End time: {end_time}, frame ID: {end_frame_id}")
    
    cs.end()
       

    

    
if __name__ == "__main__":
    main()
