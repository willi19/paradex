import argparse
import os
from typing import Tuple

import numpy as np

from paradex.utils.path import rsc_path
from paradex.visualization.visualizer.viser import ViserViewer


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


def inspire_action_to_qpos(action: np.ndarray) -> np.ndarray:
    # action order: little, ring, middle, index, thumb_2, thumb_1
    limits = {
        "pinky_proximal_joint": 1.6,
        "ring_proximal_joint": 1.6,
        "middle_proximal_joint": 1.6,
        "index_proximal_joint": 1.6,
        "thumb_proximal_pitch_joint": 0.55,
        "thumb_proximal_yaw_joint": 1.15,
    }
    qpos = np.zeros_like(action, dtype=float)
    qpos[:, 0] = limits["thumb_proximal_yaw_joint"] * (1.0 - action[:, 5] / 1000.0)
    qpos[:, 1] = limits["thumb_proximal_pitch_joint"] * (1.0 - action[:, 4] / 1000.0)
    qpos[:, 2] = limits["index_proximal_joint"] * (1.0 - action[:, 3] / 1000.0)
    qpos[:, 3] = limits["middle_proximal_joint"] * (1.0 - action[:, 2] / 1000.0)
    qpos[:, 4] = limits["ring_proximal_joint"] * (1.0 - action[:, 1] / 1000.0)
    qpos[:, 5] = limits["pinky_proximal_joint"] * (1.0 - action[:, 0] / 1000.0)
    return qpos


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--arm", type=str, required=True)
    parser.add_argument("--hand", type=str, required=True)
    parser.add_argument(
        "--data-root",
        type=str,
        default="/home/temp_id/shared_data/capture/hri_inspire_left/blue_speaker/5/raw",
        help="Capture root containing arm/ and hand/ directories.",
    )
    args = parser.parse_args()

    arm_dir = os.path.join(args.data_root, "arm")
    hand_dir = os.path.join(args.data_root, "hand")

    arm_qpos, arm_time = load_series(arm_dir, ("position.npy", "action_qpos.npy", "action.npy"))
    hand_action, hand_time = load_series(hand_dir, ("action.npy", "position.npy"))

    hand_action = resample_to(hand_time, hand_action, arm_time)
    if args.hand == "inspire":
        hand_qpos = inspire_action_to_qpos(hand_action)
    else:
        hand_qpos = hand_action

    full_qpos = np.concatenate([arm_qpos, hand_qpos], axis=1)

    urdf_path = os.path.join(rsc_path, "robot", f"{args.arm}_{args.hand}_left_new.urdf")
    vis = ViserViewer()
    vis.add_floor(height=0.0)
    vis.add_robot("robot", urdf_path)
    vis.add_traj("traj", {"robot": full_qpos})
    vis.start_viewer()


if __name__ == "__main__":
    main()
