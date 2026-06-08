"""Graphics — image at every trajectory waypoint, sweeping (exposure, gain) pairs.

Mirrors src/calibration/handeye/capture.py pattern:
move xarm to each waypoint (blocking) → for each (exp, gain) pair:
rcc.start("image", False, ...) → rcc.stop() → save xarm snapshot.

Defaults pair 4 exposures with 4 gains 1:1 (matches motion_blur sweep).
"""
import argparse
import glob
import json
import os
import shutil
from datetime import datetime

import numpy as np

from paradex.io.camera_system.remote_camera_controller import remote_camera_controller
from paradex.io.robot_controller.xarm_controller import XArmController
from paradex.utils.file_io import remove_home
from paradex.utils.path import shared_dir
from paradex.utils.system import network_info
from paradex.calibration.utils import save_current_camparam

DEFAULT_TRAJ = os.path.expanduser("~/mcc_minimal/traj/dynamic/xarm/seed42_fwd100.npz")
DEFAULT_EXPOSURES = [2500, 8000, 16000, 30000]   # us
DEFAULT_GAINS = [12.0, 6.0, 0.0, 0.0]            # dB, paired 1:1 with exposures


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--name", required=True)
    p.add_argument("--traj", default=DEFAULT_TRAJ)
    p.add_argument("--exposures", type=int, nargs="+", default=DEFAULT_EXPOSURES, help="us")
    p.add_argument("--gains", type=float, nargs="+", default=DEFAULT_GAINS,
                   help="dB, paired 1:1 with --exposures")
    p.add_argument("--step", type=int, default=1,
                   help="waypoint stride (1 = every waypoint, 20 = every 20th)")
    p.add_argument("--start_idx", type=int, default=0)
    p.add_argument("--end_idx", type=int, default=None,
                   help="exclusive; default = len(q_deg)")
    args = p.parse_args()

    if len(args.gains) != len(args.exposures):
        raise SystemExit(f"--gains length ({len(args.gains)}) must equal --exposures length ({len(args.exposures)})")

    d = np.load(args.traj)
    q_deg_full = np.asarray(d["q_deg"], dtype=float)
    end_idx = args.end_idx if args.end_idx is not None else len(q_deg_full)
    indices = list(range(args.start_idx, end_idx, args.step))
    pairs = list(zip(args.exposures, args.gains))
    print(f"will capture {len(indices)} waypoints × {len(pairs)} (exp,gain) pairs = {len(indices) * len(pairs)} cells")

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    rel_base = os.path.join("capture", "graphics", "sharp_traj", args.name, timestamp)
    base_abs = os.path.join(shared_dir, rel_base)
    os.makedirs(base_abs, exist_ok=True)
    save_current_camparam(base_abs)

    controller = XArmController(**network_info["xarm"]["param"])
    rcc = remote_camera_controller("image_traj")
    try:
        for k, i in enumerate(indices):
            target_rad = np.deg2rad(q_deg_full[i])
            print(f"[{k+1}/{len(indices)}] q[{i}]")

            controller.move(target_rad, is_servo=False)

            waypoint_dir = os.path.join(base_abs, f"q{i:04d}")
            os.makedirs(waypoint_dir, exist_ok=True)

            for exp, gain in pairs:
                cell_abs = os.path.join(waypoint_dir, f"exp{int(exp)}_gain{gain:g}")
                os.makedirs(os.path.join(cell_abs, "images"), exist_ok=True)
                rcc.start("image", False, remove_home(cell_abs),
                          exposure_time=int(exp), gain=float(gain))
                rcc.stop()

            robot_data = controller.get_data()
            np.save(os.path.join(waypoint_dir, "robot.npy"), robot_data)
            np.save(os.path.join(waypoint_dir, "qpos.npy"), robot_data["qpos"])
            np.save(os.path.join(waypoint_dir, "eef.npy"), robot_data["position"])

        meta = {
            "traj": args.traj,
            "n_waypoints_total": int(len(q_deg_full)),
            "indices": indices,
            "exposures_us": list(map(int, args.exposures)),
            "gains_db": list(map(float, args.gains)),
            "syncMode": False,
            "timestamp": timestamp,
        }
        with open(os.path.join(base_abs, "meta.json"), "w") as f:
            json.dump(meta, f, indent=2)

        # Reorganize: by_serial/{serial}/q{i:04d}_exp{exp}_gain{gain}.png
        by_serial = os.path.join(base_abs, "by_serial")
        os.makedirs(by_serial, exist_ok=True)
        for i in indices:
            for exp, gain in pairs:
                cell_imgs_dir = os.path.join(
                    base_abs, f"q{i:04d}", f"exp{int(exp)}_gain{gain:g}", "images")
                for src in glob.glob(os.path.join(cell_imgs_dir, "*.png")):
                    serial = os.path.splitext(os.path.basename(src))[0]
                    dst_dir = os.path.join(by_serial, serial)
                    os.makedirs(dst_dir, exist_ok=True)
                    dst = os.path.join(dst_dir, f"q{i:04d}_exp{int(exp)}_gain{gain:g}.png")
                    try:
                        if os.path.exists(dst):
                            os.remove(dst)
                        os.link(src, dst)
                    except OSError:
                        shutil.copyfile(src, dst)
        print(f"reorganized → {by_serial}/{{serial}}/q####_exp_gain.png")
    finally:
        controller.end(True)
        rcc.end()

    print(f"done. base = {base_abs}")


if __name__ == "__main__":
    main()
