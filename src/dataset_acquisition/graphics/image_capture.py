"""Graphics — exposure x gain grid search of images at a static xarm pose.

Uses remote_camera_controller directly (no CaptureSession, no
TimestampMonitor, no signal_generator). syncMode=False (free-run).

For each (exposure, gain) cell, saves one PNG per camera under
~/shared_data/.../exp{exp}_gain{gain}/raw/images/. Also records the
actual xarm joint pose (radians) and cartesian pose at capture time.
"""
import argparse
import glob
import json
import os
import shutil
import time
from datetime import datetime

import numpy as np

from paradex.io.camera_system.remote_camera_controller import remote_camera_controller
from paradex.io.robot_controller import get_arm
from paradex.utils.path import shared_dir

DEFAULT_TRAJ = os.path.expanduser("~/mcc_minimal/traj/dynamic/xarm/seed42_fwd100.npz")
DEFAULT_EXPOSURES = [2500, 8000, 16000, 30000]    # us
DEFAULT_GAINS = [0.0, 3.0, 6.0, 9.0, 12.0]        # dB
RESET_SPEED_RAD_S = 0.35                          # ~20 deg/s


def _reset_to(arm, q_deg):
    arm.move(np.deg2rad(np.asarray(q_deg, dtype=float)), is_servo=False, speed=RESET_SPEED_RAD_S)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--name", required=True)
    p.add_argument("--traj", default=DEFAULT_TRAJ)
    p.add_argument("--pose_idx", type=int, default=0,
                   help="index into q_deg (negative ok, e.g. -1 for last)")
    p.add_argument("--exposures", type=int, nargs="+", default=DEFAULT_EXPOSURES, help="us")
    p.add_argument("--gains", type=float, nargs="+", default=DEFAULT_GAINS, help="dB")
    p.add_argument("--settle", type=float, default=1.5, help="seconds after pose reset")
    args = p.parse_args()

    d = np.load(args.traj)
    q_deg = np.asarray(d["q_deg"], dtype=float)
    target = q_deg[args.pose_idx]
    print(f"target pose (q_deg[{args.pose_idx}]) = {target.tolist()}")

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    rel_base = os.path.join("capture", "graphics", "sharp_grid", args.name, timestamp)
    base_abs = os.path.join(shared_dir, rel_base)
    os.makedirs(base_abs, exist_ok=True)

    arm = get_arm("xarm")
    camera = remote_camera_controller(name="image_capture")
    try:
        _reset_to(arm, target)
        time.sleep(args.settle)

        snap = arm.get_data()
        np.save(os.path.join(base_abs, "qpos.npy"), snap["qpos"])
        np.save(os.path.join(base_abs, "ee_pose.npy"), snap["position"])
        print(f"saved xarm snapshot @ {base_abs}")

        for exp in args.exposures:
            for gain in args.gains:
                cell_rel = os.path.join("shared_data", rel_base,
                                        f"exp{int(exp)}_gain{gain:g}", "raw")
                print(f"=== exp={exp}us gain={gain}dB -> ~/{cell_rel}")
                camera.start("image", False, cell_rel,
                             exposure_time=exp, gain=gain)
                camera.stop()

        meta = {
            "traj": args.traj,
            "pose_idx": args.pose_idx,
            "q_deg_target": target.tolist(),
            "qpos_actual_rad": snap["qpos"].tolist(),
            "ee_pose_actual": snap["position"].tolist(),
            "exposures_us": list(map(int, args.exposures)),
            "gains_db": list(map(float, args.gains)),
            "syncMode": False,
            "timestamp": timestamp,
        }
        with open(os.path.join(base_abs, "meta.json"), "w") as f:
            json.dump(meta, f, indent=2)

        # Reorganize: by_serial/{serial}/exp{exp}_gain{gain}.png
        by_serial = os.path.join(base_abs, "by_serial")
        os.makedirs(by_serial, exist_ok=True)
        for exp in args.exposures:
            for gain in args.gains:
                cell_imgs_dir = os.path.join(
                    base_abs, f"exp{int(exp)}_gain{gain:g}", "raw", "images")
                for src in glob.glob(os.path.join(cell_imgs_dir, "*.png")):
                    serial = os.path.splitext(os.path.basename(src))[0]
                    dst_dir = os.path.join(by_serial, serial)
                    os.makedirs(dst_dir, exist_ok=True)
                    dst = os.path.join(dst_dir, f"exp{int(exp)}_gain{gain:g}.png")
                    try:
                        if os.path.exists(dst):
                            os.remove(dst)
                        os.link(src, dst)
                    except OSError:
                        shutil.copyfile(src, dst)
        print(f"reorganized → {by_serial}/{{serial}}/exp_gain.png")
    finally:
        camera.end()
        arm.end()

    print(f"done. base = {base_abs}")


if __name__ == "__main__":
    main()
