"""Graphics — Motion Blur Capture.

Sweeps (exposure_time us) × (joint speed deg/s) over a fixed pre-generated
joint trajectory. Captures one sync video per cell. Also captures sharp
single-frame images at trajectory start/end pose for reference.

trajectory file: ~/mcc_minimal/traj/dynamic/xarm/seed42.npz (key `q_deg`)
"""
import argparse
import json
import os
import sys
import time
from datetime import datetime

import numpy as np

from paradex.dataset_acqusition.capture import CaptureSession
from paradex.utils.path import shared_dir

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from replay import replay_q_deg, reset_to  # noqa: E402

DEFAULT_TRAJ = os.path.expanduser("~/mcc_minimal/traj/dynamic/xarm/seed42.npz")
DEFAULT_EXPOSURES = [2500, 8000, 16000, 30000]   # us
DEFAULT_SPEEDS = [10, 30, 60, 120]               # deg/s
SHARP_EXPOSURE = 2500                            # us
RESET_SPEED_RAD_S = 0.35                         # ~20 deg/s


def _capture_sharp(cs, save_path):
    cs.start(save_path, mode="image", fps=30, exposure_time=SHARP_EXPOSURE)
    time.sleep(2.0)
    cs.stop()


def _capture_trial(cs, save_path, q_deg, exposure_us, speed_deg_s, fps):
    reset_to(cs.arm, q_deg[0], speed_rad_s=RESET_SPEED_RAD_S)
    time.sleep(0.5)
    cs.start(save_path, mode="video", fps=fps, exposure_time=exposure_us)
    replay_q_deg(cs.arm, q_deg, speed_deg_s=speed_deg_s)
    cs.stop()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--name", required=True)
    p.add_argument("--traj", default=DEFAULT_TRAJ)
    p.add_argument("--exposures", type=int, nargs="+", default=DEFAULT_EXPOSURES)
    p.add_argument("--speeds", type=float, nargs="+", default=DEFAULT_SPEEDS)
    p.add_argument("--fps", type=int, default=30)
    args = p.parse_args()

    d = np.load(args.traj)
    q_deg = np.asarray(d["q_deg"], dtype=float)
    print(f"Loaded {len(q_deg)} waypoints from {args.traj}")

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    base = os.path.join("capture", "graphics", "motion_blur", args.name, timestamp)
    base_abs = os.path.join(shared_dir, base)
    os.makedirs(base_abs, exist_ok=True)

    meta = {
        "traj": args.traj,
        "n_waypoints": int(len(q_deg)),
        "exposures_us": list(map(int, args.exposures)),
        "speeds_deg_s": list(map(float, args.speeds)),
        "sharp_exposure_us": SHARP_EXPOSURE,
        "fps": args.fps,
        "q_start_deg": q_deg[0].tolist(),
        "q_end_deg": q_deg[-1].tolist(),
        "timestamp": timestamp,
    }
    with open(os.path.join(base_abs, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    cs = CaptureSession(camera=True, arm="xarm", hand=None, teleop=None)
    try:
        reset_to(cs.arm, q_deg[0], speed_rad_s=RESET_SPEED_RAD_S)
        time.sleep(1.0)
        _capture_sharp(cs, os.path.join(base, "sharp", "start"))

        for exp in args.exposures:
            for spd in args.speeds:
                trial = os.path.join(base, "trials", f"exp{int(exp)}_spd{int(spd)}")
                print(f"=== trial exposure={exp}us speed={spd}deg/s -> {trial}")
                _capture_trial(cs, trial, q_deg, exp, spd, args.fps)

        reset_to(cs.arm, q_deg[-1], speed_rad_s=RESET_SPEED_RAD_S)
        time.sleep(1.0)
        _capture_sharp(cs, os.path.join(base, "sharp", "end"))

        reset_to(cs.arm, q_deg[0], speed_rad_s=RESET_SPEED_RAD_S)
    finally:
        cs.end()

    print(f"done. base = {base_abs}")


if __name__ == "__main__":
    main()
