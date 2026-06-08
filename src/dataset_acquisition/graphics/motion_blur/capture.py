"""Graphics — Motion Blur Capture.

Sweeps (exposure_time us) × (joint speed deg/s) over a fixed pre-generated
joint trajectory. One sync video per cell.

trajectory file: ~/mcc_minimal/traj/dynamic/xarm/seed42_fwd100.npz (key `q_deg`)
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

DEFAULT_TRAJ = os.path.expanduser("~/mcc_minimal/traj/dynamic/xarm/seed42_fwd100.npz")
DEFAULT_EXPOSURES = [2500, 8000, 16000, 30000]   # us
DEFAULT_GAINS = [12.0, 6.0, 0.0, 0.0]            # dB, paired 1:1 with exposures
DEFAULT_SPEEDS = [60]                            # deg/s
RESET_SPEED_RAD_S = 0.35                         # ~20 deg/s


def _capture_trial(cs, base, stage, q_deg, exposure_us, gain_db, speed_deg_s, fps):
    reset_to(cs.arm, q_deg[0], speed_rad_s=RESET_SPEED_RAD_S)
    time.sleep(0.5)
    cs.start(base, mode="video", fps=fps, exposure_time=exposure_us, gain=gain_db, stage=stage)
    replay_q_deg(cs.arm, q_deg, speed_deg_s=speed_deg_s)
    cs.stop()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--name", required=True)
    p.add_argument("--traj", default=DEFAULT_TRAJ)
    p.add_argument("--exposures", type=int, nargs="+", default=DEFAULT_EXPOSURES)
    p.add_argument("--gains", type=float, nargs="+", default=DEFAULT_GAINS,
                   help="dB, paired 1:1 with --exposures")
    p.add_argument("--speeds", type=float, nargs="+", default=DEFAULT_SPEEDS)
    p.add_argument("--fps", type=int, default=30)
    args = p.parse_args()

    if len(args.gains) != len(args.exposures):
        raise SystemExit(f"--gains length ({len(args.gains)}) must equal --exposures length ({len(args.exposures)})")

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
        "gains_db": list(map(float, args.gains)),
        "speeds_deg_s": list(map(float, args.speeds)),
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

        for exp, gain in zip(args.exposures, args.gains):
            for spd in args.speeds:
                stage = f"exp{int(exp)}_gain{gain:g}_spd{int(spd)}"
                print(f"=== trial stage={stage} -> {base}/raw/{stage}")
                _capture_trial(cs, base, stage, q_deg, exp, gain, spd, args.fps)

        reset_to(cs.arm, q_deg[0], speed_rad_s=RESET_SPEED_RAD_S)
    finally:
        cs.end()

    print(f"done. base = {base_abs}")


if __name__ == "__main__":
    main()
