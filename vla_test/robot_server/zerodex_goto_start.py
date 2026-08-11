"""Move the xArm6 to a zerodex training start pose (slow joint move), then hold.

Real arm motion. Position-control (is_servo=False) joint move to a pose sampled
from the pepsi training episodes' first frames, so the policy state is in-distribution
before inference.

Usage:
  PYTHONPATH=. python vla_test/robot_server/zerodex_goto_start.py            # representative
  PYTHONPATH=. python vla_test/robot_server/zerodex_goto_start.py --median   # median pose
  PYTHONPATH=. python vla_test/robot_server/zerodex_goto_start.py --joints "-0.108,0.028,-1.422,-1.504,0.271,0.121"
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from paradex.io.robot_controller import get_arm
from paradex.vla.client import read_arm_qpos_rad
from paradex.vla import zerodex as Z

# from pepsi training episodes' first frames (build_zerodex_fk dataset)
START_REPRESENTATIVE = np.array([-0.1083, 0.028, -1.4219, -1.504, 0.2709, 0.1208])
START_MEDIAN = np.array([-0.1351, -0.1012, -1.2839, -1.581, 0.0559, 0.2527])


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--median", action="store_true", help="use the median start pose")
    p.add_argument("--joints", type=str, default=None, help="comma-separated 6 joint rad")
    p.add_argument("--speed", type=float, default=0.2, help="rad/s (slow)")
    args = p.parse_args()

    if args.joints:
        target = np.array([float(x) for x in args.joints.split(",")])
        assert target.shape == (6,), "need 6 joint values"
    elif args.median:
        target = START_MEDIAN
    else:
        target = START_REPRESENTATIVE

    arm = get_arm("xarm")
    try:
        time.sleep(0.5)
        cur = read_arm_qpos_rad(arm)
        print(f"current  arm_joint = {np.round(cur,4)}")
        print(f"target   arm_joint = {np.round(target,4)}")
        print(f"max joint delta    = {np.round(np.abs(target-cur).max(),4)} rad")
        print(f"[goto] moving at {args.speed} rad/s (is_servo=False, waits for completion) ...")
        t0 = time.time()
        arm.move(target.astype(np.float64), is_servo=False, speed=args.speed)
        print(f"[goto] done in {time.time()-t0:.1f}s")
        time.sleep(0.5)
        final = read_arm_qpos_rad(arm)
        eef = Z.eef_pose9(final)
        print(f"final    arm_joint = {np.round(final,4)}")
        print(f"final    eef xyz(m)= {np.round(eef[:3],4)}")
        print(f"residual to target = {np.round(np.abs(final-target).max(),4)} rad")
    finally:
        try: arm.end(set_break=False)
        except Exception as e: print(f"[goto] arm.end err: {e}")


if __name__ == "__main__":
    main()
