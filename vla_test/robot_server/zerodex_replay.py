"""Open-loop replay of a zerodex demonstration on the real xArm6 + Inspire.

NO policy server. Loads a demo episode (arm_joint / eef_target / hand_cmd),
moves the arm to the demo's first joint pose, then streams the demo's ABSOLUTE
eef_target [xyz, rot6d] + hand_cmd frame-by-frame at 30 Hz through the SAME
execution path the VLA client uses (send_arm_cartesian tool_frame=tcp + send_hand).

This validates the execution pipeline (IK/tcp frame/hand mapping) against ground
truth: if the robot reproduces the demonstrated pick-place, execution is correct
and any weirdness in live inference is the policy, not the plumbing.

Usage:
  # dry (no motion): print trajectory + per-step jumps + co-frame check
  PYTHONPATH=. python vla_test/robot_server/zerodex_replay.py

  # real replay at half speed (safer first run), then full speed
  PYTHONPATH=. python vla_test/robot_server/zerodex_replay.py --execute --rate 0.5
  PYTHONPATH=. python vla_test/robot_server/zerodex_replay.py --execute --rate 1.0

Ctrl-C stops (arm holds last commanded step). Keep the E-STOP ready.
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from paradex.vla import zerodex as Z
from paradex.vla.client import read_arm_qpos_rad


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--npz", default=str(Path(__file__).with_name("replay_pepsi.npz")))
    p.add_argument("--execute", action="store_true", help="actually MOVE the robot")
    p.add_argument("--rate", type=float, default=1.0,
                   help="replay speed multiplier (0.5 = half speed / safer)")
    p.add_argument("--stride", type=int, default=1, help="use every Nth demo frame")
    p.add_argument("--goto-speed", type=float, default=0.2, help="rad/s for the initial move")
    p.add_argument("--max-step-mm", type=float, default=30.0,
                   help="reject replay if any consecutive eef_target jump exceeds this")
    args = p.parse_args()

    d = np.load(args.npz, allow_pickle=True)
    arm_joint = d["arm_joint"].astype(np.float64)      # (N,6)
    eef_target = d["eef_target"].astype(np.float64)    # (N,9) absolute [xyz,rot6d]
    hand_cmd = d["hand_cmd"].astype(np.float64)        # (N,6) raw 0..1000
    fps = float(d["fps"]) if "fps" in d else 30.0
    N = len(eef_target)
    idx = list(range(0, N, args.stride))
    print(f"[replay] {args.npz}")
    print(f"[replay] episode={d['episode']} prompt={str(d['prompt'])!r} "
          f"N={N} using {len(idx)} frames @ {fps*args.rate:.0f}Hz effective")

    # per-step jump check on the (strided) absolute trajectory
    tgt = eef_target[idx]
    jumps = np.linalg.norm(np.diff(tgt[:, :3], axis=0), axis=1) * 1000.0
    print(f"[replay] eef_target step jump: max={jumps.max():.1f}mm mean={jumps.mean():.1f}mm")
    if jumps.max() > args.max_step_mm:
        bad = int(np.argmax(jumps))
        print(f"[replay] WARN: max jump {jumps.max():.1f}mm at frame {bad} "
              f"> {args.max_step_mm}mm (reduce --stride).")

    # co-frame sanity: our FK@T_tool of demo joints vs demo eef_target
    ef0 = Z.eef_pose9(arm_joint[0])
    cof = np.linalg.norm(ef0[:3] - eef_target[0][:3]) * 1000.0
    print(f"[replay] co-frame check: fk_tcp(arm_joint[0]) vs eef_target[0] = {cof:.1f}mm "
          f"(should be small)")
    print(f"[replay] start joints = {np.round(arm_joint[0],4)}")
    print(f"[replay] eef xyz: start={np.round(tgt[0,:3],3)} end={np.round(tgt[-1,:3],3)}")
    print(f"[replay] hand: start={np.round(hand_cmd[idx[0]],0)} end={np.round(hand_cmd[idx[-1]],0)}")

    if not args.execute:
        print("[replay] DRY (no --execute): nothing moved.")
        return

    from paradex.io.robot_controller import get_arm, get_hand
    print("[replay] connecting xArm + Inspire (hand parks at neutral) ...")
    arm = get_arm("xarm")
    hand = get_hand("inspire", ip=True)
    try:
        time.sleep(0.6)
        # 1) move to the demo's first joint pose (exact demo start)
        cur = read_arm_qpos_rad(arm)
        print(f"[replay] goto demo start: max joint delta="
              f"{np.abs(arm_joint[0]-cur).max():.3f} rad @ {args.goto_speed} rad/s ...")
        arm.move(arm_joint[0], is_servo=False, speed=args.goto_speed)
        hand.move(np.clip(hand_cmd[idx[0]], 0, 1000).astype(np.int32))
        time.sleep(0.8)

        # 2) stream absolute eef_target + hand_cmd
        dt = 1.0 / (fps * args.rate)
        print(f"[replay] streaming {len(idx)} steps (dt={dt*1000:.0f}ms). Ctrl-C to stop.")
        t0 = time.time()
        for k, i in enumerate(idx):
            Z.send_arm_cartesian(arm, eef_target[i], tool_frame="tcp")
            hand.move(np.clip(hand_cmd[i], 0, 1000).astype(np.int32))
            if k % 30 == 0:
                print(f"  step {k:4d}/{len(idx)}  eef={np.round(eef_target[i,:3],3)} "
                      f"hand={np.round(hand_cmd[i],0)}")
            time.sleep(dt)
        print(f"[replay] done — {len(idx)} steps in {time.time()-t0:.1f}s")
    except KeyboardInterrupt:
        print("\n[replay] stopped by user")
    finally:
        try: arm.end(set_break=False)
        except Exception as e: print(f"[replay] arm.end err: {e}")
        try: hand.end()
        except Exception as e: print(f"[replay] hand.end err: {e}")


if __name__ == "__main__":
    main()
