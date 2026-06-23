"""Stream taeksoo full trajectory to xarm + allegro_v5 over ROS.

Loads ``qpos_arm_full.npy`` (T,6 rad) + reconstructs the matching hand
trajectory (T,16 rad) and streams them at a fixed rate.

Safety:
  * Prompts for confirmation before any motion.
  * Sends home pose first (wait for arrival), then streams.
  * Aborts on any error event.

Run on the same host that has the xarm/allegro ROS bringup running.
"""

from __future__ import annotations

import argparse
import datetime
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

sys.path.append(str(Path(__file__).resolve().parents[2]))

from paradex.calibration.utils import save_current_C2R
from paradex.dataset_acqusition.capture import CaptureSession
from paradex.io.robot_controller import get_arm, get_hand
from paradex.utils.path import shared_dir


HOME_DEG = [-84.9, -25.5, -41.1, 118.8, -1.9, 158.1]
LOGICAL_JOINT_ORDER = [f"joint_{i}_0" for i in range(16)]


def _hand_qpos_logical(qpos_hand_rt: np.ndarray, rt_joint_names: list) -> np.ndarray:
    """Remap dex-retargeting hand qpos -> logical 0..15 order."""
    name_to_idx = {n: i for i, n in enumerate(rt_joint_names)}
    remap = np.array([name_to_idx[n] for n in LOGICAL_JOINT_ORDER], dtype=np.int64)
    return qpos_hand_rt[:, remap].astype(np.float64)


def _wait_xarm_at_target(xarm, target: np.ndarray, tol_rad: float, timeout: float) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        data = xarm.get_data()
        cur = data.get("qpos")
        if cur is not None and np.all(np.isfinite(cur)):
            if np.max(np.abs(cur - target)) < tol_rad:
                return True
        time.sleep(0.05)
    return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default=str(Path(__file__).parent / "out_taeksoo"))
    ap.add_argument("--side", default="right", choices=["right"])
    ap.add_argument("--rate-hz", type=float, default=30.0,
                    help="Trajectory streaming rate (matches plan_dt=0.033 -> 30 Hz).")
    ap.add_argument("--home-deg", nargs=6, type=float, default=HOME_DEG)
    ap.add_argument("--home-tol-deg", type=float, default=2.0)
    ap.add_argument("--home-timeout", type=float, default=20.0)
    ap.add_argument("--dry-run", action="store_true",
                    help="Init controllers and validate trajectory but don't move.")
    ap.add_argument("--yes", action="store_true",
                    help="Skip the interactive confirmation prompt.")
    ap.add_argument("--hand-close-start", type=int, default=28,
                    help="Frame at which the hand begins closing (linear interp from open). "
                    "Frames before this stay at zeros (open).")
    ap.add_argument("--hand-close-end", type=int, default=50,
                    help="Frame at which the hand reaches the trajectory pose. "
                    "Frames after this follow the loaded trajectory as-is.")
    ap.add_argument("--record-video", action="store_true",
                    help="Record video via CaptureSession (sync + timestamps + camparam).")
    ap.add_argument("--video-save-path", default="nips2026/taeksoo_send",
                    help="Save path under shared_data/.")
    ap.add_argument("--realsense", action="store_true",
                    help="Also record realsense depth cam (passed to CaptureSession).")
    args = ap.parse_args()

    out = Path(args.out_dir)
    qpos_arm = np.load(out / "qpos_arm_full.npy").astype(np.float64)
    qpos_hand_full_rt = np.load(out / "qpos_hand_full.npy").astype(np.float64)
    n_approach = int(np.load(out / "n_approach.npy")[0])
    rt_joint_names = json.loads((out / f"joint_names_{args.side}.json").read_text())

    Tn = qpos_arm.shape[0]
    if qpos_hand_full_rt.shape[0] != Tn:
        raise RuntimeError(
            f"Length mismatch: arm={Tn}, hand_full={qpos_hand_full_rt.shape[0]}"
        )
    hand_traj = _hand_qpos_logical(qpos_hand_full_rt, rt_joint_names)

    # Override hand trajectory: hold open until --hand-close-start, linearly
    # close to trajectory[--hand-close-end] over the gap, then follow trajectory.
    s = int(np.clip(args.hand_close_start, 0, Tn - 1))
    e = int(np.clip(args.hand_close_end, s + 1, Tn - 1))
    target_at_e = hand_traj[e].copy()
    hand_traj[:s] = 0.0
    if e > s:
        alphas = np.linspace(0.0, 1.0, e - s + 1).reshape(-1, 1)
        hand_traj[s : e + 1] = alphas * target_at_e[None, :]
    print(f"[send] T={Tn} (n_approach={n_approach}, rate={args.rate_hz} Hz)")
    print(f"[send] arm range deg: {np.rad2deg(qpos_arm).min(0)} .. {np.rad2deg(qpos_arm).max(0)}")
    print(f"[send] hand_close window: {s}..{e}, frames 0..{s-1} open, {e+1}..{Tn-1} trajectory")
    print(f"[send] hand frame 0: {hand_traj[0]}")
    print(f"[send] hand frame {s}: {hand_traj[s]}")
    print(f"[send] hand frame {e}: {hand_traj[e]}")

    home_rad = np.deg2rad(np.asarray(args.home_deg, dtype=np.float64))
    if not np.allclose(qpos_arm[0], home_rad, atol=1e-3):
        print(f"[send] WARNING: trajectory[0] != home; trajectory[0]={qpos_arm[0]}")

    if args.dry_run:
        print("[send] dry-run: skipping controllers")
        return

    if not args.yes:
        ans = input(
            f"Will move xarm to home then stream {Tn} frames @ {args.rate_hz} Hz "
            f"(~{Tn / args.rate_hz:.1f} s). Proceed? [y/N] "
        ).strip().lower()
        if ans != "y":
            print("[send] aborted")
            return
        for s in range(5, 0, -1):
            print(f"[send] starting in {s}...")
            time.sleep(1.0)

    cs = None
    capture_rel_path = None
    capture_dir_abs = None
    if args.record_video:
        date_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        capture_rel_path = os.path.join(args.video_save_path, date_str)
        capture_dir_abs = os.path.join(shared_dir, capture_rel_path)
        cs = CaptureSession(camera=True, realsense=args.realsense)
        cs.start(capture_rel_path)
        print(f"[send] recording video -> {capture_dir_abs}/raw")
        print("[send] video pre-roll: 3 s before robot motion ...")
        time.sleep(3.0)

    print("[send] init xarm (servo_api=angle_j)...")
    xarm = get_arm("xarm", servo_api="angle_j")
    print("[send] init allegro_v5...")
    allegro = get_hand("allegro_v5", hand_side=args.side)

    while not allegro.connection_event.is_set():
        time.sleep(0.1)
    print("[send] allegro connected")

    period = 1.0 / float(args.rate_hz)

    try:
        # Phase 0: send home + frame-0 hand pose (≈ open from low-pass warmup),
        # wait for arm to arrive at home.
        print(f"[send] commanding home_deg={args.home_deg} + hand={hand_traj[0]}")
        xarm.move(home_rad, is_servo=True)
        allegro.move(hand_traj[0])
        ok = _wait_xarm_at_target(
            xarm, home_rad, np.deg2rad(args.home_tol_deg), args.home_timeout
        )
        if not ok:
            print(f"[send] WARNING: arm did not reach home within {args.home_timeout}s")

        if xarm.is_error() or allegro.is_error():
            raise RuntimeError("controller in error state before streaming")

        # Phase 1: stream arm + hand together (baked trajectory; hand ramps
        # from open to retargeted via the dex-retargeting low-pass filter).
        print(f"[send] streaming {Tn} waypoints @ {args.rate_hz} Hz ...")
        t0 = time.perf_counter()
        for t in range(Tn):
            target_t = t0 + t * period
            xarm.move(qpos_arm[t], is_servo=True)
            allegro.move(hand_traj[t])
            if xarm.is_error() or allegro.is_error():
                raise RuntimeError(f"controller error at frame {t}")
            sleep_for = target_t + period - time.perf_counter()
            if sleep_for > 0:
                time.sleep(sleep_for)
        print("[send] streaming done")
        # Wait for the arm to physically settle at the last waypoint before
        # stopping the video.
        if cs is not None:
            time.sleep(1.0)
    finally:
        try:
            xarm.end(set_break=False)
        except Exception as e:
            print(f"[send] xarm.end error: {e}")
        try:
            allegro.end()
        except Exception as e:
            print(f"[send] allegro.end error: {e}")
        if cs is not None:
            try:
                cs.stop()
            except Exception as e:
                print(f"[send] cs.stop error: {e}")
            try:
                save_current_C2R(capture_dir_abs)
            except Exception as e:
                print(f"[send] save_current_C2R error: {e}")
            try:
                cs.end()
            except Exception as e:
                print(f"[send] cs.end error: {e}")
            print(f"[send] capture saved to {capture_dir_abs}")


if __name__ == "__main__":
    main()
