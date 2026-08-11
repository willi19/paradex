"""zerodex preflight — verify the full inference pipeline WITHOUT moving the arm.

Stages (each optional):
  cameras  : bring up the 2-cam stream (divisor 3), grab one frame each, save PNGs,
             print resolution. NO robot connection.
  --robot  : also connect xArm(read)+Inspire, read state, run ONE server query,
             print plan + safety + IK. The Inspire hand parks at neutral on connect;
             the xArm is only READ (never moved).

Usage:
  PYTHONPATH=. python vla_test/robot_server/zerodex_preflight.py --out /tmp/pf
  PYTHONPATH=. python vla_test/robot_server/zerodex_preflight.py --out /tmp/pf --robot \
      --prompt "move pepsi to the pink basket"
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import cv2

from paradex.vla import zerodex as Z
from vla_test.robot_server.zerodex_client import (
    bring_up_cameras, read_frame, teardown_cameras, _read_hand_raw)
from paradex.vla.client import read_arm_qpos_rad, prepare_frame_for_vla


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--host", default="147.46.15.24")
    p.add_argument("--port", type=int, default=8901)
    p.add_argument("--out", default="/tmp/zerodex_preflight")
    p.add_argument("--cam-main", default=Z.CAM_MAIN)
    p.add_argument("--cam-secondary", default=Z.CAM_SECONDARY)
    p.add_argument("--camera-fps", type=int, default=30)
    p.add_argument("--resize-divisor", type=int, default=3)
    p.add_argument("--warmup", type=float, default=3.0, help="seconds to let the stream warm up")
    p.add_argument("--robot", action="store_true", help="also read robot state + run one query")
    p.add_argument("--prompt", default=Z.PROMPTS[2], choices=Z.PROMPTS)
    args = p.parse_args()

    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)

    print("[preflight] ping server ...")
    policy = Z.ZerodexPolicyClient(args.host, args.port)
    print("  ping:", policy.ping())

    rcc = dc = cam_source = None
    arm = hand = None
    try:
        rcc, dc, cam_source = bring_up_cameras(
            args.camera_fps, decode_serials={args.cam_main, args.cam_secondary},
            resize_divisor=args.resize_divisor)
        print(f"[preflight] warming up {args.warmup}s ...")
        time.sleep(args.warmup)

        # grab raw tiles + prepared frames for both cams
        frames = {}
        for tag, serial in (("main", args.cam_main), ("secondary", args.cam_secondary)):
            raw = None
            deadline = time.time() + 5.0
            while time.time() < deadline and raw is None:
                raw = cam_source.latest(serial)
                if raw is None: time.sleep(0.05)
            if raw is None:
                print(f"  [{tag}] {serial}: NO FRAME. available={cam_source.available_serials()}")
                continue
            vla = prepare_frame_for_vla(raw)  # (480,640,3) RGB
            frames[tag] = vla
            cv2.imwrite(str(out / f"{tag}_{serial}_raw.png"), raw)
            cv2.imwrite(str(out / f"{tag}_{serial}_vla640x480.png"),
                        cv2.cvtColor(vla, cv2.COLOR_RGB2BGR))
            print(f"  [{tag}] {serial}: stream tile {raw.shape} -> vla {vla.shape}  "
                  f"saved to {out}/{tag}_{serial}_*.png")

        if args.robot:
            print("[preflight] connecting xArm (READ) + Inspire (parks at neutral) ...")
            from paradex.io.robot_controller import get_arm, get_hand
            arm = get_arm("xarm")
            hand = get_hand("inspire", ip=True)
            time.sleep(1.0)
            arm_q = read_arm_qpos_rad(arm)
            hand_raw = _read_hand_raw(hand)
            print(f"  arm_joint(rad)={np.round(arm_q,4)}")
            print(f"  hand_joint(raw)={np.round(hand_raw,0)}")
            eef_now = Z.eef_pose9(arm_q)
            print(f"  eef_pose xyz(m)={np.round(eef_now[:3],4)} rot6d={np.round(eef_now[3:],3)}")

            if "main" in frames and "secondary" in frames:
                obs = Z.build_observation(frames["main"], frames["secondary"],
                                          arm_q, hand_raw, args.prompt)
                t0 = time.perf_counter()
                eef_chunk, hand_chunk = policy.get_action(obs)
                infer_ms = 1e3 * (time.perf_counter() - t0)
                print(f"[preflight] QUERY prompt={args.prompt!r} infer={infer_ms:.0f}ms "
                      f"eef{eef_chunk.shape} hand{hand_chunk.shape}")
                d0 = (eef_chunk[0, :3] - eef_now[:3]) * 1000
                dN = (eef_chunk[-1, :3] - eef_now[:3]) * 1000
                print(f"  tgt0(m)={np.round(eef_chunk[0,:3],3)} d0={np.round(d0,1)}mm | "
                      f"tgtN(m)={np.round(eef_chunk[-1,:3],3)} dN={np.round(dN,1)}mm")
                print(f"  hand0={np.round(hand_chunk[0],0)} handN={np.round(hand_chunk[-1],0)}")
                ok, why = Z.check_plan_safe(eef_chunk, eef_now)
                print(f"  safety: {'OK' if ok else 'REJECT'} — {why}")
                q, pe, re = Z.ik_tcp(eef_chunk[0], arm_q)
                print(f"  IK tgt0: pos_err={pe*1000:.2f}mm rot_err={re:.2f}deg  q={np.round(q,4)}")
            else:
                print("  [skip query] missing one of the camera frames")
    finally:
        if cam_source is not None:
            teardown_cameras(rcc, dc, cam_source)
        for dev in (arm, hand):
            if dev is not None:
                try: dev.end()
                except Exception: pass
        try: policy.close()
        except Exception: pass
        print("[preflight] done.")


if __name__ == "__main__":
    main()
