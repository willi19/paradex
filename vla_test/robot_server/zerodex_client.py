"""zerodex GR00T deployment client for paradex xArm6 + Inspire.

Connects to the zerodex policy server on vclabserver2 (tcp://147.46.15.24:8901),
reads the real robot + two FLIR cameras, queries the VLA, and (optionally)
executes the returned absolute-TCP action chunk on the physical robot with a
receding horizon.

All obs/action math lives in paradex.vla.zerodex and is byte-identical to the
training pipeline (FK@T_tool, column rot6d, absolute eef_target, raw hand_cmd).

REPL:
  q  -> query the policy with the current obs, print + safety-check the plan
  e  -> execute the first --exec-steps of the most-recent plan (arm motion!)
  l  -> closed loop: query+execute repeatedly until STOP (Ctrl-C)
  x  -> exit

SAFETY: arm motion happens ONLY with --execute AND a passing safety check.
Without --execute, q still reads real state and queries the server (the Inspire
hand parks at its neutral pose on connect regardless — that is inherent to the
paradex controller). The xArm holds position until you press 'e'.

Usage:
  PYTHONPATH=. python vla_test/robot_server/zerodex_client.py \
      --host 147.46.15.24 --port 8901 \
      --prompt "move pepsi to the pink basket" \
      --execute

NOTE: the xArm SDK allows ONE client at a time — do not run viewer.py alongside.
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import subprocess

from paradex.io.camera_system.remote_camera_controller import remote_camera_controller
from paradex.io.capture_pc.data_sender import DataCollector
from paradex.io.capture_pc.ssh import run_script, ssh_port
from paradex.utils.system import get_pc_list, get_pc_ip
from paradex.vla.client import CameraFrameSource, prepare_frame_for_vla, read_arm_qpos_rad
from paradex.vla import zerodex as Z


def _read_hand_raw(hand):
    """Inspire OBSERVED angleAct, RAW 0..1000, native order
    [little,ring,middle,index,thumb_bend,thumb_rot] — exactly state.hand_joint."""
    return np.asarray(hand.get_qpos(), dtype=np.float32)


def read_frame(cam_source, serial):
    deadline = time.time() + 5.0
    while time.time() < deadline:
        bgr = cam_source.latest(serial)
        if bgr is not None:
            return prepare_frame_for_vla(bgr)   # -> (480,640,3) RGB uint8
        time.sleep(0.05)
    raise RuntimeError(
        f"no frame for serial {serial} within 5s. "
        f"available: {cam_source.available_serials()}. "
        f"Is the daemon up and stream_client.py running on every capture PC?")


def _kill_stale_stream_clients():
    """Kill leftover stream_client.py on every capture PC (daemons untouched).
    A stale stream_client from a previous run still binds DataPublisher:1234, so a
    freshly launched one gets no port and publishes nothing -> DataCollector sees
    no serials. Killing only stream_client (not server_daemon) keeps the cameras up.
    """
    for pc in get_pc_list():
        ip = get_pc_ip(pc)
        cmd = ["ssh", "-p", str(ssh_port), "-o", "ConnectTimeout=6",
               f"{pc}@{ip}", "pkill -f 'src/capture/camera/stream_client.py' || true"]
        try:
            subprocess.run(cmd, timeout=12,
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        except Exception as e:
            print(f"[zerodex] stream cleanup on {pc}: {e}")


def bring_up_cameras(fps, decode_serials, resize_divisor=3):
    _kill_stale_stream_clients()
    time.sleep(1.0)
    # Training frames were full sensor (1536x2048) downscaled to 640x480 (~/3.2).
    # stream_client's default divisor is 8 -> 192x256, which we would then UPSCALE
    # to 640x480 (heavy quality loss vs training). Use divisor 3 -> 512x682 and
    # restrict to the 2 needed serials so the bandwidth stays sane; prepare_frame_
    # for_vla then does the final clean downscale to 640x480 (~training quality).
    cams = ",".join(sorted(decode_serials))
    print(f"[zerodex] launching stream_client (divisor={resize_divisor}, cams={cams}) ...")
    run_script(f"python src/capture/camera/stream_client.py "
               f"--resize-divisor {resize_divisor} --cameras {cams}")
    print("[zerodex] opening remote_camera_controller as OWNER ...")
    rcc = remote_camera_controller("zerodex_client")
    print("[zerodex] starting DataCollector (port 1234) ...")
    dc = DataCollector()
    dc.start()
    print(f"[zerodex] commanding daemons -> stream mode (fps={fps}) ...")
    rcc.start("stream", False, fps=fps)
    cam_source = CameraFrameSource(dc, decode_serials=decode_serials)
    return rcc, dc, cam_source


def teardown_cameras(rcc, dc, cam_source):
    if cam_source is not None:
        cam_source.stop()
    for fn in (rcc.stop, rcc.end, dc.end):
        try: fn()
        except Exception as e: print(f"[zerodex] teardown err: {e}")


def query(policy, cam_source, arm_reader, hand, prompt, cam_main, cam_secondary,
          couple_thumb=False, couple_rot=False, thumb_gain=1.0):
    """Read real obs, query the server. Returns (eef_chunk, hand_chunk, arm_q, eef_now9)."""
    arm_q = read_arm_qpos_rad(arm_reader)
    hand_raw = _read_hand_raw(hand)
    img_main = read_frame(cam_source, cam_main)
    img_secondary = read_frame(cam_source, cam_secondary)
    obs = Z.build_observation(img_main, img_secondary, arm_q, hand_raw, prompt)
    eef_now9 = Z.eef_pose9(arm_q)
    t0 = time.perf_counter()
    eef_chunk, hand_chunk = policy.get_action(obs)
    infer_ms = 1e3 * (time.perf_counter() - t0)
    if couple_thumb:
        # Fix the stuck-thumb bias: drive the thumb from the predicted fingers.
        hand_chunk = Z.couple_thumb_to_fingers(hand_chunk, couple_rot=couple_rot, gain=thumb_gain)
    return eef_chunk, hand_chunk, arm_q, eef_now9, infer_ms


def print_plan(eef_chunk, hand_chunk, eef_now9, infer_ms):
    d0 = (eef_chunk[0, :3] - eef_now9[:3]) * 1000.0
    dN = (eef_chunk[-1, :3] - eef_now9[:3]) * 1000.0
    print(f"  infer={infer_ms:.0f}ms H={len(eef_chunk)} | eef_now(m)={np.round(eef_now9[:3],3)}")
    print(f"    -> tgt0(m)={np.round(eef_chunk[0,:3],3)} d0={np.round(d0,1)}mm  "
          f"tgtN(m)={np.round(eef_chunk[-1,:3],3)} dN={np.round(dN,1)}mm")
    print(f"    hand0={np.round(hand_chunk[0],0)}  handN={np.round(hand_chunk[-1],0)}")


def execute(arm, hand, eef_chunk, hand_chunk, exec_steps, hz, tool_frame, stop_flag=None):
    dt = 1.0 / hz
    n = min(exec_steps, len(eef_chunk))
    for i in range(n):
        if stop_flag is not None and stop_flag():
            print(f"  STOP at step {i}")
            return i
        Z.send_arm_cartesian(arm, eef_chunk[i], tool_frame=tool_frame)
        Z.send_hand(hand, hand_chunk[i])
        time.sleep(dt)
    return n


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--host", default="147.46.15.24")
    p.add_argument("--port", type=int, default=8901)
    p.add_argument("--prompt", default=Z.PROMPTS[2], choices=Z.PROMPTS,
                   help="in-distribution instruction (only these 4 were trained)")
    p.add_argument("--exec-steps", type=int, default=16,
                   help="steps of the 16-step chunk to run before replanning. 16 = run the "
                        "whole chunk (inference ~0.5s hides behind the 0.53s of motion); "
                        "8 = more reactive but can starve between chunks.")
    p.add_argument("--hz", type=float, default=30.0, help="per-step command rate (training=30)")
    p.add_argument("--tool-frame", choices=["flange", "tcp"], default="tcp",
                   help="frame arm.move() commands. VERIFIED on this rig: xArm cartesian "
                        "== FK@T_TOOL (TCP), so 'tcp' (send eef_target unchanged). "
                        "'flange' caused a ~123mm x offset / backward drift.")
    p.add_argument("--cam-main", default=Z.CAM_MAIN)
    p.add_argument("--cam-secondary", default=Z.CAM_SECONDARY)
    p.add_argument("--camera-fps", type=int, default=30,
                   help="stream fps (training was 30). VLA uses one frame per query so "
                        "10 also suffices, but 30 gives a fresher frame and matches training; "
                        "drop to 10 only if the capture PC struggles under load.")
    p.add_argument("--resize-divisor", type=int, default=3,
                   help="stream downsample divisor (3 -> 512x682 ~= training quality; "
                        "8 is the bandwidth default but too low for the VLA)")
    p.add_argument("--couple-thumb", action="store_true",
                   help="override the policy's thumb: drive thumb bend from the mean of the 4 "
                        "fingers so it grips when they grip / opens when they open (fixes the "
                        "stuck-thumb bias baked into the training data).")
    p.add_argument("--couple-thumb1", action="store_true", help="also couple thumb rotation.")
    p.add_argument("--thumb-gain", type=float, default=1.0,
                   help="closure gain for the coupled thumb (>1 closes more than the fingers).")
    p.add_argument("--show", action="store_true",
                   help="serve the live VLA input frames (main|secondary, exactly as sent "
                        "to the policy) as an MJPEG stream to open in a browser.")
    p.add_argument("--show-port", type=int, default=8100)
    p.add_argument("--execute", action="store_true", help="ACTUALLY move the xArm on 'e'/'l'")
    p.add_argument("--no-safety", action="store_true", help="skip the chunk safety check (DANGEROUS)")
    p.add_argument("--max-cycles", type=int, default=1000)
    p.add_argument("--auto", type=int, default=0,
                   help="non-interactive: run this many closed-loop cycles then exit "
                        "(requires --execute). 0 = interactive REPL.")
    args = p.parse_args()

    policy = Z.ZerodexPolicyClient(args.host, args.port)
    print("ping:", policy.ping())
    print(f"[zerodex] connected {args.host}:{args.port}  execute={args.execute} "
          f"tool_frame={args.tool_frame}  prompt={args.prompt!r}")

    # arm: for reading joints we use the paradex controller (holds position; only
    # moves when send_arm_cartesian is called, i.e. on 'e'/'l' with --execute).
    from paradex.io.robot_controller import get_arm, get_hand
    print("[zerodex] connecting xArm + Inspire (hand parks at neutral on connect) ...")
    arm = get_arm("xarm")
    hand = get_hand("inspire", ip=True)

    rcc, dc, cam_source = bring_up_cameras(
        args.camera_fps, decode_serials={args.cam_main, args.cam_secondary},
        resize_divisor=args.resize_divisor)

    # Optional live view of the exact frames sent to the VLA (main | secondary).
    live = None
    if args.show:
        import cv2
        from paradex.vla.liveview import MJPEGLiveView

        def _obs_frame():
            a = cam_source.latest(args.cam_main)
            b = cam_source.latest(args.cam_secondary)
            if a is None or b is None:
                return None
            ra = prepare_frame_for_vla(a)   # (480,640,3) RGB — exactly the VLA input
            rb = prepare_frame_for_vla(b)
            comp = cv2.cvtColor(np.hstack([ra, rb]), cv2.COLOR_RGB2BGR)
            cv2.putText(comp, f"main {args.cam_main}", (6, 22),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            cv2.putText(comp, f"secondary {args.cam_secondary}", (646, 22),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            return comp

        live = MJPEGLiveView(_obs_frame, port=args.show_port, title="zerodex VLA input")
        live.start()

    eef_chunk = hand_chunk = eef_now9 = None
    qn = en = 0

    if args.auto > 0:
        if not args.execute:
            print("[zerodex] --auto requires --execute; aborting (no motion).")
        else:
            print(f"[zerodex] AUTO closed loop: {args.auto} cycles — Ctrl-C to stop")
            try:
                for cyc in range(args.auto):
                    eef_chunk, hand_chunk, _, eef_now9, infer_ms = query(
                        policy, cam_source, arm, hand, args.prompt,
                        args.cam_main, args.cam_secondary,
                        couple_thumb=args.couple_thumb, couple_rot=args.couple_thumb1,
                        thumb_gain=args.thumb_gain)
                    print(f"  cycle {cyc:03d}", end=" "); print_plan(eef_chunk, hand_chunk, eef_now9, infer_ms)
                    if not args.no_safety:
                        ok, why = Z.check_plan_safe(eef_chunk, eef_now9)
                        if not ok:
                            print(f"  STOP: safety REJECT — {why}"); break
                    execute(arm, hand, eef_chunk, hand_chunk,
                            args.exec_steps, args.hz, args.tool_frame)
            except KeyboardInterrupt:
                print("\n  auto loop stopped by user")
        if live is not None:
            live.stop()
        teardown_cameras(rcc, dc, cam_source)
        try: arm.end(set_break=False)
        except Exception: pass
        try: hand.end()
        except Exception: pass
        try: policy.close()
        except Exception: pass
        return

    print("\nReady. Commands:  q=query  e=execute  l=loop  x=exit")
    try:
        while True:
            try:
                cmd = input("\n> ").strip().lower()
            except (EOFError, KeyboardInterrupt):
                print("\nexit"); break

            if cmd == "x":
                break

            elif cmd == "q":
                try:
                    eef_chunk, hand_chunk, _, eef_now9, infer_ms = query(
                        policy, cam_source, arm, hand, args.prompt,
                        args.cam_main, args.cam_secondary,
                        couple_thumb=args.couple_thumb, couple_rot=args.couple_thumb1,
                        thumb_gain=args.thumb_gain)
                except Exception as e:
                    print(f"  query failed: {e}"); continue
                qn += 1
                print(f"  [query #{qn}]"); print_plan(eef_chunk, hand_chunk, eef_now9, infer_ms)
                ok, why = Z.check_plan_safe(eef_chunk, eef_now9)
                print(f"  safety: {'OK' if ok else 'REJECT'} — {why}")

            elif cmd == "e":
                if eef_chunk is None:
                    print("  no plan yet — run 'q' first"); continue
                if not args.no_safety:
                    ok, why = Z.check_plan_safe(eef_chunk, eef_now9)
                    if not ok:
                        print(f"  REFUSED: {why} (re-run with --no-safety to override)"); continue
                if not args.execute:
                    print("  --execute not set: simulating (no motion)."); continue
                en += 1
                t = time.time()
                sent = execute(arm, hand, eef_chunk, hand_chunk,
                               args.exec_steps, args.hz, args.tool_frame)
                print(f"  [execute #{en}] {sent} steps in {time.time()-t:.2f}s")

            elif cmd == "l":
                if not args.execute:
                    print("  --execute not set: loop would only simulate. Aborting."); continue
                print("  closed loop — Ctrl-C to stop")
                try:
                    for cyc in range(args.max_cycles):
                        eef_chunk, hand_chunk, _, eef_now9, infer_ms = query(
                            policy, cam_source, arm, hand, args.prompt,
                            args.cam_main, args.cam_secondary,
                        couple_thumb=args.couple_thumb, couple_rot=args.couple_thumb1,
                        thumb_gain=args.thumb_gain)
                        print(f"  cycle {cyc:03d}", end=" "); print_plan(eef_chunk, hand_chunk, eef_now9, infer_ms)
                        if not args.no_safety:
                            ok, why = Z.check_plan_safe(eef_chunk, eef_now9)
                            if not ok:
                                print(f"  STOP loop: safety REJECT — {why}"); break
                        execute(arm, hand, eef_chunk, hand_chunk,
                                args.exec_steps, args.hz, args.tool_frame)
                except KeyboardInterrupt:
                    print("\n  loop stopped by user")

            else:
                print("  unknown command. Use q / e / l / x")
    finally:
        if live is not None:
            live.stop()
        teardown_cameras(rcc, dc, cam_source)
        try: arm.end(set_break=False)
        except Exception as e: print(f"[zerodex] arm.end err: {e}")
        try: hand.end()
        except Exception as e: print(f"[zerodex] hand.end err: {e}")
        try: policy.close()
        except Exception: pass
        print(f"\nQueries={qn} Executions={en}")


if __name__ == "__main__":
    main()
