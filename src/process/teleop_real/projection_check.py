"""
Project the recorded robot trajectory onto the captured camera videos to
visually verify image <-> action time sync.

Why this exists:
    Without TimestampMonitor (raw/timestamps/ missing) there is no exact
    frame_id <-> pc_time bridge between the cameras and the robot logs.
    Videos are 30fps while arm/hand are sampled much faster (e.g. 555 video
    frames vs 1761 action samples here).  This tool maps each video frame to
    an action sample by wall-time, applies C2R, projects the robot mesh, and
    overlays it so you can SEE whether the mesh tracks the real robot.

Sync workflow:
    1. Run with --show. A window opens with two trackbars: frame + offset(ms).
    2. Scrub frames. If the green mesh leads/lags the real robot, drag the
       offset slider until it locks on. Read the offset -> that is your
       camera/action time skew.
    3. Re-run with --offset_ms <value> to bake it in / export a video.

Preconditions ({shared_dir}/{demo_path}/):
    videos/<serial>.avi          (raw .avi preferred; corrupt/0-frame skipped)
    cam_param/{intrinsics,extrinsics}.json
    C2R.npy
    arm/position.npy , arm/time.npy      (postprocess_session output)
    hand/position.npy, hand/time.npy
"""
import os
import json
import argparse
import numpy as np
import cv2
import tqdm

from paradex.utils.path import shared_dir
from paradex.calibration.utils import load_camparam, load_c2r
from paradex.image.image_dict import ImageDict
from paradex.image.merge import merge_image
from paradex.visualization.robot import RobotModule
from paradex.robot.utils import get_robot_urdf_path
from paradex.robot.inspire import parse_inspire


def open_valid_videos(video_dir, want_serials):
    """Open .avi (preferred) per serial, skipping corrupt / 0-frame files."""
    caps = {}
    for serial in want_serials:
        path = None
        for ext in (".avi", ".mp4"):
            p = os.path.join(video_dir, serial + ext)
            if os.path.exists(p):
                cap = cv2.VideoCapture(p)
                if int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) > 0:
                    path = p
                    caps[serial] = cap
                    break
                cap.release()
        if path is None:
            print(f"[skip] {serial}: no usable video (corrupt or missing)")
    return caps


def load_action(demo_dir, arm_name, hand_name):
    arm_pos = np.load(os.path.join(demo_dir, "arm", "position.npy"))
    arm_time = np.load(os.path.join(demo_dir, "arm", "time.npy"))
    hand_pos = np.load(os.path.join(demo_dir, "hand", "position.npy"))

    if hand_name == "inspire":
        hand_pos = parse_inspire(hand_pos)
    elif hand_name == "allegro":
        hand_pos = hand_pos.astype(np.float64)
    else:
        raise ValueError(f"Unsupported hand: {hand_name}")

    n = min(len(arm_pos), len(hand_pos), len(arm_time))
    cfg = np.concatenate([arm_pos[:n], hand_pos[:n]], axis=1)
    return cfg, arm_time[:n]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--demo_path", required=True,
                    help="under shared_dir, e.g. teleop_test/01/2026-05-15_17-11-47")
    ap.add_argument("--arm", default="xarm")
    ap.add_argument("--hand", default="inspire")
    ap.add_argument("--serial", default=None,
                    help="single camera serial. default: all valid cameras")
    ap.add_argument("--offset_ms", type=float, default=0.0,
                    help="camera->action time skew in ms (video frame time += offset)")
    ap.add_argument("--show", action="store_true",
                    help="interactive window with frame + offset trackbars")
    ap.add_argument("--out", default=None,
                    help="output merged mp4 path (default: {demo}/projection_check.mp4)")
    ap.add_argument("--stride", type=int, default=1,
                    help="process every Nth video frame (export mode)")
    args = ap.parse_args()

    demo_dir = os.path.join(shared_dir, args.demo_path)
    intr, extr = load_camparam(demo_dir)
    c2r = load_c2r(demo_dir)
    cfg, act_time = load_action(demo_dir, args.arm, args.hand)
    Na = len(cfg)

    # Start-time anchor: prefer camera_meta.json (main-PC wall clock at camera
    # trigger). Falls back to act_time[0] for old captures without the meta.
    meta_path = os.path.join(demo_dir, "camera_meta.json")
    if os.path.exists(meta_path):
        meta = json.load(open(meta_path))
        cam_anchor = float(meta["start_time"])
        if "offset_ms" in meta and args.offset_ms == 0.0:
            args.offset_ms = float(meta["offset_ms"])
            print(f"[offset] using saved offset_ms={args.offset_ms} from camera_meta.json")
        print(f"[anchor] camera_meta.json start_time={cam_anchor:.3f} "
              f"(action[0]={act_time[0]:.3f}, raw skew={cam_anchor-act_time[0]:+.3f}s)")
    else:
        meta = None
        cam_anchor = float(act_time[0])
        print("[anchor] no camera_meta.json -> assuming video[0] == action[0] "
              "(use --offset_ms to correct)")

    def save_offset(off_ms):
        m = meta if meta is not None else {}
        m["offset_ms"] = float(off_ms)
        with open(meta_path, "w") as f:
            json.dump(m, f)
        print(f"[offset] saved offset_ms={off_ms} -> {meta_path}")

    want = [args.serial] if args.serial else sorted(
        s for s in intr.keys()
        if os.path.exists(os.path.join(demo_dir, "videos", s + ".avi"))
        or os.path.exists(os.path.join(demo_dir, "videos", s + ".mp4"))
    )
    caps = open_valid_videos(os.path.join(demo_dir, "videos"), want)
    if not caps:
        raise FileNotFoundError(f"No usable videos in {demo_dir}/videos")

    keys = sorted(caps.keys())
    vfps = caps[keys[0]].get(cv2.CAP_PROP_FPS) or 30.0
    Nv = min(int(caps[k].get(cv2.CAP_PROP_FRAME_COUNT)) for k in keys)
    W = int(caps[keys[0]].get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(caps[keys[0]].get(cv2.CAP_PROP_FRAME_HEIGHT))

    # --- sync diagnostics ---
    dur_action = float(act_time[-1] - act_time[0])
    dur_video = Nv / vfps
    print(f"[sync] action: {Na} samples, {dur_action:.2f}s "
          f"({Na/max(dur_action,1e-9):.1f} Hz)")
    print(f"[sync] video : {Nv} frames @ {vfps:.1f}fps, {dur_video:.2f}s")
    print(f"[sync] duration diff (video-action) = {dur_video - dur_action:+.2f}s "
          f"-> if large, recording windows are misaligned")

    rm = RobotModule(get_robot_urdf_path(arm_name=args.arm, hand_name=args.hand))

    def action_index(vframe, offset_ms):
        # video frame i (blank-filled => uniform) -> main-PC wall time
        t = cam_anchor + vframe / vfps + offset_ms / 1000.0
        return int(np.clip(np.searchsorted(act_time, t), 0, Na - 1))

    def render(vframe, offset_ms):
        frames = {}
        for k in keys:
            caps[k].set(cv2.CAP_PROP_POS_FRAMES, vframe)
            ok, fr = caps[k].read()
            if ok:
                frames[k] = fr
        if not frames:
            return None
        ai = action_index(vframe, offset_ms)
        rm.update_cfg(cfg[ai])
        mesh = rm.get_robot_mesh()
        mesh.apply_transform(c2r)
        idict = ImageDict(frames,
                          {k: intr[k] for k in frames},
                          {k: extr[k] for k in frames}, path=None)
        ov = idict.project_mesh(mesh, color=(0, 255, 0)).images
        merged = merge_image(ov)
        ai_t = act_time[ai] - act_time[0]
        v_t = vframe / vfps + offset_ms / 1000.0
        cv2.putText(merged, f"vframe {vframe}/{Nv}  t_v={v_t:.2f}s  "
                            f"act[{ai}] t_a={ai_t:.2f}s  off={offset_ms:.0f}ms",
                    (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
        return merged

    if args.show:
        win = "projection sync check"
        cv2.namedWindow(win, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(win, 1600, 900)
        state = {"f": 0, "off": int(args.offset_ms), "play": False}
        cv2.createTrackbar("frame", win, 0, max(Nv - 1, 1),
                           lambda v: state.update(f=v))
        # offset slider: 0..4000 maps to -2000..+2000 ms
        cv2.createTrackbar("offset+2000ms", win, int(args.offset_ms) + 2000, 4000,
                           lambda v: state.update(off=v - 2000))
        print("keys: n/./d=next  p/,/a=prev  PgUp/PgDn=+-10  "
              "[ ]=offset -/+10ms  w=save offset  space=play/pause  q/ESC=quit")
        last = None
        while True:
            cur = (state["f"], state["off"])
            if cur != last:
                img = render(state["f"], float(state["off"]))
                if img is not None:
                    cv2.imshow(win, img)
                last = cur

            k = cv2.waitKey(30 if state["play"] else 0) & 0xFF
            if k in (27, ord("q")):
                break
            elif k in (ord("n"), ord("."), ord("d"), 83):      # next (83=Right)
                state["f"] = min(state["f"] + 1, Nv - 1)
            elif k in (ord("p"), ord(","), ord("a"), 81):      # prev (81=Left)
                state["f"] = max(state["f"] - 1, 0)
            elif k in (85,):                                    # PageUp +10
                state["f"] = min(state["f"] + 10, Nv - 1)
            elif k in (84,):                                    # PageDown -10
                state["f"] = max(state["f"] - 10, 0)
            elif k == ord("]"):
                state["off"] += 10
            elif k == ord("["):
                state["off"] -= 10
            elif k == ord("w"):
                save_offset(state["off"])
            elif k == ord(" "):
                state["play"] = not state["play"]
            elif state["play"]:
                state["f"] = state["f"] + 1
                if state["f"] >= Nv:
                    state["f"], state["play"] = Nv - 1, False

            cv2.setTrackbarPos("frame", win, state["f"])
            cv2.setTrackbarPos("offset+2000ms", win,
                               int(np.clip(state["off"] + 2000, 0, 4000)))
        cv2.destroyAllWindows()
    else:
        out_path = args.out or os.path.join(demo_dir, "projection_check.mp4")
        vw = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"),
                             vfps / args.stride, (W, H))
        for vf in tqdm.tqdm(range(0, Nv, args.stride)):
            img = render(vf, args.offset_ms)
            if img is None:
                break
            vw.write(cv2.resize(img, (W, H)))
        vw.release()
        print(f"[done] {out_path}")

    for c in caps.values():
        c.release()


if __name__ == "__main__":
    main()
