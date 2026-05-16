"""
Project the recorded robot trajectory onto each camera view and save overlay videos.

Preconditions (everything under {shared_dir}/{demo_path}/):
    videos/<serial>.{avi,mp4}        - per-camera videos
    cam_param/                       - intrinsics + extrinsics (load_camparam)
    handeye_calibration/             - camera-to-robot transform (load_c2r)
    arm/position.npy                 - synced robot arm joint angles  (postprocess_session)
    hand/position.npy                - synced hand state              (postprocess_session)

Outputs:
    overlay/<serial>.avi             - per-camera overlay video
    overlay/merged.mp4               - all cameras tiled into one video
"""
import os
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
from paradex.video.util import convert_avi_to_mp4


def _open_videos(video_dir):
    caps = {}
    for fname in os.listdir(video_dir):
        stem, ext = os.path.splitext(fname)
        if ext.lower() not in [".avi", ".mp4", ".mov"]:
            continue
        cap = cv2.VideoCapture(os.path.join(video_dir, fname))
        n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if n == 0:
            cap.release()
            continue
        caps[stem] = cap
    return caps


def overlay(demo_dir, arm_name, hand_name, color=(0, 255, 0)):
    video_dir = os.path.join(demo_dir, "videos")
    if not os.path.isdir(video_dir):
        raise FileNotFoundError(
            f"No videos/ at {demo_dir}. Enable camera recording first "
            f"(camera=True in CaptureSession)."
        )

    arm_path = os.path.join(demo_dir, "arm", "position.npy")
    hand_path = os.path.join(demo_dir, "hand", "position.npy")
    for p in (arm_path, hand_path):
        if not os.path.exists(p):
            raise FileNotFoundError(
                f"Missing {p}. Run postprocess_session() first."
            )

    intrinsics, extrinsics = load_camparam(demo_dir)
    c2r = load_c2r(demo_dir)

    rm = RobotModule(get_robot_urdf_path(arm_name=arm_name, hand_name=hand_name))
    arm_state = np.load(arm_path)
    hand_state = np.load(hand_path)
    if hand_name == "inspire":
        hand_state = parse_inspire(hand_state)

    assert len(arm_state) == len(hand_state), \
        f"frame count mismatch arm={len(arm_state)} hand={len(hand_state)}"

    caps = _open_videos(video_dir)
    if not caps:
        raise FileNotFoundError(f"No usable videos under {video_dir}")

    keys = sorted(caps.keys())
    ref = caps[keys[0]]
    W = int(ref.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(ref.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = ref.get(cv2.CAP_PROP_FPS) or 30.0
    max_n = max(int(c.get(cv2.CAP_PROP_FRAME_COUNT)) for c in caps.values())
    n_frames = min(max_n, len(arm_state))
    print(f"[overlay] {len(caps)} cameras, {n_frames} frames, {W}x{H}@{fps:.1f}fps")

    out_dir = os.path.join(demo_dir, "overlay")
    os.makedirs(out_dir, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"MJPG")
    writers = {
        k: cv2.VideoWriter(os.path.join(out_dir, f"{k}.avi"), fourcc, fps, (W, H))
        for k in keys
    }
    merged_writer = cv2.VideoWriter(
        os.path.join(out_dir, "merged.avi"), fourcc, fps, (W, H)
    )

    open_flag = {k: True for k in keys}
    try:
        for idx in tqdm.tqdm(range(n_frames)):
            frame_dict = {}
            for k in keys:
                if not open_flag[k]:
                    continue
                ret, frame = caps[k].read()
                if not ret:
                    open_flag[k] = False
                    continue
                frame_dict[k] = frame
            if not frame_dict:
                break

            part_int = {k: intrinsics[k] for k in frame_dict if k in intrinsics}
            part_ext = {k: extrinsics[k] for k in frame_dict if k in extrinsics}
            if not part_int:
                continue

            imgdict = ImageDict(frame_dict, part_int, part_ext, path=None)

            rm.update_cfg(np.concatenate([arm_state[idx], hand_state[idx]]))
            mesh = rm.get_robot_mesh()
            mesh.apply_transform(c2r)

            overlayed = imgdict.project_mesh(mesh, color=color)

            for k in keys:
                if open_flag[k] and k in overlayed.images:
                    writers[k].write(overlayed.images[k])

            merged = merge_image(overlayed.images)
            merged = cv2.resize(merged, (W, H))
            merged_writer.write(merged)
    finally:
        for c in caps.values():
            c.release()
        for w in writers.values():
            w.release()
        merged_writer.release()

    merged_avi = os.path.join(out_dir, "merged.avi")
    merged_mp4 = os.path.join(out_dir, "merged.mp4")
    convert_avi_to_mp4(merged_avi, merged_mp4)
    print(f"[overlay] saved per-camera overlays + merged.mp4 to {out_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--demo_path", type=str, required=True,
                        help="path under shared_dir, e.g. teleop_test/01/2026-05-15_14-23-00")
    parser.add_argument("--arm", type=str, default="xarm")
    parser.add_argument("--hand", type=str, default="inspire")
    parser.add_argument("--color", type=int, nargs=3, default=[0, 255, 0],
                        help="BGR color for projected mesh")
    args = parser.parse_args()

    demo_dir = os.path.join(shared_dir, args.demo_path)
    overlay(demo_dir, args.arm, args.hand, color=tuple(args.color))
