"""
Stage 1 of the prompt-labeling pipeline.

For every session under save_path:
  - run the validity check (check_data logic, tol) and write check.json
  - if valid, extract the FIRST and LAST frame of the 26053260 video and a
    side-by-side composite (260_firstlast.jpg) for later object/movement review.

Writes into each session dir:
  check.json                      {valid, reason, frames{...}, med_frames, ...}
  _frames/260_first.jpg
  _frames/260_last.jpg
  _frames/260_firstlast.jpg       (first | last, for the VLM to compare)

Usage:
  python src/process/teleop_real/prep_prompts.py --save_path capture/hri_vive/test --tol 15
"""
import os
import sys
import glob
import json
import argparse

import numpy as np
import cv2

from paradex.utils.path import shared_dir, local_shared_dir

sys.path.insert(0, os.path.dirname(__file__))
from check_data import check_session   # noqa: E402  (sibling module)

FOCUS_DEFAULT = ["26053260", "25305465"]
COMPOSITE_W = 1280   # downscaled composite width (keeps review images light)


def extract_first_last(video, out_first, out_last):
    cap = cv2.VideoCapture(video)
    if not cap.isOpened():
        return None, None
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    ok, first = cap.read()
    first = first if ok else None
    # seek near the end, then read forward to the true last decodable frame
    cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, n - 3))
    last = None
    while True:
        ok, fr = cap.read()
        if not ok:
            break
        last = fr
    cap.release()
    if first is not None:
        cv2.imwrite(out_first, first)
    if last is not None:
        cv2.imwrite(out_last, last)
    return first, last


def make_composite(first, last, out_path):
    if first is None or last is None:
        return False
    h = min(first.shape[0], last.shape[0])

    def rs(img):
        return cv2.resize(img, (int(img.shape[1] * h / img.shape[0]), h))
    first, last = rs(first), rs(last)
    pad = np.full((h, 8, 3), 255, np.uint8)
    comp = np.hstack([first, pad, last])
    scale = COMPOSITE_W / comp.shape[1]
    comp = cv2.resize(comp, (COMPOSITE_W, int(comp.shape[0] * scale)))
    # label first/last
    cv2.putText(comp, "FIRST", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
    cv2.putText(comp, "LAST", (comp.shape[1] // 2 + 10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
    cv2.imwrite(out_path, comp)
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--save_path", required=True)
    ap.add_argument("--dest", choices=["local", "nas"], default="local")
    ap.add_argument("--focus", nargs=2, default=FOCUS_DEFAULT)
    ap.add_argument("--tol", type=int, default=15)
    ap.add_argument("--frames-serial", default="26053260",
                    help="which camera's first/last frame to extract for review")
    args = ap.parse_args()

    root = local_shared_dir if args.dest == "local" else shared_dir
    base = os.path.join(root, args.save_path)
    sessions = sorted(d for d in glob.glob(os.path.join(base, "*")) if os.path.isdir(d))
    print(f"[prep] {base}: {len(sessions)} sessions, tol={args.tol}")

    n_valid = 0
    n_frames = 0
    for sdir in sessions:
        name, counts, arm_len, hand_len, med, issues = check_session(sdir, args.focus, args.tol)
        has_vid = len(counts) > 0
        valid = has_vid and len(issues) == 0
        rec = {
            "session": name,
            "valid": valid,
            "reason": issues,
            "n_videos": len(counts),
            "frame_counts": counts,
            "med_frames": med,
            "arm_len": arm_len,
            "hand_len": hand_len,
        }
        with open(os.path.join(sdir, "check.json"), "w") as f:
            json.dump(rec, f, indent=2)
        if valid:
            n_valid += 1
        # extract frames for any session that HAS the review camera (valid or not,
        # so borderline ones can still be eyeballed) -- but only when readable.
        vid = os.path.join(sdir, "videos", f"{args.frames_serial}.mp4")
        if os.path.exists(vid) and counts.get(args.frames_serial, 0) > 0:
            fdir = os.path.join(sdir, "_frames")
            os.makedirs(fdir, exist_ok=True)
            first, last = extract_first_last(
                vid, os.path.join(fdir, f"{args.frames_serial}_first.jpg"),
                os.path.join(fdir, f"{args.frames_serial}_last.jpg"))
            if make_composite(first, last, os.path.join(fdir, f"{args.frames_serial}_firstlast.jpg")):
                n_frames += 1

    print(f"[prep] valid(tol={args.tol}): {n_valid}/{len(sessions)}   "
          f"frame composites written: {n_frames}")


if __name__ == "__main__":
    main()
