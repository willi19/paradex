"""
Integrity check for collected teleop sessions.

Flags, per session:
  - videos that didn't come in / are corrupt (missing serial, 0 frames, tiny file)
  - robot data that's broken (missing, empty, NaN/inf, object/inhomogeneous dtype)
  - frames that don't line up:
      * spread of frame counts across cameras
      * the FOCUS pair (default 26053260 & 25305465) differing too much
        (they are expected to have ~equal length)
      * postprocessed arm/hand length vs the video frame count

Usage:
  python src/process/teleop_real/check_data.py --save_path capture/hri_vive/test
  python src/process/teleop_real/check_data.py --save_path capture/hri_vive/test --dest nas
  python src/process/teleop_real/check_data.py --save_path ... --focus 26053260 25305465 --tol 3
"""
import os
import glob
import argparse

import numpy as np

from paradex.utils.path import shared_dir, local_shared_dir

MIN_MP4_BYTES = 1_000_000


def video_frame_count(path):
    import cv2
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        return -1
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return n


def load_npy(path):
    """Return (array_or_None, error_str_or_None)."""
    if not os.path.exists(path):
        return None, "MISSING"
    try:
        a = np.load(path, allow_pickle=True)
    except Exception as exc:
        return None, f"unreadable ({exc})"
    if a.dtype == object:
        return a, "object/inhomogeneous dtype (corrupt)"
    if a.size == 0:
        return a, "empty"
    if not np.isfinite(a).all():
        return a, "contains NaN/inf"
    return a, None


def check_robot(sdir, subdir):
    """Check one robot stream's position.npy. Returns (length, list_of_issues)."""
    issues = []
    lengths = {}
    base = os.path.join(sdir, subdir)
    if not os.path.isdir(base):
        return None, [f"{subdir}/ MISSING"]
    # position is the key stream; also sanity-check time/action if present
    for name in ("position", "action", "time"):
        p = os.path.join(base, f"{name}.npy")
        if not os.path.exists(p):
            if name == "position":
                issues.append(f"{subdir}/position.npy MISSING")
            continue
        a, err = load_npy(p)
        if err:
            issues.append(f"{subdir}/{name}.npy: {err}")
        if a is not None and a.dtype != object and a.size > 0:
            lengths[name] = len(a)
    # streams within one robot dir should be equal length
    if len(set(lengths.values())) > 1:
        issues.append(f"{subdir}/ stream length mismatch: {lengths}")
    return (lengths.get("position"), issues)


def check_session(sdir, focus, tol):
    name = os.path.basename(sdir.rstrip("/"))
    issues = []

    # --- videos ---
    vids = sorted(glob.glob(os.path.join(sdir, "videos", "*.mp4")))
    counts = {}
    for v in vids:
        serial = os.path.splitext(os.path.basename(v))[0]
        size = os.path.getsize(v)
        n = video_frame_count(v)
        counts[serial] = n
        if size < MIN_MP4_BYTES:
            issues.append(f"video {serial}: tiny ({size//1024} KB) -> likely corrupt/partial")
        if n <= 0:
            issues.append(f"video {serial}: {n} frames -> unreadable/corrupt")
    if not vids:
        issues.append("NO videos collected")

    # --- frame spread across cameras ---
    good = {s: n for s, n in counts.items() if n > 0}
    if good:
        lo, hi = min(good.values()), max(good.values())
        if hi - lo > tol:
            issues.append(f"frame-count spread {hi - lo} (min={lo} max={hi}) over {tol}")

    # --- focus pair (should be ~equal length) ---
    fa, fb = focus
    if fa in counts and fb in counts:
        da, db = counts[fa], counts[fb]
        if da <= 0 or db <= 0:
            issues.append(f"FOCUS {fa}={da} {fb}={db}: one is unreadable")
        elif abs(da - db) > tol:
            issues.append(f"FOCUS mismatch {fa}={da} vs {fb}={db} (diff {abs(da-db)} > {tol})")
    else:
        missing = [s for s in focus if s not in counts]
        if missing:
            issues.append(f"FOCUS camera(s) missing from videos: {missing}")

    # --- robot data (postprocessed at session root; raw under raw/) ---
    arm_len, arm_iss = check_robot(sdir, "arm")
    hand_len, hand_iss = check_robot(sdir, "hand")
    issues += arm_iss + hand_iss
    # also verify raw exists (capture actually recorded)
    for rsub in ("raw/arm", "raw/hand"):
        _, riss = check_robot(sdir, rsub)
        issues += riss

    # --- alignment: postprocessed robot length vs video frame count ---
    med = int(np.median(list(good.values()))) if good else None
    for label, rlen in (("arm", arm_len), ("hand", hand_len)):
        if rlen is not None and med is not None and abs(rlen - med) > tol:
            issues.append(f"{label} length {rlen} != video frames ~{med} (diff {abs(rlen-med)} > {tol})")

    return name, counts, arm_len, hand_len, med, issues


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--save_path", required=True, help="e.g. capture/hri_vive/test")
    ap.add_argument("--dest", choices=["local", "nas"], default="local",
                    help='where the data lives: "local" (data2, default) or "nas".')
    ap.add_argument("--focus", nargs=2, default=["26053260", "25305465"],
                    help="two serials expected to have ~equal frame counts")
    ap.add_argument("--tol", type=int, default=3,
                    help="allowed frame-count difference before flagging (default 3)")
    ap.add_argument("--sessions", nargs="*", default=None,
                    help="limit to these session leaf names")
    args = ap.parse_args()

    root = local_shared_dir if args.dest == "local" else shared_dir
    base = os.path.join(root, args.save_path)
    if not os.path.isdir(base):
        print(f"[check] no data at {base}")
        return

    if args.sessions:
        sessions = [os.path.join(base, s) for s in args.sessions]
    else:
        sessions = sorted(d for d in glob.glob(os.path.join(base, "*")) if os.path.isdir(d))

    print(f"[check] {base}  ({len(sessions)} sessions)  focus={tuple(args.focus)}  tol={args.tol}\n")
    bad = []
    for sdir in sessions:
        name, counts, arm_len, hand_len, med, issues = check_session(sdir, args.focus, args.tol)
        fa, fb = args.focus
        focus_str = f"{fa}={counts.get(fa,'-')} {fb}={counts.get(fb,'-')}"
        status = "OK " if not issues else "!! "
        print(f"{status}{name}  vids={len(counts)} med_frames={med} "
              f"arm={arm_len} hand={hand_len} | {focus_str}")
        for i in issues:
            print(f"      - {i}")
        if issues:
            bad.append(name)

    print(f"\n[check] {len(bad)}/{len(sessions)} sessions with issues"
          + (f": {bad}" if bad else " (all clean)"))


if __name__ == "__main__":
    main()
