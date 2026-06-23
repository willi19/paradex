"""
Per scene under base_dir/<object>/<scene>/:

Input:
    video_extracted/<cam>/*.jpg
    sam_object_result/<cam>/*.png   (optional, mask per-frame by filename stem)

Trim:
    - drop first `front_cut` frames (default 3)
    - drop tail so all cams have same length (min across cams)

Frame drop replacement:
    - compute mean pixel value of each frame
    - if mean <= drop_thresh → replace with previous good frame (image AND mask)

Output (kept alongside originals; originals are NOT removed):
    frames/<cam>/00000.jpg, 00001.jpg, ...
    obj_mask/<cam>/00000.png, ...        (only for frames that have a mask)
    vid/<cam>.mp4                         (fps 30 by default)
    processed/timestamps/frame_id.npy     (1..N, int64)
    processed/timestamps/timestamp.npy    (front-trimmed, length N)
"""

import argparse
import os
from multiprocessing import Pool
import cv2
import numpy as np

# each worker already runs in its own process — keep OpenCV single-threaded
# inside each process to avoid oversubscribing CPU cores.
cv2.setNumThreads(1)


def _worker_init():
    cv2.setNumThreads(1)


def list_files(folder):
    return sorted(f for f in os.listdir(folder) if not f.startswith("."))


def list_dirs(folder):
    return sorted(d for d in os.listdir(folder) if os.path.isdir(os.path.join(folder, d)))


def process_cam(
    cam: str,
    ve_dir: str,
    sam_dir: str,
    out_dir: str,
    keep_start: int,
    keep_end: int,
    drop_thresh: float,
    fps: int,
) -> int:
    """Process one cam. Returns number of frames replaced due to drop detection."""
    src_files = list_files(os.path.join(ve_dir, cam))[keep_start:keep_end]

    out_frames = os.path.join(out_dir, "frames", cam)
    out_vid_dir = os.path.join(out_dir, "vid")
    os.makedirs(out_frames, exist_ok=True)
    os.makedirs(out_vid_dir, exist_ok=True)

    has_sam = sam_dir is not None and os.path.isdir(os.path.join(sam_dir, cam))
    mask_map = {}  # stem -> filename in sam/<cam>/
    out_mask_dir = None
    if has_sam:
        mask_src_dir = os.path.join(sam_dir, cam)
        out_mask_dir = os.path.join(out_dir, "obj_mask", cam)
        os.makedirs(out_mask_dir, exist_ok=True)
        for mf in os.listdir(mask_src_dir):
            mask_map[os.path.splitext(mf)[0]] = mf

    writer = None
    prev_img = None
    prev_mask = None
    prev_mask_ext = None
    drops = 0

    for new_idx, src_name in enumerate(src_files):
        src_path = os.path.join(ve_dir, cam, src_name)
        img = cv2.imread(src_path)
        is_drop = (img is None) or (float(img.mean()) <= drop_thresh)

        # Decide current image + mask
        if is_drop and prev_img is not None:
            img_out = prev_img
            mask_out = prev_mask
            mask_ext = prev_mask_ext
            drops += 1
        else:
            img_out = img
            mask_out = None
            mask_ext = None
            if has_sam:
                stem = os.path.splitext(src_name)[0]
                mask_fname = mask_map.get(stem)
                if mask_fname:
                    mask_path = os.path.join(sam_dir, cam, mask_fname)
                    mask_out = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)
                    mask_ext = os.path.splitext(mask_fname)[1]

        if img_out is None:
            # nothing to write (first frame was drop with no prev and unreadable)
            continue

        # Save frame
        cv2.imwrite(
            os.path.join(out_frames, f"{new_idx:05d}.jpg"),
            img_out,
            [cv2.IMWRITE_JPEG_QUALITY, 95],
        )

        # Lazy init of video writer
        if writer is None:
            h, w = img_out.shape[:2]
            writer = cv2.VideoWriter(
                os.path.join(out_vid_dir, f"{cam}.mp4"),
                cv2.VideoWriter_fourcc(*"mp4v"),
                fps,
                (w, h),
            )
        writer.write(img_out)

        # Save mask if available for this frame
        if has_sam and mask_out is not None:
            ext = mask_ext or ".png"
            cv2.imwrite(os.path.join(out_mask_dir, f"{new_idx:05d}{ext}"), mask_out)

        prev_img = img_out
        prev_mask = mask_out
        prev_mask_ext = mask_ext

    if writer is not None:
        writer.release()

    return drops


def _cam_worker(args):
    return process_cam(*args)


def _write_timestamps(scene_dir: str, front_cut: int, new_n: int):
    """Write trimmed timestamps/ (frame_id restarted from 1, timestamp front-trimmed)."""
    raw_ts_dir = os.path.join(scene_dir, "raw", "timestamps")
    ts_path = os.path.join(raw_ts_dir, "timestamp.npy")
    out_ts_dir = os.path.join(scene_dir, "processed", "timestamps")
    os.makedirs(out_ts_dir, exist_ok=True)

    # frame_id: sequential 1..N
    frame_id_new = np.arange(1, new_n + 1, dtype=np.int64)
    np.save(os.path.join(out_ts_dir, "frame_id.npy"), frame_id_new)

    # timestamp: drop first front_cut, clip to new_n
    if os.path.exists(ts_path):
        ts = np.load(ts_path)
        if len(ts) < front_cut + new_n:
            print(f"  ⚠ timestamp.npy len={len(ts)} < front_cut+new_n={front_cut + new_n}, clipping")
        ts_new = ts[front_cut : front_cut + new_n]
        np.save(os.path.join(out_ts_dir, "timestamp.npy"), ts_new)
    else:
        print(f"  ⚠ no raw/timestamps/timestamp.npy in {scene_dir}")


def process_scene(
    scene_dir: str,
    front_cut: int,
    warn_diff: int,
    drop_thresh: float,
    fps: int,
    dry_run: bool,
    workers: int = 1,
    hand: bool = False,
):
    ve_dir = os.path.join(scene_dir, "video_extracted")
    sam_dir = os.path.join(scene_dir, "sam_object_result")

    if not os.path.isdir(ve_dir):
        print(f"[skip] no video_extracted: {scene_dir}")
        return

    cams = list_dirs(ve_dir)
    if not cams:
        print(f"[skip] no cams: {scene_dir}")
        return

    counts = {c: len(list_files(os.path.join(ve_dir, c))) for c in cams}
    min_c = min(counts.values())
    max_c = max(counts.values())

    if min_c <= front_cut:
        print(f"[skip] {scene_dir} | min={min_c} max={max_c} (min<=front_cut)")
        return

    keep_start = front_cut
    keep_end = min_c
    new_n = keep_end - keep_start
    warn = f" ⚠spread={max_c - min_c}" if max_c - min_c > warn_diff else ""
    tag = "dry" if dry_run else "run"
    print(f"[{tag}] {scene_dir} | cams={len(cams)} min={min_c} max={max_c} -> new={new_n}{warn}")

    if dry_run:
        return

    # Trim raw/timestamps to match new frame count (skipped in --hand mode)
    if not hand:
        _write_timestamps(scene_dir, keep_start, new_n)

    sam_in = sam_dir if os.path.isdir(sam_dir) else None
    job_args = [
        (c, ve_dir, sam_in, scene_dir, keep_start, keep_end, drop_thresh, fps)
        for c in cams
    ]
    if workers > 1:
        with Pool(min(workers, len(job_args)), initializer=_worker_init) as pool:
            drops_list = list(pool.imap_unordered(_cam_worker, job_args))
    else:
        drops_list = [_cam_worker(a) for a in job_args]
    total_drops = sum(drops_list)
    if total_drops:
        print(f"  replaced {total_drops} frame(s) via drop_thresh={drop_thresh}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-dir", required=True)
    ap.add_argument("--front-cut", type=int, default=3)
    ap.add_argument("--warn-diff", type=int, default=10)
    ap.add_argument("--drop-thresh", type=float, default=70.0,
                    help="replace frame with previous if img.mean() <= this")
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--workers", type=int, default=min(24, os.cpu_count() or 1),
                    help="parallel cam workers per scene")
    ap.add_argument("--hand", action="store_true",
                    help="hand capture mode: skip timestamp trimming")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    for obj in list_dirs(args.base_dir):
        obj_dir = os.path.join(args.base_dir, obj)
        for scene in list_dirs(obj_dir):
            process_scene(
                os.path.join(obj_dir, scene),
                front_cut=args.front_cut,
                warn_diff=args.warn_diff,
                drop_thresh=args.drop_thresh,
                fps=args.fps,
                dry_run=args.dry_run,
                workers=args.workers,
                hand=args.hand,
            )


if __name__ == "__main__":
    main()
