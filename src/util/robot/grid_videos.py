import argparse
import glob
import os
from typing import List, Tuple

import cv2
import numpy as np


def open_caps(paths: List[str]) -> Tuple[List[cv2.VideoCapture], float, Tuple[int, int]]:
    caps = []
    fps = 10.0
    size = (640, 480)
    if len(paths) == 0:
        return caps, fps, size
    for p in paths:
        cap = cv2.VideoCapture(p)
        if not cap.isOpened():
            raise RuntimeError(f"Failed to open video: {p}")
        caps.append(cap)
    fps = caps[0].get(cv2.CAP_PROP_FPS) or fps
    w = int(caps[0].get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(caps[0].get(cv2.CAP_PROP_FRAME_HEIGHT))
    size = (w, h)
    return caps, fps, size


def read_frame_or_hold(cap: cv2.VideoCapture, hold: np.ndarray) -> Tuple[np.ndarray, np.ndarray, bool]:
    ok, frame = cap.read()
    if ok:
        return frame, frame, True
    return hold, hold, False


def build_grid_videos(
    video_paths: List[str],
    output_path: str,
    grid_rows: int,
    grid_cols: int,
    scale: float = 1.0,
) -> None:
    # Open all videos and determine output properties
    caps, fps, (w, h) = open_caps(video_paths)
    if len(caps) == 0:
        raise RuntimeError("No videos to process.")

    # Pad with None to fill grid slots
    total_slots = grid_rows * grid_cols
    while len(caps) < total_slots:
        caps.append(None)

    w = int(w * scale)
    h = int(h * scale)
    grid_w = w * grid_cols
    grid_h = h * grid_rows

    fourcc = cv2.VideoWriter_fourcc(*"XVID")  # more permissive than mp4v
    writer = cv2.VideoWriter(output_path, fourcc, fps, (grid_w, grid_h))

    # Track whether each video is still running
    holds = []
    for cap in caps:
        if cap is None:
            holds.append(np.zeros((h, w, 3), dtype=np.uint8))
        else:
            ok, frame = cap.read()
            if not ok:
                frame = np.zeros((h, w, 3), dtype=np.uint8)
            holds.append(frame)

    running = True
    while running:
        running = False
        grid_rows_frames = []
        for r in range(grid_rows):
            row_frames = []
            for c in range(grid_cols):
                idx = r * grid_cols + c
                cap = caps[idx]
                hold = holds[idx]
                if cap is None:
                    frame = hold
                else:
                    frame, holds[idx], alive = read_frame_or_hold(cap, hold)
                    running = running or alive
                if scale != 1.0:
                    frame = cv2.resize(frame, (w, h))
                row_frames.append(frame)
            grid_rows_frames.append(np.concatenate(row_frames, axis=1))
        grid_frame = np.concatenate(grid_rows_frames, axis=0)
        writer.write(grid_frame)

    writer.release()
    for cap in caps:
        if cap is not None:
            cap.release()


def main():
    parser = argparse.ArgumentParser(
        description="Create grid videos from multiple overlay/mask videos."
    )
    parser.add_argument(
        "--input-dir",
        type=str,
        default="tmp/robot_projection/blue_speaker/3",
        help="Directory containing *_overlay.mp4 and *_mask.mp4",
    )
    parser.add_argument(
        "--rows", type=int, default=4, help="Grid rows"
    )
    parser.add_argument(
        "--cols", type=int, default=6, help="Grid cols"
    )
    parser.add_argument(
        "--output-prefix",
        type=str,
        default="grid",
        help="Prefix for output files",
    )
    parser.add_argument(
        "--scale",
        type=float,
        default=1.0,
        help="Resize factor applied to each video before tiling (e.g., 0.5 to halve resolution).",
    )
    args = parser.parse_args()

    overlay_paths = sorted(glob.glob(os.path.join(args.input_dir, "*_overlay.mp4")))
    mask_paths = sorted(glob.glob(os.path.join(args.input_dir, "*_mask.mp4")))

    if len(overlay_paths) == 0 and len(mask_paths) == 0:
        raise RuntimeError("No overlay or mask videos found.")

    if len(overlay_paths) > 0:
        out_overlay = os.path.join(args.input_dir, f"{args.output_prefix}_overlay_grid.mp4")
        build_grid_videos(overlay_paths, out_overlay, args.rows, args.cols, scale=args.scale)
        print(f"Saved overlay grid to {out_overlay}")

    if len(mask_paths) > 0:
        out_mask = os.path.join(args.input_dir, f"{args.output_prefix}_mask_grid.mp4")
        build_grid_videos(mask_paths, out_mask, args.rows, args.cols, scale=args.scale)
        print(f"Saved mask grid to {out_mask}")


if __name__ == "__main__":
    main()
