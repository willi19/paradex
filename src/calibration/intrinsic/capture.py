"""Single-camera intrinsic calibration capture.

Live preview with charuco overlay. Press 'c' to capture, 'q' to finish.
A coverage heatmap shows where on the image plane corners have already
been captured, so you can deliberately fill gaps (corners, edges) — this
is the main driver of intrinsic quality.

Output: ~/shared_data/intrinsic/<serial>/keypoint/<timestamp>.npy
        shape (N, num_corners, 1, 2)
        Optionally also raw images under <serial>/images/<timestamp>/
"""

import argparse
import os
import time
import numpy as np
import cv2

from paradex.calibration.utils import intrinsic_dir
from paradex.io.camera_system.pyspin import load_camera
from paradex.image.aruco import (
    detect_charuco, _charuco_board_cache, get_charuco_detector, draw_charuco,
    boardinfo_dict,
)


def n_corners_of_board(b_id):
    get_charuco_detector()
    board = _charuco_board_cache[b_id]
    return board.getChessboardCorners().reshape(-1, 3).shape[0]


def fixed_corner_array(detection, b_id, n_corners):
    """Return (n_corners, 2) with NaN for undetected corners, or None if zero detections."""
    if b_id not in detection:
        return None
    ids = detection[b_id]["checkerIDs"]
    cor = detection[b_id]["checkerCorner"]
    if len(ids) == 0:
        return None
    out = np.full((n_corners, 2), np.nan, dtype=np.float32)
    out[ids] = cor
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--serial", required=True, help="camera serial number")
    parser.add_argument("--board", default="3", help="charuco board id from charuco_info.json (default 3)")
    parser.add_argument("--min-fraction", type=float, default=0.6,
                        help="minimum fraction of board corners that must be detected to accept capture (default 0.6)")
    parser.add_argument("--save-images", action="store_true",
                        help="also save raw .png per capture (slower, more disk)")
    args = parser.parse_args()

    n_corners = n_corners_of_board(args.board)
    min_count = max(6, int(np.ceil(args.min_fraction * n_corners)))
    print(f"board {args.board}: {n_corners} corners total | accept frame if >= {min_count} detected")

    cam = load_camera(args.serial)
    cam.start("continuous", False, frame_rate=30)

    timestamp = time.strftime("%Y%m%d_%H%M%S", time.localtime())
    base = os.path.join(intrinsic_dir, args.serial)
    os.makedirs(os.path.join(base, "keypoint"), exist_ok=True)
    image_dir = None
    if args.save_images:
        image_dir = os.path.join(base, "images", timestamp)
        os.makedirs(image_dir, exist_ok=True)

    captured = []
    coverage = None
    win = f"intrinsic capture {args.serial}"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win, 1024, 768)

    swap_rb = True  # toggle with 'r' if colors look wrong

    try:
        while True:
            frame, _ = cam.get_image()
            if frame is None:
                continue
            if frame.ndim == 2:
                bgr = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
            elif frame.shape[2] == 3 and swap_rb:
                bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            else:
                bgr = frame.copy()

            if coverage is None:
                coverage = np.zeros((bgr.shape[0], bgr.shape[1]), dtype=np.uint16)

            detection = detect_charuco(bgr)
            cor = fixed_corner_array(detection, args.board, n_corners)
            valid_count = 0 if cor is None else int(np.sum(~np.isnan(cor[:, 0])))

            disp = bgr.copy()
            heat = np.minimum(coverage, 5).astype(np.uint8) * (255 // 5)
            heat_bgr = cv2.applyColorMap(heat, cv2.COLORMAP_HOT)
            disp = cv2.addWeighted(disp, 0.7, heat_bgr, 0.3, 0)

            board_colors = {"1": (255, 0, 255), "2": (0, 255, 255),
                            "3": (0, 255, 0), "4": (255, 128, 0)}
            for det_b_id, det_data in detection.items():
                pts = det_data["checkerCorner"]
                color = board_colors.get(det_b_id, (255, 255, 255))
                if det_b_id == args.board:
                    draw_charuco(disp, pts, color=(0, 255, 0), radius=5, thickness=-1)
                else:
                    draw_charuco(disp, pts, color=color, radius=3, thickness=-1)

            cv2.putText(disp, f"captured: {len(captured)}  using board {args.board}: {valid_count}/{n_corners}",
                        (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
            cv2.putText(disp, f"[c] capture  [r] swap_rb({swap_rb})  [q] save & quit",
                        (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            y = 120
            for b_id in boardinfo_dict.keys():
                cnt = len(detection.get(b_id, {}).get("checkerIDs", []))
                marker = " <- selected" if b_id == args.board else ""
                cv2.putText(disp, f"board {b_id}: {cnt} corners{marker}",
                            (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, board_colors.get(b_id, (255, 255, 255)), 2)
                y += 30

            if valid_count < min_count:
                cv2.putText(disp, "below threshold", (20, y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

            cv2.imshow(win, disp)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('r'):
                swap_rb = not swap_rb
                print(f"swap_rb -> {swap_rb}")
            elif key == ord('c'):
                if cor is None or valid_count < min_count:
                    print(f"rejected: only {valid_count}/{n_corners} detected (need >= {min_count})")
                    continue
                captured.append(cor)
                pts = cor[~np.isnan(cor[:, 0])].astype(int)
                for x, y in pts:
                    if 0 <= y < coverage.shape[0] and 0 <= x < coverage.shape[1]:
                        cv2.circle(coverage, (x, y), 30, 1, -1)
                if image_dir is not None:
                    cv2.imwrite(os.path.join(image_dir, f"{len(captured)-1:04d}.png"), bgr)
                print(f"captured #{len(captured)} ({valid_count}/{n_corners} corners)")
    finally:
        cam.stop()
        try:
            cam.release()
        except Exception:
            pass
        cv2.destroyAllWindows()

    if not captured:
        print("nothing to save — exiting")
        return

    arr = np.stack(captured, axis=0)[:, :, None, :]
    out_path = os.path.join(base, "keypoint", f"{timestamp}.npy")
    np.save(out_path, arr)
    print(f"saved {len(captured)} frames -> {out_path}  shape={arr.shape}")


if __name__ == "__main__":
    main()
