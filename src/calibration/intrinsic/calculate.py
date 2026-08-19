"""Single-camera intrinsic calibration from saved charuco keypoints.

Loads the latest keypoint file under ~/shared_data/intrinsic/<serial>/keypoint/,
runs cv2.calibrateCamera with per-frame masking of NaN entries, and writes the
result to ~/shared_data/intrinsic/<serial>/param/<timestamp>.json in the format
paradex.calibration.utils.load_current_intrinsic reads back
({K, distortion, RMS_error, width, height}).

Run this on the machine that can see ~/shared_data after capture.py finishes
(the Capture PCs have already written the keypoint files).
"""

import argparse
import os
import glob
import json
import time
import numpy as np
import cv2

from paradex.calibration.utils import intrinsic_dir
from paradex.image.aruco import _charuco_board_cache, get_charuco_detector


def load_object_points(b_id):
    get_charuco_detector()
    return _charuco_board_cache[b_id].getChessboardCorners().reshape(-1, 3).astype(np.float32)


def calibration_image_size(src, requested_size):
    """Resolve the raw-frame size, preferring capture metadata over defaults."""
    if requested_size is not None:
        return requested_size

    metadata_path = os.path.splitext(src)[0] + ".json"
    if os.path.exists(metadata_path):
        with open(metadata_path) as f:
            metadata = json.load(f)
        width = int(metadata["width"])
        height = int(metadata["height"])
        if width <= 0 or height <= 0:
            raise ValueError(f"invalid image size in {metadata_path}: {width}x{height}")
        return width, height

    # Keypoints captured before the daemon migration have no sidecar.  Preserve
    # their historical resolution, but make the assumption visible.
    print(f"WARN: {metadata_path} missing; assuming legacy 2048x1536. "
          "Pass --width/--height if the source images used another size.")
    return 2048, 1536


def calibrate_one(serial, b_id, requested_size, min_corners):
    base = os.path.join(intrinsic_dir, serial)
    files = sorted(glob.glob(os.path.join(base, "keypoint", "*.npy")))
    if not files:
        raise FileNotFoundError(f"no keypoint files under {base}/keypoint")
    src = files[-1]
    kpts = np.load(src)  # (N, M, 1, 2) possibly with NaN
    image_size = calibration_image_size(src, requested_size)
    print(f"[{serial}] loading {os.path.basename(src)}  shape={kpts.shape}")

    obj_full = load_object_points(b_id)
    n_corners_board = obj_full.shape[0]
    if kpts.shape[1] != n_corners_board:
        print(f"[{serial}] WARN: keypoint dim {kpts.shape[1]} != board {b_id} corners {n_corners_board}; "
              f"using min(...) for safety")
    n_corners = min(kpts.shape[1], n_corners_board)
    obj_full = obj_full[:n_corners]

    obj_list, img_list, used = [], [], 0
    for i in range(kpts.shape[0]):
        f = kpts[i, :n_corners, 0, :]
        valid = ~np.isnan(f[:, 0])
        if int(valid.sum()) < min_corners:
            continue
        obj_list.append(obj_full[valid].copy())
        img_list.append(f[valid].astype(np.float32))
        used += 1
    if used < 5:
        raise ValueError(f"[{serial}] only {used} usable frames after filtering "
                         f"(need >= 10 ideally). Recapture with more diverse poses.")
    print(f"[{serial}] using {used} frames (image size {image_size[0]}x{image_size[1]})")

    rms, K, dist, _, _ = cv2.calibrateCamera(
        obj_list, img_list, image_size, None, None,
    )
    print(f"[{serial}] RMS={rms:.4f} px  fx={K[0,0]:.1f} fy={K[1,1]:.1f}  "
          f"cx={K[0,2]:.1f} cy={K[1,2]:.1f}")
    print(f"[{serial}] dist={dist.ravel().tolist()}")

    sanity_warn(serial, K, dist, image_size)

    out = {
        "RMS_error": float(rms),
        "K": K.tolist(),
        "distortion": dist.ravel().tolist(),
        "width": image_size[0],
        "height": image_size[1],
    }
    name = time.strftime("%Y%m%d_%H%M%S", time.localtime()) + ".json"
    out_path = os.path.join(base, "param", name)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(out, f, indent=4)
    print(f"[{serial}] saved -> {out_path}")
    return rms, K, dist


def sanity_warn(serial, K, dist, image_size):
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]
    w, h = image_size
    issues = []
    if abs(fx - fy) / max(fx, fy) > 0.05:
        issues.append(f"fx vs fy differ by {abs(fx - fy):.0f}px ({100 * abs(fx - fy) / max(fx, fy):.1f}%) — square pixels expected")
    if abs(cx - w / 2) > w * 0.15:
        issues.append(f"cx={cx:.1f} far from image center {w/2:.0f}")
    if abs(cy - h / 2) > h * 0.15:
        issues.append(f"cy={cy:.1f} far from image center {h/2:.0f}")
    k3 = dist.ravel()[4] if dist.size >= 5 else 0.0
    if abs(k3) > 1.0:
        issues.append(f"|k3|={abs(k3):.2f} — distortion model likely diverged")
    if issues:
        print(f"[{serial}] sanity check WARN:")
        for s in issues:
            print(f"  - {s}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--serials", nargs="+", default=None,
                        help="camera serials to calibrate (default: every serial "
                             "that has a keypoint/ dir under ~/shared_data/intrinsic)")
    parser.add_argument("--board", default="3", help="charuco board id (default 3)")
    parser.add_argument("--width", type=int, default=None,
                        help="override raw image width (normally read from capture metadata)")
    parser.add_argument("--height", type=int, default=None,
                        help="override raw image height (normally read from capture metadata)")
    parser.add_argument("--min-corners", type=int, default=10,
                        help="minimum detected corners to use a frame (default 10)")
    args = parser.parse_args()

    serials = args.serials
    if not serials:
        serials = sorted(
            s for s in os.listdir(intrinsic_dir)
            if os.path.isdir(os.path.join(intrinsic_dir, s, "keypoint"))
        )
        if not serials:
            print(f"no serials with keypoint/ found under {intrinsic_dir}")
            return
        print(f"auto-detected serials: {serials}")

    if (args.width is None) != (args.height is None):
        parser.error("--width and --height must be supplied together")
    image_size = None if args.width is None else (args.width, args.height)
    failures = []
    for s in serials:
        try:
            calibrate_one(s, args.board, image_size, args.min_corners)
        except Exception as e:
            print(f"[{s}] FAILED: {e}")
            failures.append(s)
    if failures:
        print(f"\nfailed: {failures}")


if __name__ == "__main__":
    main()
