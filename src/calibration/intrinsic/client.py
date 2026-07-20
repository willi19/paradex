"""Capture-PC side of distributed intrinsic calibration.

Continuously reads each local camera from shared memory, detects the charuco
board, and AUTO-accumulates a frame's corners whenever the board pose is
sufficiently novel vs. the frames already kept for that camera. There is no
save button — this mirrors the original design intent:

    "Detect charuco, compared with previous charucos, if diff > thres save;
     send current charuco to main pc."

Streams a downscaled preview + current corners + per-camera kept count to the
Main PC (capture.py). On the 'exit' command it writes one keypoint file per
camera:

    ~/shared_data/intrinsic/<serial>/keypoint/<timestamp>.npy
    shape (N, num_corners, 1, 2), NaN for undetected corners

— exactly the format calculate.py consumes.

Coupled to capture.py by data types ('image' downscaled /8, 'charuco_detection'
/8 float32) and ports (1234 publish, 6890 command). Change both together.
"""

import time
import os
from threading import Event

import numpy as np
import cv2

from paradex.io.camera_system.camera_reader import MultiCameraReader
from paradex.io.capture_pc.data_sender import DataPublisher
from paradex.io.capture_pc.command_sender import CommandReceiver
from paradex.image.aruco import (
    detect_charuco, get_charuco_detector, _charuco_board_cache,
)
from paradex.calibration.utils import intrinsic_dir

# --- capture policy (matches the proven should_save from the old distributed version) ---
BOARD = "3"          # charuco board id from charuco_info.json (11x8 -> 70 corners)
NOVELTY_PX = 10.0    # accept only if mean per-corner displacement to every kept frame >= this
MAX_FRAMES = 60      # stop accumulating for a camera once it has this many frames
# Only full-board detections (all n_corners) are kept, so every stored frame is a
# clean, corner-aligned (n_corners, 2) array and the per-corner distance is well-defined.


def n_corners_of_board(b_id):
    get_charuco_detector()
    return _charuco_board_cache[b_id].getChessboardCorners().reshape(-1, 3).shape[0]


def fixed_corner_array(detection, b_id, n_corners):
    """(n_corners, 2) with NaN for undetected corners, or None if zero detections."""
    if b_id not in detection:
        return None
    ids = detection[b_id]["checkerIDs"]
    cor = detection[b_id]["checkerCorner"]
    if len(ids) == 0:
        return None
    out = np.full((n_corners, 2), np.nan, dtype=np.float32)
    out[ids] = cor
    return out


n_corners = n_corners_of_board(BOARD)
print(f"intrinsic client: board {BOARD} -> {n_corners} corners | "
      f"accept full board only, mean corner shift >= {NOVELTY_PX}px from kept (max {MAX_FRAMES})")

dp = DataPublisher(port=1234, name="camera_stream")

exit_event = Event()
cr = CommandReceiver(event_dict={"exit": exit_event}, port=6890)

reader = MultiCameraReader()
last_frame_ids = {name: 0 for name in reader.camera_names}
kept = {name: [] for name in reader.camera_names}  # list of full-board (n_corners, 2)


def try_accumulate(camera_name, cor):
    """Keep this frame if the whole board is visible and its pose is novel.

    Mirrors the old should_save: full board only (no NaN), and mean per-corner
    displacement to every kept frame must be >= NOVELTY_PX.
    """
    if len(kept[camera_name]) >= MAX_FRAMES:
        return False
    if np.any(np.isnan(cor[:, 0])):  # require the full board (all n_corners detected)
        return False
    for prev in kept[camera_name]:
        if np.linalg.norm(cor - prev, axis=1).mean() < NOVELTY_PX:
            return False
    kept[camera_name].append(cor.copy())
    return True


while not exit_event.is_set():
    images_data = reader.get_images(copy=False)

    meta_data = []
    binary_data = []

    for camera_name, (image, frame_id) in images_data.items():
        if frame_id <= last_frame_ids[camera_name] or frame_id <= 0:
            continue

        cur_image = image.copy()
        detection = detect_charuco(cur_image)
        cor = fixed_corner_array(detection, BOARD, n_corners)

        if cor is not None:
            try_accumulate(camera_name, cor)

        # downscaled preview
        small = cv2.resize(cur_image, (cur_image.shape[1] // 8, cur_image.shape[0] // 8))
        success, encoded_image = cv2.imencode(
            '.jpg', small, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
        if not success:
            continue

        if cor is not None:
            cur_corners = (cor[~np.isnan(cor[:, 0])] / 8).astype(np.float32)
        else:
            cur_corners = np.zeros((0, 2), dtype=np.float32)

        meta_data.append({
            'type': 'image',
            'name': camera_name,
            'frame_id': int(frame_id),
            'kept': len(kept[camera_name]),
            'target': MAX_FRAMES,
            'data_index': len(binary_data),
        })
        binary_data.append(encoded_image)

        meta_data.append({
            'type': 'charuco_detection',
            'name': camera_name + "_corners",
            'frame_id': int(frame_id),
            'data_index': len(binary_data),
            'shape': cur_corners.shape,
        })
        binary_data.append(cur_corners.tobytes())

        last_frame_ids[camera_name] = frame_id

    if meta_data:
        dp.send_data(meta_data, binary_data)

    time.sleep(0.01)

# --- 'exit' received: persist one keypoint file per camera ------------------
timestamp = time.strftime("%Y%m%d_%H%M%S", time.localtime())
for camera_name, frames in kept.items():
    if not frames:
        print(f"[{camera_name}] no frames kept, skipping")
        continue
    arr = np.stack(frames, axis=0)[:, :, None, :]  # (N, num_corners, 1, 2)
    out_dir = os.path.join(intrinsic_dir, camera_name, "keypoint")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{timestamp}.npy")
    np.save(out_path, arr)
    print(f"[{camera_name}] saved {len(frames)} frames -> {out_path}  shape={arr.shape}")

dp.close()
cr.end()
print("intrinsic client done.")
