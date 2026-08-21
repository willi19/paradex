"""Capture-PC side of extrinsic calibration.

Reads every local camera out of the daemon's shared-memory stream, detects ChArUco
corners, and publishes a downscaled JPEG preview + the corners to the main PC.

Live preview cost is what caps the displayed frame rate, so the detection the
preview needs is deliberately cheap:

* preview detection runs on a **grayscale, half-resolution** copy (~3.4x faster
  than full-res BGR, same corners on real board images),
* cameras are processed **in parallel** (OpenCV releases the GIL in ``detectBoard``),
* frames are processed at most ``PREVIEW_FPS`` times per second per camera.

The data that actually gets calibrated is untouched: on a ``save`` command the
camera re-detects on the **full-resolution** frame and writes that, plus the
full-res PNG. Preview quality never leaks into the calibration inputs.
"""

import argparse
import os
import time
from concurrent.futures import ThreadPoolExecutor
from threading import Event, Thread

import cv2
import numpy as np


# Set by _save_camera_data when a write fails, read by the main loop so the failure
# reaches the operator's screen instead of dying in a discarded stdout.
save_error = None


def _save_camera_data(save_path, camera_name, corners, ids, image, frame_id):
    """Write one camera's calibration input. Creates its own directories.

    The Main PC also mkdirs this path, but it is a *different machine*: when its
    shared_data mount is down its mkdir lands on its local disk while we write to the
    NAS, and every np.save here then dies with FileNotFoundError. That happened
    silently for a whole 25-pose session (run_script discards our stdout, and the
    save_id we publish had already been incremented), so own the mkdir here and make
    a failure loud rather than trusting the other machine's filesystem.
    """
    global save_error
    try:
        os.makedirs(os.path.join(save_path, "markers_2d"), exist_ok=True)
        os.makedirs(os.path.join(save_path, "images"), exist_ok=True)
        np.save(os.path.join(save_path, "markers_2d", f"{camera_name}_corner.npy"), corners)
        np.save(os.path.join(save_path, "markers_2d", f"{camera_name}_id.npy"), ids)
        if not cv2.imwrite(os.path.join(save_path, "images", f"{camera_name}.png"), image):
            raise IOError(f"cv2.imwrite returned False for {camera_name}.png")
        print(f"Saved data for camera {camera_name} at frame {frame_id} to {save_path}")
    except Exception as e:
        save_error = f"{camera_name}: {type(e).__name__}: {e}"
        print(f"[SAVE FAILED] {save_error} (path {save_path})", flush=True)

from paradex.io.camera_system.camera_reader import MultiCameraReader
from paradex.io.capture_pc.data_sender import DataPublisher
from paradex.io.capture_pc.command_sender import CommandReceiver
from paradex.image.aruco import detect_charuco, merge_charuco_detection
from paradex.calibration.utils import extrinsic_dir

parser = argparse.ArgumentParser()
parser.add_argument("--preview_fps", type=float, default=15.0,
                    help="Max frames/s processed per camera for the live preview (0 = every frame).")
parser.add_argument("--preview_scale", type=int, default=2,
                    help="Downscale factor for preview detection (1 = full res). 2 keeps all corners; 4 loses them.")
parser.add_argument("--boards", type=str, default=None,
                    help="Comma-separated board ids to detect (default: all in charuco_info.json). "
                         "Each board is a full detection pass — drop unused ones to go faster.")
parser.add_argument("--stream_scale", type=int, default=4,
                    help="Downscale factor of the preview sent to the main PC (bigger display = "
                         "smaller factor). 4 = 512x384/cam (~30 kB JPEG); 8 = 256x192.")
args = parser.parse_args()

PREVIEW_SCALE = max(1, args.preview_scale)
PREVIEW_PERIOD = 0.0 if args.preview_fps <= 0 else 1.0 / args.preview_fps
BOARD_IDS = [b.strip() for b in args.boards.split(",")] if args.boards else None
# Preview + corner coordinates are both sent at 1/STREAM_DOWNSCALE of full res; the
# factor rides along in the metadata so the main PC never has to assume it.
STREAM_DOWNSCALE = max(1, args.stream_scale)

dp = DataPublisher(port=1234, name="camera_stream")

exit_event = Event()
save_event = Event()
cr = CommandReceiver(event_dict={"exit": exit_event, "save": save_event}, port=6890)

reader = MultiCameraReader()
last_frame_ids = {name: 0 for name in reader.camera_names}
last_preview_ts = {name: 0.0 for name in reader.camera_names}

saved_this_round = set()
save_id = {name: 0 for name in reader.camera_names}

# Cameras are processed concurrently; keep OpenCV's own thread pool from
# oversubscribing the machine on top of that.
n_cams = max(1, len(reader.camera_names))
cv2.setNumThreads(max(1, (os.cpu_count() or 4) // n_cams))
pool = ThreadPoolExecutor(max_workers=n_cams)

print(f"[extrinsic client] {n_cams} cameras | preview scale 1/{PREVIEW_SCALE} "
      f"| preview_fps {args.preview_fps} | boards {BOARD_IDS or 'all'}")


def _process_camera(camera_name, image, frame_id, do_save, save_path):
    """Detect + encode one camera's frame. Runs on the pool; returns wire items."""
    cur_image = image.copy()  # shm double-buffer may flip under us
    gray = cv2.cvtColor(cur_image, cv2.COLOR_BGR2GRAY)

    if do_save and save_path is not None:
        # Calibration input: full-resolution detection, exactly as before.
        merged_full = merge_charuco_detection(detect_charuco(gray, board_ids=BOARD_IDS))
        corner_file = os.path.join(save_path, "markers_2d", f"{camera_name}_corner.npy")
        if os.path.exists(corner_file):
            print(f"Data for camera {camera_name} already saved, skipping.")
        else:
            Thread(target=_save_camera_data, args=(
                save_path, camera_name,
                merged_full["checkerCorner"].copy(),
                merged_full["checkerIDs"].copy(),
                cur_image,
                frame_id,
            ), daemon=True).start()
        preview_corners = merged_full["checkerCorner"] / STREAM_DOWNSCALE
    else:
        # Preview only: half-res grayscale detection is ~3.4x cheaper.
        if PREVIEW_SCALE > 1:
            det_img = cv2.resize(gray, (gray.shape[1] // PREVIEW_SCALE,
                                        gray.shape[0] // PREVIEW_SCALE),
                                 interpolation=cv2.INTER_AREA)
        else:
            det_img = gray
        merged = merge_charuco_detection(detect_charuco(det_img, board_ids=BOARD_IDS))
        preview_corners = merged["checkerCorner"] * (PREVIEW_SCALE / STREAM_DOWNSCALE)

    small = cv2.resize(cur_image, (cur_image.shape[1] // STREAM_DOWNSCALE,
                                   cur_image.shape[0] // STREAM_DOWNSCALE),
                       interpolation=cv2.INTER_AREA)
    ok, encoded_image = cv2.imencode('.jpg', small, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
    if not ok:
        return None

    return {
        "name": camera_name,
        "frame_id": int(frame_id),
        "shape": tuple(int(x) for x in image.shape),
        "jpeg": encoded_image,
        "corners": np.asarray(preview_corners, dtype=np.float32).tobytes(),
        "n_corners": int(len(preview_corners)),
    }


while not exit_event.is_set():
    images_data = reader.get_images(copy=False)
    now = time.time()

    saving = save_event.is_set()
    save_path = None
    if saving:
        info = cr.event_info.get("save", {})
        filename = info.get("filename")
        capture_idx = info.get("capture_idx")
        if filename and capture_idx:
            save_path = os.path.join(extrinsic_dir, filename, capture_idx)
        else:
            print("[WARN] save event set but no save_path in event_info, skipping.")

    jobs = []
    for camera_name, (image, frame_id) in images_data.items():
        if frame_id <= last_frame_ids[camera_name] or frame_id <= 0:
            continue

        do_save = saving and camera_name not in saved_this_round and save_path is not None
        # Throttle the preview, but never throttle a pending save.
        if not do_save and PREVIEW_PERIOD and now - last_preview_ts[camera_name] < PREVIEW_PERIOD:
            continue

        last_frame_ids[camera_name] = frame_id
        last_preview_ts[camera_name] = now
        if do_save:
            saved_this_round.add(camera_name)
            save_id[camera_name] += 1

        jobs.append(pool.submit(_process_camera, camera_name, image, frame_id,
                                do_save, save_path))

    meta_data = []
    binary_data = []
    for job in jobs:
        try:
            res = job.result()
        except Exception as e:
            print(f"[WARN] camera processing failed: {e}")
            continue
        if res is None:
            continue

        meta_data.append({
            'type': 'image',
            'name': res["name"],
            'frame_id': res["frame_id"],
            'save_id': save_id[res["name"]],
            # None unless a write failed; capture.py turns this into an on-screen
            # error so a silent save failure can't be mistaken for a good session.
            'save_error': save_error,
            'shape': res["shape"],
            'downscale': STREAM_DOWNSCALE,
            'data_index': len(binary_data),
        })
        binary_data.append(res["jpeg"])

        meta_data.append({
            'type': 'charuco_detection',
            'name': res["name"] + "_corners",
            'frame_id': res["frame_id"],
            'data_index': len(binary_data),
            'downscale': STREAM_DOWNSCALE,
            'shape': (res["n_corners"], 2),
        })
        binary_data.append(res["corners"])

    if save_event.is_set() and len(saved_this_round) >= len(reader.camera_names):
        save_event.clear()
        saved_this_round = set()
        print("Completed saving data for all cameras.")

    if meta_data:
        dp.send_data(meta_data, binary_data)

    time.sleep(0.002)  # yield; the preview throttle does the real rate limiting

pool.shutdown(wait=False)
