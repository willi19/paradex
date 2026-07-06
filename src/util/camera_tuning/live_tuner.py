"""Live per-camera gain/exposure tuner.

Shows every camera's live feed in one window and lets you adjust the SELECTED
camera's exposure/gain in real time — FLIR ExposureTime/Gain are writable during
acquisition — so you can dial each camera in by eye. Press 's' to save the tuned
values back to system/current/camera.json.

Run on a CAPTURE PC (needs PySpin + cameras + a display):

    python src/util/camera_tuning/live_tuner.py
    python src/util/camera_tuning/live_tuner.py --serials 25305460 25305462 --fps 15

Controls (OpenCV window):
    trackbar 'camera'  : which camera the exp/gain sliders control
    trackbar 'exp_us'  : exposure time (microseconds) for the selected camera
    trackbar 'gain_db' : gain (dB) for the selected camera
    keys:  [ ]  prev/next camera    - =  exposure -/+    ; '  gain -/+
           s  save all to camera.json     q  quit

Note: this owns the cameras directly (single thread) and reaches into
PyspinCamera._configureExposure/_configureGain to push live changes — that is
intentional for a tuning tool. Run it standalone (not while a capture daemon
holds the cameras).
"""
import argparse
import json
import os

import cv2
import numpy as np

from paradex.io.camera_system.pyspin import get_serial_list, autoforce_ip, load_camera
from paradex.image.merge import merge_image
from paradex.utils.system import config_dir

EXP_MIN, EXP_MAX = 50, 30000      # microseconds
GAIN_MIN, GAIN_MAX = 0, 47        # dB
WIN = "live_tuner"


def _apply_exposure(cam, us):
    cam.set_exposure(us)         # public live setter (was _configureExposure)


def _apply_gain(cam, db):
    cam.set_gain(db)             # public live setter (was _configureGain)


def _save(serials, exp, gain):
    path = os.path.join(config_dir, "camera.json")
    try:
        with open(path) as f:
            cfg = json.load(f)
    except FileNotFoundError:
        cfg = {}
    for s, e, g in zip(serials, exp, gain):
        cfg.setdefault(s, {})
        cfg[s]["exposure"] = float(e)
        cfg[s]["gain"] = float(g)
    with open(path, "w") as f:
        json.dump(cfg, f, indent=4)
    print(f"[saved] {len(serials)} cameras -> {path}")


def main():
    parser = argparse.ArgumentParser(description="Live per-camera gain/exposure tuner.")
    parser.add_argument("--serials", type=str, nargs="+", default=None,
                        help="cameras to tune (default: all detected)")
    parser.add_argument("--fps", type=int, default=15)
    args = parser.parse_args()

    autoforce_ip()
    serials = args.serials or get_serial_list()
    if not serials:
        print("No cameras found.")
        return 1
    print(f"tuning {len(serials)} cameras: {serials}")

    cams = []
    for s in serials:
        cam = load_camera(s)                       # applies camera.json gain/exposure
        cam.start("continuous", False, frame_rate=args.fps)
        cams.append(cam)

    exp = [int(c.exposure_time) for c in cams]     # current values per camera
    gain = [int(round(c.gain)) for c in cams]
    sel = [0]
    syncing = {"v": False}                         # suppress slider callbacks during programmatic set

    cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)

    def on_cam(i):
        sel[0] = i
        syncing["v"] = True
        cv2.setTrackbarPos("exp_us", WIN, exp[i])
        cv2.setTrackbarPos("gain_db", WIN, gain[i])
        syncing["v"] = False

    def on_exp(v):
        if syncing["v"]:
            return
        i = sel[0]
        exp[i] = max(EXP_MIN, v)
        _apply_exposure(cams[i], exp[i])

    def on_gain(v):
        if syncing["v"]:
            return
        i = sel[0]
        gain[i] = v
        _apply_gain(cams[i], gain[i])

    cv2.createTrackbar("camera", WIN, 0, max(0, len(cams) - 1), on_cam)
    cv2.createTrackbar("exp_us", WIN, exp[0], EXP_MAX, on_exp)
    cv2.setTrackbarMin("exp_us", WIN, EXP_MIN)
    cv2.createTrackbar("gain_db", WIN, gain[0], GAIN_MAX, on_gain)

    try:
        while True:
            imgs = {}
            for idx, cam in enumerate(cams):
                frame, _ = cam.get_image()
                if frame is None:
                    frame = np.zeros((256, 341, 3), dtype=np.uint8)
                im = frame.copy()
                tag = f"[{idx}] {cam.serial_num}  exp={exp[idx]}us gain={gain[idx]}dB"
                sel_col = (0, 255, 255)
                cv2.putText(im, tag, (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.2,
                            sel_col if idx == sel[0] else (0, 255, 0), 3)
                if idx == sel[0]:
                    cv2.rectangle(im, (0, 0), (im.shape[1] - 1, im.shape[0] - 1), sel_col, 8)
                imgs[cam.serial_num] = im

            cv2.imshow(WIN, cv2.resize(merge_image(imgs), None, fx=0.5, fy=0.5))
            k = cv2.waitKey(1) & 0xFF
            i = sel[0]
            if k == ord('q'):
                break
            elif k == ord('['):
                cv2.setTrackbarPos("camera", WIN, (i - 1) % len(cams))
            elif k == ord(']'):
                cv2.setTrackbarPos("camera", WIN, (i + 1) % len(cams))
            elif k in (ord('-'), ord('_')):
                cv2.setTrackbarPos("exp_us", WIN, max(EXP_MIN, exp[i] - 100))
            elif k in (ord('='), ord('+')):
                cv2.setTrackbarPos("exp_us", WIN, min(EXP_MAX, exp[i] + 100))
            elif k == ord(';'):
                cv2.setTrackbarPos("gain_db", WIN, max(GAIN_MIN, gain[i] - 1))
            elif k == ord("'"):
                cv2.setTrackbarPos("gain_db", WIN, min(GAIN_MAX, gain[i] + 1))
            elif k == ord('s'):
                _save(serials, exp, gain)
    finally:
        for cam in cams:
            try:
                cam.stop()
                cam.release()
            except Exception:
                pass
        cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
