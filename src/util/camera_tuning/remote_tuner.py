"""Main-PC live per-camera gain/exposure tuner for the WHOLE rig.

Unlike ``live_tuner.py`` (which owns cameras on one capture PC and needs the daemon
off), this tunes every camera across all capture PCs in real time **through the
running daemon** — it drives ``rcc.set_param`` and shows a live merged preview, so
you dial each camera in by eye without stopping anything. Save writes the tuned
values to the main-PC ``system/current/camera.json``.

Run on the MAIN PC (camera daemons must be up, like any ``rcc`` app; needs a display):

    python src/util/camera_tuning/remote_tuner.py
    python src/util/camera_tuning/remote_tuner.py --fps 10

Controls (OpenCV window):
    [ ]   select prev / next camera
    - =   exposure -/+  (selected camera, microseconds)
    ; '   gain -/+      (selected camera, dB)
    s     save all tuned values -> system/current/camera.json (main PC)
    q     quit

Preview reuses the distributed stream pipeline: it SSH-launches
``src/capture/camera/stream_client.py`` on the capture PCs (they read the daemon's
shared memory and publish JPEGs), and collects them here via ``DataCollector``.
"""
import argparse
import json
import os

import cv2
import numpy as np

from paradex.io.camera_system.remote_camera_controller import remote_camera_controller
from paradex.io.capture_pc.ssh import run_script
from paradex.io.capture_pc.data_sender import DataCollector
from paradex.image.merge import merge_image
from paradex.utils.system import config_dir, get_camera_config

EXP_MIN, EXP_MAX = 50, 30000       # microseconds
GAIN_MIN, GAIN_MAX = 0, 47         # dB
EXP_STEP, GAIN_STEP = 100, 1
DEFAULT_EXP, DEFAULT_GAIN = 2500, 3
WIN = "remote_tuner"


def _save(params):
    """Merge {serial: {exposure, gain}} into the main-PC camera.json."""
    path = os.path.join(config_dir, "camera.json")
    try:
        with open(path) as f:
            cfg = json.load(f)
    except FileNotFoundError:
        cfg = {}
    for s, (exp, gain) in params.items():
        cfg.setdefault(s, {})
        cfg[s]["exposure"] = float(exp)
        cfg[s]["gain"] = float(gain)
    with open(path, "w") as f:
        json.dump(cfg, f, indent=4)
    print(f"[saved] {len(params)} cameras -> {path}")
    print("  (sync to capture PCs: git commit + git_pull, or copy system/current/camera.json)")


def main():
    parser = argparse.ArgumentParser(description="Main-PC live per-camera gain/exposure tuner.")
    parser.add_argument("--fps", type=int, default=10)
    args = parser.parse_args()

    baseline = get_camera_config()     # per-serial camera.json baseline (main PC)

    run_script("python src/capture/camera/stream_client.py")
    rcc = remote_camera_controller("remote_tuner")
    dc = DataCollector()
    dc.start()

    rcc.arm(syncMode=False, fps=args.fps)
    rcc.set_stream(True)

    serials = []                       # discovered from the preview stream (ordered)
    params = {}                        # serial -> [exp_us, gain_db]
    sel = [0]

    def _seed(s):
        if s not in params:
            b = baseline.get(s, {})
            params[s] = [int(b.get("exposure", DEFAULT_EXP)), int(round(b.get("gain", DEFAULT_GAIN)))]
            serials.append(s)

    cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)
    print("controls: [ ] select | - = exposure | ; ' gain | s save | q quit")

    try:
        while True:
            img_dict, img_text = {}, {}
            for name, data in dc.get_data().items():
                if data.get('type') != 'image':
                    continue
                _seed(name)
                buf = data.get('data')
                if not buf:
                    continue
                img = cv2.imdecode(np.frombuffer(buf, np.uint8), cv2.IMREAD_COLOR)
                if img is None:
                    continue
                exp, gain = params[name]
                is_sel = serials and serials[sel[0] % len(serials)] == name
                if is_sel:
                    cv2.rectangle(img, (0, 0), (img.shape[1] - 1, img.shape[0] - 1), (0, 255, 255), 6)
                img_dict[name] = img
                img_text[name] = f"{name} exp={exp} gain={gain}" + (" *" if is_sel else "")

            if img_dict:
                cv2.imshow(WIN, merge_image(img_dict, img_text))

            k = cv2.waitKey(20) & 0xFF
            if not serials:
                if k == ord('q'):
                    break
                continue
            i = sel[0] % len(serials)
            s = serials[i]

            changed = None
            if k == ord('q'):
                break
            elif k == ord('['):
                sel[0] = (i - 1) % len(serials)
            elif k == ord(']'):
                sel[0] = (i + 1) % len(serials)
            elif k in (ord('-'), ord('_')):
                params[s][0] = max(EXP_MIN, params[s][0] - EXP_STEP); changed = 'exp'
            elif k in (ord('='), ord('+')):
                params[s][0] = min(EXP_MAX, params[s][0] + EXP_STEP); changed = 'exp'
            elif k == ord(';'):
                params[s][1] = max(GAIN_MIN, params[s][1] - GAIN_STEP); changed = 'gain'
            elif k == ord("'"):
                params[s][1] = min(GAIN_MAX, params[s][1] + GAIN_STEP); changed = 'gain'
            elif k == ord('s'):
                _save(params)

            # Push just the changed value for the selected camera, live.
            if changed == 'exp':
                rcc.set_param(exposure={s: float(params[s][0])})
            elif changed == 'gain':
                rcc.set_param(gain={s: float(params[s][1])})
    finally:
        cv2.destroyAllWindows()
        rcc.stop()
        rcc.end()
        dc.end()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
