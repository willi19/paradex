#!/usr/bin/env python3
"""
Simple live camera tuner for pyspin cameras.

Controls (per-window):
 - Keys:
   g/G : decrease/increase Gain by step
   h/H : decrease/increase Gamma by step
   a   : toggle BalanceWhiteAuto (if available)
   s   : save current frame to disk (in shared_data/capture/live_tuner/<serial>.png)
   q   : quit

Run on the capture PC where cameras are connected. Make sure no other process holds the cameras.
"""

import os
import time
import cv2
import argparse
import traceback

from paradex.utils.system import get_camera_list
from paradex.io.camera_system.pyspin import load_camera


def safe_get_node(cam, name, ntype):
    try:
        return cam._get_node(cam.nodeMap, name, ntype, True, True)
    except Exception:
        return None


def try_set_float(cam, name, value):
    node = safe_get_node(cam, name, 'float')
    if node is None:
        return False
    try:
        cam._set_node_value(node, 'float', float(value))
        return True
    except Exception:
        return False


def try_get_float(cam, name):
    node = safe_get_node(cam, name, 'float')
    if node is None:
        return None
    try:
        return node.GetValue()
    except Exception:
        return None


def try_toggle_enum(cam, name):
    node = safe_get_node(cam, name, 'enum')
    if node is None:
        return None
    try:
        cur = node.GetCurrentEntry().GetSymbolic()
        entries = [e.GetSymbolic() for e in node.GetEntries() if e.GetSymbolic() is not None]
        # toggle between first two entries (if available)
        if len(entries) >= 2:
            nxt = entries[1] if cur == entries[0] else entries[0]
            cam._set_node_value(node, 'enum', nxt)
            return nxt
        return cur
    except Exception:
        return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gain_step', type=float, default=10.0)
    parser.add_argument('--gamma_step', type=float, default=1.0)
    args = parser.parse_args()

    serials = get_camera_list()
    if len(serials) == 0:
        print('No cameras configured in system/current/pc.json')
        return

    cams = []
    for s in serials:
        try:
            cam = load_camera(s)
            # start continuous acquisition so GetNextImage works
            try:
                cam.start("continuous", False, frame_rate=30)
            except Exception as e:
                print(f"Warning: failed to start continuous on {s}: {e}")
                print("If another process (server_daemon.py) is holding the camera, stop it first.")
            cams.append((s, cam))
        except Exception as e:
            print(f'Failed to open camera {s}: {e}')

    if not cams:
        print('No cameras available')
        return

    windows = {}
    for serial, cam in cams:
        win = f'cam_{serial}'
        cv2.namedWindow(win, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(win, 800, 600)
        windows[serial] = win

    save_dir = os.path.join('shared_data', 'capture', 'live_tuner')
    os.makedirs(save_dir, exist_ok=True)

    # color_mode: 'rgb' means frames coming from cam are RGB and need conversion to BGR
    # 'bgr' means frames are already BGR and can be shown directly.
    color_mode = 'rgb'

    try:
        while True:
            for serial, cam in cams:
                try:
                    frame, meta = cam.get_image()
                except Exception as e:
                    print(f'Error getting frame from {serial}: {e}')
                    frame = None

                if frame is None:
                    continue

                # show frame
                if color_mode == 'rgb':
                    # convert from RGB (camera wrapper) to BGR for OpenCV display
                    try:
                        img_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                    except Exception:
                        # if conversion fails for some reason, fall back to raw frame
                        img_bgr = frame
                else:
                    # assume frame already BGR
                    img_bgr = frame

                # overlay color mode on the frame
                try:
                    disp = img_bgr.copy()
                    cv2.putText(disp, f'MODE: {color_mode.upper()}', (10, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
                    cv2.imshow(windows[serial], disp)
                except Exception:
                    cv2.imshow(windows[serial], img_bgr)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key in (ord('g'), ord('G')):
                # increase/decrease gain for all cameras
                for serial, cam in cams:
                    cur = try_get_float(cam, 'Gain')
                    if cur is None:
                        continue
                    new = cur + (args.gain_step if key == ord('G') else -args.gain_step)
                    if try_set_float(cam, 'Gain', new):
                        print(f"{serial} Gain set to {new}")
            elif key in (ord('h'), ord('H')):
                for serial, cam in cams:
                    cur = try_get_float(cam, 'Gamma')
                    if cur is None:
                        continue
                    new = cur + (args.gamma_step if key == ord('H') else -args.gamma_step)
                    if try_set_float(cam, 'Gamma', new):
                        print(f"{serial} Gamma set to {new}")
            elif key == ord('a'):
                for serial, cam in cams:
                    nxt = try_toggle_enum(cam, 'BalanceWhiteAuto')
                    if nxt is not None:
                        print(f"{serial} BalanceWhiteAuto -> {nxt}")
            elif key == ord('c'):
                # toggle color mode
                color_mode = 'bgr' if color_mode == 'rgb' else 'rgb'
                print(f'Color mode -> {color_mode}')
            elif key == ord('s'):
                # save current frames
                t = int(time.time())
                for serial, cam in cams:
                    frame, _ = cam.get_image()
                    if frame is None:
                        continue
                    if color_mode == 'rgb':
                        try:
                            img_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                        except Exception:
                            img_bgr = frame
                    else:
                        img_bgr = frame
                    outp = os.path.join(save_dir, f"{serial}_{t}.png")
                    cv2.imwrite(outp, img_bgr)
                    print('Saved', outp)

    finally:
        cv2.destroyAllWindows()
        for serial, cam in cams:
            try:
                cam.stop()
                cam.release()
            except Exception:
                pass


if __name__ == '__main__':
    main()
