"""Verify hardware sync across cameras (frame_ids line up under the UTGE900 trigger).

With a hardware trigger every camera exposes on the same pulse, so at any instant
all cameras should report the SAME frame_id. This starts the trigger + all local
cameras in sync mode and measures the spread of frame_ids across cameras. A small,
transient spread (read-timing jitter, ≤1) is fine; a persistent or growing spread
means a camera is dropping frames or is not actually triggered.

Scope: checks the cameras on THIS capture PC. True cross-PC sync also relies on all
PCs sharing the one UTGE900 trigger — run this on each PC.

Run on a CAPTURE PC (needs PySpin + cameras + the UTGE900 signal generator):

    python src/validate/camera_system/sync_check.py
    python src/validate/camera_system/sync_check.py --seconds 10 --view
"""
import argparse
import time

from paradex.io.camera_system.camera_loader import CameraLoader
from paradex.io.camera_system.camera_reader import MultiCameraReader
from paradex.io.camera_system.signal_generator import UTGE900
from paradex.utils.system import network_info


def main():
    parser = argparse.ArgumentParser(description="Check multi-camera hardware sync (frame_id alignment).")
    parser.add_argument("--seconds", type=float, default=8.0)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--tolerance", type=int, default=1,
                        help="max acceptable frame_id spread across cameras")
    parser.add_argument("--view", action="store_true", help="show merged live view with frame_ids")
    args = parser.parse_args()

    gen = UTGE900(**network_info["signal_generator"]["param"])
    cl = CameraLoader()
    reader = None
    worst = 0
    bad = 0
    total = 0
    try:
        gen.start(fps=args.fps)
        cl.start("stream", True, fps=args.fps)      # syncMode=True
        reader = MultiCameraReader(cl.camera_names)
        time.sleep(1.0)                             # let frames start flowing

        t0 = time.time()
        while time.time() - t0 < args.seconds:
            data = reader.get_images(copy=True)
            fids = {name: fid for name, (img, fid) in data.items()}
            if len(fids) >= 2:
                hi = max(fids.values())
                spread = hi - min(fids.values())
                worst = max(worst, spread)
                total += 1
                if spread > args.tolerance:
                    bad += 1
                    lagging = [n for n, f in fids.items() if f != hi]
                    print(f"  spread={spread}  lagging={lagging}  fids={fids}")
            if args.view:
                import cv2
                from paradex.image.merge import merge_image
                imgs = {}
                for name, (img, fid) in data.items():
                    im = img.copy()
                    cv2.putText(im, f"{name} #{fid}", (30, 60),
                                cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)
                    imgs[name] = im
                cv2.imshow("sync_check", cv2.resize(merge_image(imgs), None, fx=0.5, fy=0.5))
                if (cv2.waitKey(1) & 0xFF) == ord('q'):
                    break
    finally:
        if reader is not None:
            reader.close()
        cl.end()
        try:
            gen.stop(); gen.end()
        except Exception:
            pass

    print("\n=== SYNC SUMMARY ===")
    print(f"  cameras: {len(cl.camera_names)}   samples: {total}")
    print(f"  worst frame_id spread: {worst}   (tolerance {args.tolerance})")
    print(f"  out-of-sync samples: {bad}/{total}")
    ok = total > 0 and worst <= args.tolerance
    print("SYNC OK — cameras aligned." if ok
          else "SYNC FAIL — cameras not aligned (frame drops / trigger not reaching a camera).")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
