import argparse
from collections import deque
from threading import Event
import time
import cv2
import numpy as np
import os

from paradex.io.camera_system.remote_camera_controller import remote_camera_controller
from paradex.io.capture_pc.ssh import run_script
from paradex.io.capture_pc.data_sender import DataCollector
from paradex.io.capture_pc.command_sender import CommandSender
from paradex.image.merge import merge_image, fit_to_screen, get_screen_size
from paradex.image.overlay import overlay_mask
from paradex.calibration.utils import extrinsic_dir
from paradex.image.aruco import draw_charuco

BOARD_COLORS = [
    (0, 0, 255), 
    (0, 255, 0)
]

parser = argparse.ArgumentParser()
parser.add_argument("--stream_scale", type=int, default=4,
                    help="Preview downscale on the capture PCs: 4 = 512x384/cam (default), "
                         "8 = the old 256x192. Lower = bigger picture, more bandwidth.")
parser.add_argument("--width", type=int, default=None,
                    help="Total width of the merged preview in pixels (default: fit the screen).")
parser.add_argument("--margin", type=float, default=0.95,
                    help="Fraction of the screen the preview may occupy when auto-fitting.")
args = parser.parse_args()

SCREEN_W, SCREEN_H = get_screen_size()
# Compose at screen width, then letterbox the finished grid to the screen: the
# limiting dimension decides, so the aspect ratio is never stretched.
CANVAS_W = args.width or SCREEN_W
print(f"[extrinsic] screen {SCREEN_W}x{SCREEN_H} -> canvas width {CANVAS_W}")

# Resizable window so it can still be dragged smaller.
cv2.namedWindow("Merged Stream", cv2.WINDOW_NORMAL)

filename = time.strftime("%Y%m%d_%H%M%S", time.localtime())
os.makedirs(os.path.join(extrinsic_dir, filename), exist_ok=True)

run_script(f"python src/calibration/extrinsic/client.py --stream_scale {args.stream_scale}")

rcc = remote_camera_controller("extrinsic_calibration")
dc = DataCollector()
dc.start()

cs = CommandSender()
rcc.arm(syncMode=False, fps=30)
rcc.set_stream(True)

saved_corner_img = {}# serial_num:np.ones((1536, 2048, 3), dtype=np.uint8)*255 for serial_num in serial_list}
saved_corner_mask = {}
cur_state = {}#serial_num:(np.array([]), np.array([]), 0) for serial_num in serial_list}

img_dict = {}
img_text = {}

save_num = 0
saved_count = {}  # serial_num -> number of captures this camera actually saved corners for

last_seen = {}      # item_name -> frame_id already decoded/drawn
dirty = True        # only re-decode + re-merge when something actually changed
arrivals = deque()  # timestamps of newly-arrived preview frames (for the fps HUD)
merged_image = None
start_ts = time.time()
printed_hint = False

while True:
    waiting_save = False
    all_data = dc.get_data()
    for item_name, item_data in all_data.items():
        # Only process image type data
        if item_data.get('type') == 'image':
            image_bytes = item_data.get('data')
            frame_id = item_data.get('frame_id', 0)
            save_id = item_data.get('save_id', 0)

            if save_id < save_num:
                waiting_save = True

            # The collector conflates to the newest item per name, so the same frame
            # is handed back on every iteration until a new one lands. Decoding it
            # again would just burn a core.
            if last_seen.get(item_name) == frame_id:
                continue
            last_seen[item_name] = frame_id

            if image_bytes:
                # Decode JPEG
                nparr = np.frombuffer(image_bytes, np.uint8)
                image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

                if image is not None:
                    img_dict[item_name] = image
                    img_text[item_name] = f"{frame_id} | n={saved_count.get(item_name, 0)}"
                    arrivals.append(time.time())
                    dirty = True


        elif item_data.get('type') == 'charuco_detection':
            data = item_data.get('data')
            serial_num = item_name.split("_")[0]
            frame_id = item_data.get('frame_id', 0)
            if last_seen.get(item_name) == frame_id:
                continue
            last_seen[item_name] = frame_id
            corners = np.frombuffer(data, dtype=np.float32).reshape(-1, 2)
            if serial_num not in saved_corner_img:
                # Accumulator lives in preview pixels, so it must match whatever
                # downscale the capture PC is streaming at (it rides in the meta).
                ds = item_data.get('downscale', 8)
                saved_corner_img[serial_num] = np.zeros((1536 // ds, 2048 // ds, 3), dtype=np.uint8)
                saved_corner_mask[serial_num] = np.zeros((0, 2), dtype=np.int32)

            cur_state[serial_num] = (corners, frame_id)
            dirty = True

    if not img_dict:
        waited = time.time() - start_ts
        blank_image = np.ones((640, 1100, 3), dtype=np.uint8)*255
        cv2.putText(blank_image, f"Waiting for stream... {waited:.0f}s", (40, 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
        st = dc.get_stats()
        live = [pc for pc, s in st.items() if s['recv'] > 0]
        cv2.putText(blank_image, f"capture PCs sending: {len(live)}/{len(st)}  {live}", (40, 140),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)
        if waited > 5:
            # Silence here is almost always the capture-PC client dying at launch;
            # run_script sends its stdout to /dev/null, so it fails invisibly.
            for i, line in enumerate([
                "Nothing arriving. Most likely client.py failed to start on the capture PCs:",
                "  1. capture PCs run their own checkout - did they pull this version?",
                "     git commit + push, then: python src/util/git_pull.py",
                "  2. see the real error:",
                "     run_script('python src/calibration/extrinsic/client.py', ['capture1'], log=True)",
                "     then read ~/test.log on that PC",
                "  3. camera daemon down -> python src/camera/reset_cameras.py",
            ]):
                cv2.putText(blank_image, line, (40, 210 + i * 34),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 180), 1, cv2.LINE_AA)
            if not printed_hint:
                printed_hint = True
                print("[extrinsic] no stream after 5s — see the hints on the window; "
                      "check that capture PCs pulled this version of client.py")
        cv2.imshow("Merged Stream", blank_image)
        key = cv2.waitKey(1)

    elif dirty:
        dirty = False
        display_dict = {}

        for serial_num in cur_state.keys():
            if serial_num not in img_dict:
                continue
            display_dict[serial_num] = img_dict[serial_num].copy()
            display_dict[serial_num][saved_corner_mask[serial_num][:, 1], saved_corner_mask[serial_num][:, 0]] = BOARD_COLORS[0]
            corners, frame = cur_state[serial_num]

            if corners.shape[0] > 0:
                draw_charuco(display_dict[serial_num], corners, BOARD_COLORS[1], 1, -1)

        merged_image = merge_image(display_dict, img_text, canvas_width=CANVAS_W)
        if waiting_save:
            cv2.putText(merged_image, "Saving...", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        now = time.time()
        while arrivals and now - arrivals[0] > 2.0:
            arrivals.popleft()
        stats = dc.get_stats()
        lat = max([s['latency_ms'] for s in stats.values()], default=0.0)
        drops = sum(s['drops'] for s in stats.values())
        # Slow because frames aren't produced (capture-PC detect/camera fps) or
        # because they arrive late (transport)? This line separates the two.
        cv2.putText(merged_image,
                    f"{len(arrivals) / 2.0 / max(1, len(display_dict)):.1f} fps/cam | "
                    f"tx {lat:.0f} ms | drops {drops}",
                    (10, merged_image.shape[0] - 12), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (0, 0, 255), 2, cv2.LINE_AA)

        # Letterbox to the screen: whichever dimension runs out first sets the
        # scale, so nothing is stretched and nothing spills off-screen.
        cv2.imshow("Merged Stream", fit_to_screen(merged_image, margin=args.margin,
                                                  screen=(SCREEN_W, SCREEN_H)))
        key = cv2.waitKey(1)

    else:
        # Nothing new arrived — stay responsive without re-rendering the same frame.
        key = cv2.waitKey(5)

    if key == ord('q'):
        break
    
    elif key == ord('c'):
        capture_idx = time.strftime("%Y%m%d_%H%M%S", time.localtime())
        save_path = os.path.join(extrinsic_dir, filename, str(capture_idx))
        os.makedirs(os.path.join(save_path, "markers_2d"), exist_ok=True)
        os.makedirs(os.path.join(save_path, "images"), exist_ok=True)

        cs.send_command("save", True, cmd_info={"filename": filename, "capture_idx": str(capture_idx)})
        save_num += 1
        for serial_num in cur_state.keys():
            corners, frame = cur_state[serial_num]
            if corners.shape[0] > 0:
                draw_charuco(saved_corner_img[serial_num], corners, BOARD_COLORS[1], 1, -1)
                ys, xs, _ = np.where(saved_corner_img[serial_num] != 0)

                saved_corner_mask[serial_num] = np.stack([xs, ys], axis=1)
                saved_count[serial_num] = saved_count.get(serial_num, 0) + 1
        time.sleep(0.01)
    
print("Stopping capture...")

# Cleanup
rcc.stop()
rcc.end()
dc.end()
cs.end()

print("Stream stopped.")