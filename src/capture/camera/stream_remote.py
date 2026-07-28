from collections import deque
from threading import Event
import time
import cv2
import numpy as np

from paradex.io.camera_system.remote_camera_controller import remote_camera_controller
from paradex.io.capture_pc.ssh import run_script
from paradex.io.capture_pc.data_sender import DataCollector
from paradex.io.capture_pc.command_sender import CommandSender
from paradex.utils.keyboard_listener import listen_keyboard
from paradex.image.merge import merge_image

run_script("python src/capture/camera/stream_client.py")

rcc = remote_camera_controller("stream_main.py")
dc = DataCollector()
dc.start()

cs = CommandSender()

start_event = Event()
exit_event = Event()
stop_event = Event()

listen_keyboard({"q":exit_event})

rcc.arm(syncMode=False, fps=10)
# A PC whose sink command failed streams nothing; surface it here instead of
# leaving a silently missing quarter of the rig on screen.
sink_reply = rcc.set_stream(True)
print(f"[stream] set_stream replies: {sink_reply}")
print(f"[stream] rcc status: {rcc.get_status()}")

img_dict = {}
img_text = {}

last_seen = {}      # serial -> frame_id already decoded
arrivals = deque()  # arrival timestamps of new frames, for the fps readout
cam_frames = deque()  # (ts, frame_id delta) — how many frames the CAMERA produced
dirty = False
last_report = 0.0     # per-PC arrival report cadence

while not exit_event.is_set():
    all_data = dc.get_data()

    # Which capture PCs are actually feeding us? A silent PC means its daemon or
    # stream_client isn't running / isn't armed — not a display problem.
    now_report = time.time()
    if now_report - last_report > 2.0:
        last_report = now_report
        st = dc.get_stats()
        live = [pc for pc, s in st.items() if s['recv'] > 0]
        silent = [pc for pc, s in st.items() if s['recv'] == 0]
        print(f"[pc] {len(live)}/{len(st)} sending | " +
              " ".join(f"{pc}:{s['recv']}" for pc, s in st.items()) +
              (f" | SILENT: {silent}" if silent else ""))

    for item_name, item_data in all_data.items():
        # Only process image type data
        if item_data.get('type') != 'image':
            continue

        image_bytes = item_data.get('data')
        frame_id = item_data.get('frame_id', 0)

        # The collector conflates to the newest item per name, so the same frame
        # comes back every iteration until a new one lands. Decoding + re-merging
        # it again is pure waste and it starves the display loop.
        prev_id = last_seen.get(item_name)
        if prev_id == frame_id:
            continue
        if prev_id is not None and frame_id > prev_id:
            # frame_id is the camera's own counter: its rate is what the camera
            # actually produced, independent of how many previews reached us.
            cam_frames.append((time.time(), frame_id - prev_id))
        last_seen[item_name] = frame_id

        if image_bytes:
            # Decode JPEG
            nparr = np.frombuffer(image_bytes, np.uint8)
            image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

            if image is not None:
                img_dict[item_name] = image
                img_text[item_name] = str(frame_id)
                arrivals.append(time.time())
                dirty = True

    if dirty and img_dict:
        now = time.time()
        while arrivals and now - arrivals[0] > 2.0:
            arrivals.popleft()
        while cam_frames and now - cam_frames[0][0] > 2.0:
            cam_frames.popleft()
        n_cam = max(1, len(img_dict))
        fps_per_cam = len(arrivals) / 2.0 / n_cam
        cam_fps = sum(d for _, d in cam_frames) / 2.0 / n_cam
        stats = dc.get_stats()
        lat = max([s['latency_ms'] for s in stats.values()], default=0.0)
        drops = sum(s['drops'] for s in stats.values())

        merged_image = merge_image(img_dict, img_text)
        # cam = frames the camera actually produced (frame_id rate); shown = frames
        # that reached this display. cam low  -> camera/GigE link is the limit.
        # cam high, shown low -> capture-PC client or transport is the limit.
        n_pc_live = sum(1 for s in stats.values() if s['recv'] > 0)
        hud = (f"pc {n_pc_live}/{len(stats)} | cam {cam_fps:.1f} fps | "
               f"shown {fps_per_cam:.1f} fps | tx {lat:.0f} ms | drops {drops}")
        # Size and place from the measured text so it can't run off the canvas.
        hud_scale = max(0.5, min(1.1, merged_image.shape[1] / 1800))
        hud_th = max(1, int(round(hud_scale * 2)))
        (hud_w, hud_h), _ = cv2.getTextSize(hud, cv2.FONT_HERSHEY_SIMPLEX, hud_scale, hud_th)
        hud_x = max(10, merged_image.shape[1] - hud_w - 14)
        hud_y = merged_image.shape[0] - max(10, hud_h // 2)
        cv2.putText(merged_image, hud, (hud_x, hud_y), cv2.FONT_HERSHEY_SIMPLEX,
                    hud_scale, (0, 0, 0), hud_th + 3, cv2.LINE_AA)
        cv2.putText(merged_image, hud, (hud_x, hud_y), cv2.FONT_HERSHEY_SIMPLEX,
                    hud_scale, (0, 0, 255), hud_th, cv2.LINE_AA)
        cv2.imshow("Merged Stream", merged_image)
        dirty = False
        cv2.waitKey(1)
    else:
        cv2.waitKey(5)

print("Stopping capture...")

# Cleanup
cv2.destroyAllWindows()
rcc.stop()
rcc.end()
dc.end()
cs.end()

print("Stream stopped.")