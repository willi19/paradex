import argparse
from threading import Event
import time
import cv2
import numpy as np

from paradex.io.camera_system.remote_camera_controller import remote_camera_controller
from paradex.io.capture_pc.ssh import run_script
from paradex.io.capture_pc.data_sender import DataCollector
from paradex.io.capture_pc.command_sender import CommandSender
from paradex.utils.keyboard_listener import listen_keyboard
from paradex.utils.system import get_camera_list, get_pc_list
from paradex.image.merge import merge_image

parser = argparse.ArgumentParser()
parser.add_argument(
    "--cameras",
    type=str,
    default=None,
    help="comma-separated camera serials to display, e.g. 26053248,26053260",
)
parser.add_argument(
    "--resize-divisor",
    type=int,
    default=8,
    help="downsample width/height before transfer; use 4 for 512x384",
)
args = parser.parse_args()
if args.resize_divisor < 1:
    raise ValueError("--resize-divisor must be >= 1")
selected_cameras = (
    {serial.strip() for serial in args.cameras.split(",") if serial.strip()}
    if args.cameras
    else None
)

stream_cmd = (
    "python src/capture/camera/stream_client.py "
    f"--resize-divisor {args.resize_divisor}"
)
stream_pcs = None
if selected_cameras is not None:
    stream_cmd += f" --cameras {','.join(sorted(selected_cameras))}"
    stream_pcs = [
        pc for pc in get_pc_list()
        if selected_cameras.intersection(get_camera_list(pc))
    ]
    if not stream_pcs:
        raise SystemExit(f"No capture PC owns requested cameras: {sorted(selected_cameras)}")

run_script(stream_cmd, pc_list=stream_pcs)

rcc = remote_camera_controller("stream_main.py")
dc = DataCollector()
dc.start()

cs = CommandSender()

start_event = Event()
exit_event = Event()
stop_event = Event()

listen_keyboard({"q":exit_event})

rcc.start("stream", False, fps=10)

img_dict = {}
img_text = {}
if selected_cameras is not None:
    print(f"Displaying cameras: {sorted(selected_cameras)}")

while not exit_event.is_set():        
    all_data = dc.get_data()
    print(all_data.keys())
    for item_name, item_data in all_data.items():
        if selected_cameras is not None and item_name not in selected_cameras:
            continue
        # Only process image type data
        if item_data.get('type') != 'image':
            continue
        
        image_bytes = item_data.get('data')
        frame_id = item_data.get('frame_id', 0)
        
        if image_bytes:
            # Decode JPEG
            nparr = np.frombuffer(image_bytes, np.uint8)
            image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            if image is not None:
                img_dict[item_name] = image
                img_text[item_name] = str(frame_id)

    if img_dict:
        merged_image = merge_image(img_dict, img_text)
        cv2.imshow("Merged Stream", merged_image)      
        cv2.waitKey(1)  
        time.sleep(0.01)
    
print("Stopping capture...")

# Cleanup
cv2.destroyAllWindows()
rcc.stop()
rcc.end()
dc.end()
cs.end()

print("Stream stopped.")
