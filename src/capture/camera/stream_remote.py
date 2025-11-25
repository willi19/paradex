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

run_script("python src/capture/stream_client.py")

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

while not exit_event.is_set():        
    all_data = dc.get_data()
    for item_name, item_data in all_data.items():
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