from threading import Event
import time
import cv2
import numpy as np

from paradex.io.camera_system.remote_camera_controller import remote_camera_controller
from paradex.io.capture_pc.ssh import run_script
from paradex.io.capture_pc.data_sender import DataCollector
from paradex.io.capture_pc.command_sender import CommandSender
from paradex.utils.keyboard_listener import listen_keyboard


# run_script("python src/capture/stream_client.py")

rcc = remote_camera_controller("stream_main.py")
dc = DataCollector()
dc.start()

cs = CommandSender()

start_event = Event()
exit_event = Event()
stop_event = Event()

listen_keyboard({"q":exit_event})

rcc.start("stream", False, fps=10)

while not exit_event.is_set():        
    all_data = dc.get_data()
    display_images = []
    # print(all_data)
    for pc_name, pc_data in all_data.items():
        if pc_data is None:
            continue
        
        camera_data = pc_data.get('data', {})
        for camera_name, cam_info in camera_data.items():
            # Decompress image
            try:
                image_bytes = cam_info.get('image')
                frame_id = cam_info.get('frame_id', 0)
                drop_count = cam_info.get('drop_count', 0)

                # if image_bytes:
                #     # Decode JPEG
                #     nparr = np.frombuffer(image_bytes, np.uint8)
                #     image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                    
                #     if image is not None:
                #         # Add text overlay with PC name, frame ID, and drop count
                #         text1 = f"{pc_name}:{camera_name}"
                #         text2 = f"Frame {frame_id} | Drops: {drop_count}"
                        
                #         cv2.putText(image, text1, (10, 30), 
                #                     cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                #         cv2.putText(image, text2, (10, 60), 
                #                     cv2.FONT_HERSHEY_SIMPLEX, 0.6, 
                #                     (0, 0, 255) if drop_count > 0 else (0, 255, 0), 2)
                        
                #         # Resize for display if needed
                #         display_h, display_w = 480, 640
                #         image_resized = cv2.resize(image, (display_w, display_h))
                #         display_images.append(image_resized)
            except:
                continue
            
    time.sleep(0.01)
    
print("Stopping capture...")

# Cleanup
cv2.destroyAllWindows()
rcc.stop()
rcc.end()
dc.end()
cs.end()

print("Stream stopped.")