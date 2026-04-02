import time
import cv2
import numpy as np
import os

from paradex.io.camera_system.remote_camera_controller import remote_camera_controller
from paradex.io.capture_pc.ssh import run_script
from paradex.io.capture_pc.data_sender import DataCollector
from paradex.io.capture_pc.command_sender import CommandSender
from paradex.image.merge import merge_image
from paradex.calibration.utils import extrinsic_dir

TARGET_CAMERAS = [
    "25452066",  # capture18
    "23012641",  # capture14
    "23022639",  # capture12
]
TARGET_CAMERA_SET = set(TARGET_CAMERAS)
TARGET_PCS = ["capture18", "capture14", "capture12"]

filename = time.strftime("%Y%m%d_%H%M%S", time.localtime())
os.makedirs(os.path.join(extrinsic_dir, filename), exist_ok=True)

# Clean stale remote client processes from previous interrupted runs.
run_script("pkill -f 'src/calibration/extrinsic/client_3cam.py' || true", pc_list=TARGET_PCS)
time.sleep(0.5)
run_script("python src/calibration/extrinsic/client_3cam.py", pc_list=TARGET_PCS)

rcc = remote_camera_controller("extrinsic_calibration", pc_list=TARGET_PCS)
dc = DataCollector(pc_list=TARGET_PCS)
dc.start()

cs = CommandSender(pc_list=TARGET_PCS)
rcc.start("stream", False, fps=30)

img_dict = {}
img_text = {}

save_num = 0

try:
    while True:
        waiting_save = False
        all_data = dc.get_data()
        for item_name, item_data in all_data.items():
            # Only process image type data
            if item_data.get('type') == 'image':
                if item_name not in TARGET_CAMERA_SET:
                    continue
                image_bytes = item_data.get('data')
                frame_id = item_data.get('frame_id', 0)
                save_id = item_data.get('save_id', 0)

                if save_id < save_num:
                    waiting_save = True

                if image_bytes:
                    # Decode JPEG
                    nparr = np.frombuffer(image_bytes, np.uint8)
                    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                    
                    if image is not None:
                        img_dict[item_name] = image
                        img_text[item_name] = str(frame_id)
                    
        if img_dict:
            merged_image = merge_image(img_dict, img_text)
            if waiting_save:
                cv2.putText(merged_image, "Saving...", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                
            cv2.imshow("Merged Stream", merged_image)
            key = cv2.waitKey(1)

        else:
            blank_image = np.ones((600, 800, 3), dtype=np.uint8)*500
            cv2.putText(blank_image, "Waiting for stream...", (50, 300), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
            cv2.imshow("Merged Stream", blank_image)
            key = cv2.waitKey(1)
            
        if key == ord('q'):
            break
        
        elif key == ord('c'):
            capture_idx = time.strftime("%Y%m%d_%H%M%S", time.localtime())
            os.makedirs(os.path.join(extrinsic_dir, filename, str(capture_idx)), exist_ok=True)
            os.makedirs(os.path.join(extrinsic_dir, filename, str(capture_idx), "images"), exist_ok=True)
            
            cs.send_command("save", True)
            save_num += 1
            time.sleep(0.01)
except KeyboardInterrupt:
    pass
finally:
    print("Stopping capture...")

    # Cleanup
    rcc.stop()
    rcc.end()
    dc.end()
    cs.end()
    cv2.destroyAllWindows()

    print("Stream stopped.")
