from threading import Event
import time
import cv2
import numpy as np
import os

from paradex.io.camera_system.remote_camera_controller import remote_camera_controller
from paradex.io.capture_pc.ssh import run_script
from paradex.io.capture_pc.data_sender import DataCollector
from paradex.io.capture_pc.command_sender import CommandSender
from paradex.image.merge import merge_image
from paradex.image.overlay import overlay_mask
from paradex.calibration.utils import extrinsic_dir
from paradex.image.aruco import draw_charuco

BOARD_COLORS = [
    (0, 0, 255), 
    (0, 255, 0)
]

filename = time.strftime("%Y%m%d_%H%M%S", time.localtime())
os.makedirs(os.path.join(extrinsic_dir, filename), exist_ok=True)

# run_script("python src/calibration/extrinsic/client.py")

rcc = remote_camera_controller("extrinsic_calibration")
dc = DataCollector()
dc.start()

cs = CommandSender()
rcc.start("stream", False, fps=30)

saved_corner_img = {}# serial_num:np.ones((1536, 2048, 3), dtype=np.uint8)*255 for serial_num in serial_list}
saved_corner_mask = {}
cur_state = {}#serial_num:(np.array([]), np.array([]), 0) for serial_num in serial_list}

img_dict = {}
img_text = {}

save_num = 0

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

            if image_bytes:
                # Decode JPEG
                nparr = np.frombuffer(image_bytes, np.uint8)
                image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                
                if image is not None:
                    img_dict[item_name] = image
                    img_text[item_name] = str(frame_id)
                
        
        elif item_data.get('type') == 'charuco_detection':
            data = item_data.get('data')
            serial_num = item_name.split("_")[0]
            corners = np.frombuffer(data, dtype=np.float32).reshape(-1, 2)
            if serial_num not in saved_corner_img:
                saved_corner_img[serial_num] = np.zeros((1536 // 8, 2048 // 8, 3), dtype=np.uint8)
                saved_corner_mask[serial_num] = np.zeros((0, 2), dtype=np.int32)
                
            cur_state[serial_num] = (corners, frame_id)

    if img_dict:
        display_dict = {}
        
        for serial_num in cur_state.keys():
            display_dict[serial_num] = img_dict[serial_num].copy()
            display_dict[serial_num][saved_corner_mask[serial_num][:, 1], saved_corner_mask[serial_num][:, 0]] = BOARD_COLORS[0]
            corners, frame = cur_state[serial_num]
            
            if corners.shape[0] > 0:
                draw_charuco(display_dict[serial_num], corners, BOARD_COLORS[1], 1, -1)

        merged_image = merge_image(display_dict, img_text)
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
        os.makedirs(os.path.join(extrinsic_dir, filename, str(capture_idx), "markers_2d"), exist_ok=True)
        os.makedirs(os.path.join(extrinsic_dir, filename, str(capture_idx), "images"), exist_ok=True)
        
        cs.send_command("save", True)
        save_num += 1
        for serial_num in cur_state.keys():
            corners, frame = cur_state[serial_num]
            if corners.shape[0] > 0:
                draw_charuco(saved_corner_img[serial_num], corners, BOARD_COLORS[1], 1, -1)
                ys, xs, _ = np.where(saved_corner_img[serial_num] != 0)
                
                saved_corner_mask[serial_num] = np.stack([xs, ys], axis=1)
        time.sleep(0.01)
    
print("Stopping capture...")

# Cleanup
rcc.stop()
rcc.end()
dc.end()
cs.end()

print("Stream stopped.")