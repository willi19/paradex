import time
import cv2
from threading import Event
import os
import numpy as np

from paradex.io.camera_system.camera_reader import MultiCameraReader
from paradex.io.capture_pc.data_sender import DataPublisher
from paradex.io.capture_pc.command_sender import CommandReceiver
from paradex.image.aruco import detect_charuco, merge_charuco_detection
from paradex.calibration.utils import extrinsic_dir
from paradex.utils.file_io import find_latest_directory

dp = DataPublisher(port=1234, name="camera_stream")

exit_event = Event()
save_event = Event()
cr = CommandReceiver(event_dict={"exit": exit_event, "save": save_event}, port=6890)

reader = MultiCameraReader()
last_frame_ids = {name: 0 for name in reader.camera_names}
last_frame_dict = {name: None for name in reader.camera_names}

root_name = find_latest_directory(extrinsic_dir)
root_dir = os.path.join(extrinsic_dir, root_name)

save_remain = len(reader.camera_names)
save_id = {name:0 for name in reader.camera_names}

while not exit_event.is_set():
    images_data = reader.get_images(copy=False)
    
    meta_data = []
    binary_data = []

    for camera_name, (image, frame_id) in images_data.items():
        if frame_id > last_frame_ids[camera_name] and frame_id > 0:
            cur_image = image.copy()
            detect_result = detect_charuco(cur_image)
            merged_detect_result = merge_charuco_detection(detect_result)
            
            if save_event.is_set():
                save_name = find_latest_directory(root_dir)
                save_path = os.path.join(root_dir, save_name)
                
                if os.path.exists(os.path.join(save_path, "markers_2d", f"{camera_name}_corner.npy")):
                    print(f"Data for camera {camera_name} already saved, skipping.")
                    continue  # Already saved for this camera
                
                np.save(os.path.join(save_path, "markers_2d", f"{camera_name}_corner.npy"), merged_detect_result["checkerCorner"])
                np.save(os.path.join(save_path, "markers_2d", f"{camera_name}_id.npy"), merged_detect_result["checkerIDs"])

                cv2.imwrite(os.path.join(save_path, "images", f"{camera_name}.png"), cur_image)
                print(f"Saved data for camera {camera_name} at frame {frame_id} to {save_path}")
                save_remain -= 1
                save_id[camera_name] += 1

            cur_image = cv2.resize(cur_image, (cur_image.shape[1]//8, cur_image.shape[0]//8))
            encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 85]
            success, encoded_image = cv2.imencode('.jpg', cur_image, encode_param)
            merged_detect_result["checkerCorner"] = (merged_detect_result["checkerCorner"] / 8).astype(np.int16)
            
            if success:
                meta_data.append({
                    'type': 'image',  # 데이터 타입
                    'name': camera_name,
                    'frame_id': int(frame_id),
                    'save_id': save_id[camera_name],
                    'shape': tuple(int(x) for x in image.shape),
                    'data_index': len(binary_data)
                })
                # Add binary data
                binary_data.append(encoded_image)
                last_frame_ids[camera_name] = frame_id
                
                meta_data.append({
                    'type': 'charuco_detection',
                    'name': camera_name+"_corners",
                    'frame_id': int(frame_id),
                    'data_index': len(binary_data),
                    'shape': merged_detect_result["checkerCorner"].shape
                })
                
                binary_data.append(
                    np.array(merged_detect_result["checkerCorner"], dtype=np.float32).tobytes()
                )
        
    if save_event.is_set() and save_remain == 0:
        save_event.clear()
        save_remain = len(reader.camera_names)
        print("Completed saving data for all cameras.")
                
                
    if meta_data:
        dp.send_data(meta_data, binary_data)

    time.sleep(0.01)  # Small sleep to prevent busy-waiting