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
                # Prefer capture_idx delivered in the save command itself
                # (eliminates the NFS-listdir race on slow networks). Fall back
                # to filesystem discovery if main didn't pass it.
                save_name = cr.event_info.get("save", {}).get("capture_idx")
                if not save_name:
                    for _ in range(20):  # retry up to ~2s for NFS to catch up
                        save_name = find_latest_directory(root_dir)
                        if save_name:
                            break
                        time.sleep(0.1)
                if not save_name:
                    print(f"[client] could not resolve save dir under {root_dir}; skipping")
                    save_event.clear()
                    save_remain = len(reader.camera_names)
                    continue
                save_path = os.path.join(root_dir, save_name)
                os.makedirs(os.path.join(save_path, "markers_2d"), exist_ok=True)
                os.makedirs(os.path.join(save_path, "images"), exist_ok=True)

                if os.path.exists(os.path.join(save_path, "markers_2d", f"{camera_name}_corner.npy")):
                    print(f"Data for camera {camera_name} already saved, skipping.", save_remain)
                    continue  # Already saved for this camera

                np.save(os.path.join(save_path, "markers_2d", f"{camera_name}_corner.npy"), merged_detect_result["checkerCorner"])
                np.save(os.path.join(save_path, "markers_2d", f"{camera_name}_id.npy"), merged_detect_result["checkerIDs"])

                # JPEG Q95 ~ 5-10MB PNG -> ~500KB. extrinsic uses corner.npy
                # for actual calibration; the image file is visual-only.
                cv2.imwrite(os.path.join(save_path, "images", f"{camera_name}.jpg"),
                            cur_image, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
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
                # cv2.imencode returns np.ndarray; convert to bytes explicitly
                # for ZMQ multipart safety (was added on vlm_dex_camera branch).
                try:
                    img_bytes = encoded_image.tobytes()
                except Exception:
                    img_bytes = bytes(encoded_image)
                binary_data.append(img_bytes)
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
        try:
            dp.send_data(meta_data, binary_data)
        except Exception as e:
            print(f"[client] Error sending data: {e}")
            import traceback
            traceback.print_exc()

    time.sleep(0.01)  # Small sleep to prevent busy-waiting