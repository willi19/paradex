import time
import cv2
from threading import Event
import os

from paradex.io.camera_system.camera_reader import MultiCameraReader
from paradex.io.capture_pc.data_sender import DataPublisher
from paradex.io.capture_pc.command_sender import CommandReceiver
from paradex.calibration.utils import extrinsic_dir
from paradex.utils.file_io import find_latest_directory
from paradex.utils.system import get_camera_list

TARGET_CAMERAS = [
    "25452066",  # capture18
    "23012641",  # capture14
    "23022639",  # capture12
]
STREAM_SIZE = (640, 480)  # (width, height)

dp = DataPublisher(port=1234, name="camera_stream")

exit_event = Event()
save_event = Event()
cr = CommandReceiver(event_dict={"exit": exit_event, "save": save_event}, port=6890)

# Each capture PC should open only its locally attached target cameras.
local_cameras = set(get_camera_list())
selected_cameras = [cam for cam in TARGET_CAMERAS if cam in local_cameras]

if not selected_cameras:
    raise RuntimeError(
        f"No target cameras found on this PC. "
        f"local={sorted(local_cameras)}, target={TARGET_CAMERAS}"
    )

reader = MultiCameraReader(camera_names=selected_cameras)
last_frame_ids = {name: 0 for name in reader.camera_names}

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
            
            if save_event.is_set():
                save_name = find_latest_directory(root_dir)
                save_path = os.path.join(root_dir, save_name)
                
                if os.path.exists(os.path.join(save_path, "images", f"{camera_name}.png")):
                    print(f"Data for camera {camera_name} already saved, skipping.", save_remain)
                    continue  # Already saved for this camera

                cv2.imwrite(os.path.join(save_path, "images", f"{camera_name}.png"), cur_image)
                print(f"Saved data for camera {camera_name} at frame {frame_id} to {save_path}")
                save_remain -= 1
                save_id[camera_name] += 1

            cur_image = cv2.resize(cur_image, STREAM_SIZE)
            encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 85]
            success, encoded_image = cv2.imencode('.jpg', cur_image, encode_param)
            
            if success:
                meta_data.append({
                    'type': 'image',  # 데이터 타입
                    'name': camera_name,
                    'frame_id': int(frame_id),
                    'save_id': save_id[camera_name],
                    'shape': tuple(int(x) for x in cur_image.shape),
                    'data_index': len(binary_data)
                })
                # Add binary data
                binary_data.append(encoded_image)
                last_frame_ids[camera_name] = frame_id
        
    if save_event.is_set() and save_remain == 0:
        save_event.clear()
        save_remain = len(reader.camera_names)
        print("Completed saving data for all cameras.")
                
                
    if meta_data:
        dp.send_data(meta_data, binary_data)

    time.sleep(0.01)  # Small sleep to prevent busy-waiting
