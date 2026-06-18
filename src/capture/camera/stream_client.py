import argparse
import time
import cv2
from threading import Event

from paradex.io.camera_system.camera_reader import MultiCameraReader
from paradex.io.capture_pc.data_sender import DataPublisher
from paradex.io.capture_pc.command_sender import CommandReceiver
from paradex.utils.system import get_camera_list

parser = argparse.ArgumentParser()
parser.add_argument(
    "--cameras",
    type=str,
    default=None,
    help="comma-separated serials to publish; defaults to every local camera",
)
parser.add_argument(
    "--resize-divisor",
    type=int,
    default=8,
    help="downsample width/height before JPEG publish; 8 means 256x192",
)
args = parser.parse_args()
if args.resize_divisor < 1:
    raise ValueError("--resize-divisor must be >= 1")

local_cameras = get_camera_list()
if args.cameras:
    requested = {serial.strip() for serial in args.cameras.split(",") if serial.strip()}
    camera_names = [serial for serial in local_cameras if serial in requested]
else:
    camera_names = local_cameras
if not camera_names:
    raise SystemExit("No requested cameras are connected to this capture PC.")

# Initialize components
dp = DataPublisher(port=1234, name="camera_stream")
exit_event = Event()
cr = CommandReceiver(event_dict={"exit": exit_event}, port=6890)

# Initialize multi-camera reader
reader = MultiCameraReader(camera_names=camera_names)
print(f"Publishing cameras: {reader.camera_names} "
      f"(resize divisor={args.resize_divisor})")

last_frame_ids = {name: 0 for name in reader.camera_names}

while not exit_event.is_set():
    # Get images from all cameras
    images_data = reader.get_images(copy=True)
    
    meta_data = []
    binary_data = []
    
    for camera_name, (image, frame_id) in images_data.items():
        # Only send if we have a new frame
        if frame_id > last_frame_ids[camera_name] and frame_id > 0:
            image = cv2.resize(
                image,
                (image.shape[1] // args.resize_divisor,
                 image.shape[0] // args.resize_divisor),
            )
            encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 85]
            success, encoded_image = cv2.imencode('.jpg', image, encode_param)
            
            if success:
                meta_data.append({
                    'type': 'image',  # 데이터 타입
                    'name': camera_name,
                    'frame_id': int(frame_id),
                    'shape': tuple(int(x) for x in image.shape),
                    'data_index': len(binary_data)
                })
                # Add binary data
                binary_data.append(encoded_image)
                last_frame_ids[camera_name] = frame_id
    if meta_data:
        dp.send_data(meta_data, binary_data)

    time.sleep(0.01)  # Small sleep to prevent busy-waiting

# Cleanup
reader.close()
dp.close()
cr.end()
print("Camera streaming client stopped.")
