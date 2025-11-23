import time
import cv2
import threading
from threading import Event

from paradex.io.camera_system.camera_reader import MultiCameraReader
from paradex.io.capture_pc.data_sender import DataPublisher
from paradex.io.capture_pc.command_sender import CommandReceiver

for _ in range(3):
    # Initialize components
    dp = DataPublisher(port=1234, name="camera_stream")
    exit_event = Event()
    # cr = CommandReceiver(event_dict={"exit": exit_event}, port=6890)

    # Initialize multi-camera reader
    reader = MultiCameraReader()

    last_frame_ids = {name: 0 for name in reader.camera_names}
    start_time = time.time()
    while time.time() - start_time < 10:  # Run for 60 seconds
        # Get images from all cameras
        images_data = reader.get_images(copy=True)
        
        # Prepare data to send
        send_data = {}
        
        for camera_name, (image, frame_id) in images_data.items():
            # Only send if we have a new frame
            if frame_id > last_frame_ids[camera_name] and frame_id > 0:
                image = cv2.resize(image, (image.shape[1]//8, image.shape[0]//8))
                # Get drop count for this camera
                encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 85]
                success, encoded_image = cv2.imencode('.jpg', image, encode_param)
                
                if success:
                    send_data[camera_name] = {
                        'frame_id': int(frame_id),
                        # 'image': encoded_image.tobytes(),
                        # 'shape': image.shape,
                    }
                    last_frame_ids[camera_name] = frame_id
        
        # Publish if we have new data
        if send_data:
            dp.send_data(send_data)
        
        time.sleep(0.01)  # Small sleep to prevent busy-waiting

    # Cleanup
    reader.close()
    dp.close()
    # cr.end()
    print("Camera streaming client stopped.")