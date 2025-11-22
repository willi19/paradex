import time
import cv2
import threading
from threading import Event

from paradex.io.camera_system.camera_reader import MultiCameraReader
from paradex.io.capture_pc.data_sender import DataPublisher
from paradex.io.capture_pc.command_sender import CommandReceiver

# Initialize components
dp = DataPublisher(port=1234, name="camera_stream")
exit_event = Event()
cr = CommandReceiver(event_dict={"exit": exit_event}, port=6890)

# Initialize multi-camera reader
reader = MultiCameraReader()

def stream_loop():
    """Main loop to read camera images and publish compressed data"""
    last_frame_ids = {name: 0 for name in reader.camera_names}
    
    while not exit_event.is_set():
        # Get images from all cameras
        images_data = reader.get_images(copy=True)
        
        # Prepare data to send
        send_data = {}
        
        for camera_name, (image, frame_id) in images_data.items():
            # Only send if we have a new frame
            if frame_id > last_frame_ids[camera_name] and frame_id > 0:
                # Get drop count for this camera
                encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 85]
                success, encoded_image = cv2.imencode('.jpg', image, encode_param)
                
                if success:
                    send_data[camera_name] = {
                        'frame_id': int(frame_id),
                        'image': encoded_image.tobytes(),
                        'shape': image.shape,
                    }
                    last_frame_ids[camera_name] = frame_id
        
        # Publish if we have new data
        if send_data:
            dp.send_data(send_data)
        
        time.sleep(0.01)  # Small sleep to prevent busy-waiting

# Start streaming thread
stream_thread = threading.Thread(target=stream_loop, daemon=True)
stream_thread.start()

print("Camera streaming client started. Waiting for exit command...")

# Wait for exit
exit_event.wait()

# Cleanup
reader.close()
dp.close()
cr.end()
print("Camera streaming client stopped.")