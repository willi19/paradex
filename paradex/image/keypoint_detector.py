from multiprocessing import Process, shared_memory, Value, Lock
import cv2
import numpy as np
import time
import mediapipe as mp

mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# OpenCV settings for drawing
POINT_COLOR = (0, 255, 0)  # Green for points
LINE_COLOR = (255, 0, 0)   # Blue for lines
POINT_RADIUS = 5
LINE_THICKNESS = 2

def draw_landmarks_opencv(image, landmarks, connections):
    """
    Draw landmarks and connections on the image using OpenCV.
    :param image: The image to draw on.
    :param landmarks: List of landmarks detected by MediaPipe.
    :param connections: List of connections between landmarks.
    """
    h, w, _ = image.shape

    # Draw landmarks
    for idx, landmark in enumerate(landmarks):
        x = int(landmark.x * w)
        y = int(landmark.y * h)
        cv2.circle(image, (x, y), POINT_RADIUS, POINT_COLOR, -1)

    # Draw connections
    if connections:
        for connection in connections:
            start_idx, end_idx = connection
            start_landmark = landmarks[start_idx]
            end_landmark = landmarks[end_idx]

            x1, y1 = int(start_landmark.x * w), int(start_landmark.y * h)
            x2, y2 = int(end_landmark.x * w), int(end_landmark.y * h)
            cv2.line(image, (x1, y1), (x2, y2), LINE_COLOR, LINE_THICKNESS)


class CameraStreamProcessor:
    def __init__(self, shared_mem, update_flag, lock, w, h, num_cameras=4):
        self.shared_memories = shared_memories
        self.update_flags = update_flags
        self.lock = lock
        self.w = w
        self.h = h
        self.num_cameras = num_cameras
        self.process_list = []

    def streaming_start(self):
        """
        Start the processing streams for all cameras.
        """
        for i in range(self.num_cameras):
            process = Process(target=self.stream_run, args=(i,), daemon=True)
            self.process_list.append(process)
            process.start()

    def streaming_stop(self):
        """
        Stop all camera stream processes.
        """
        for process in self.process_list:
            if process.is_alive():
                process.terminate()
            process.join()
        print("All processes stopped.")

    def stream_run(self, cam_id):
        update_flag = self.update_flags[cam_id]
        shm_info = self.shared_memories[cam_id]
        shm_array = shm_info["array"]

        static_image_mode = False
        model_complexity = 1
        enable_segmentation = False
        min_detection_confidence = 0.5

        # Initialize MediaPipe Pose
        with mp_pose.Pose(
            static_image_mode=static_image_mode,
            model_complexity=model_complexity,
            enable_segmentation=enable_segmentation,
            min_detection_confidence=min_detection_confidence,
            smooth_landmarks=False,
        ) as pose:


            while True:
                if update_flag.value == 1:  # New frame available
                    with shm_info["lock"]:
                        frame = np.copy(shm_array)
                        update_flag.value = 0  # Reset update flag

                    print(f"Processing frame {cam_id}...")
                    # # Convert to RGB and process
                    # img = cv2.resize(frame, (160, 120))
                    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                    # Process the frame with the Pose model
                    results = pose.process(img_rgb)

                    # # Draw landmarks
                    # print(f"Found {len(results.multi_hand_landmarks)} hands.")
                    if results.pose_landmarks:
                        draw_landmarks_opencv(
                            frame,  # Original frame
                            results.pose_landmarks.landmark,  # List of landmarks
                            mp_pose.POSE_CONNECTIONS  # Connections to draw
                        )

                    # Write annotated frame back to shared memory
                    with shm_info["lock"]:
                        np.copyto(shm_array, frame)
                        update_flag.value = 2  # Mark as processed
    

if __name__ == "__main__":
    num_cameras = 4
    frame_shape = (480, 640, 3)
    shared_memories = {}
    update_flags = {}

    from ..camera.camera_loader import CameraManager
    manager = CameraManager(num_cameras=num_cameras, duration=300, is_streaming=True, shared_memories=shared_memories, update_flags=update_flags)
    manager.start()
    
    # Initialize and start the viewer
    from ..viewer.multiimage import MultiStreamViewer
    viewer = MultiStreamViewer(640, 480, True, shared_memories, update_flags, 4, 2)
    viewer.stream_start()

    # Initialize and start the processor
    processor = CameraStreamProcessor(shared_memories, update_flags, 640, 480, num_cameras)
    processor.streaming_start()

    try:
        print("Processing streams. Press Ctrl+C to exit.")
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nTerminating program...")
        viewer.stream_stop()
        processor.streaming_stop()
        manager.signal_handler()
        print("Stopping processes...")
    finally:

        # Clean up shared memory
        for shm_info in shared_memories.values():
            shm_info["shm"].close()
            shm_info["shm"].unlink()
