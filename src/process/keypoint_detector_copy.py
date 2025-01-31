from multiprocessing import Process, shared_memory, Value, Lock
import cv2
import numpy as np
from mediapipe import solutions
import time

class CameraStreamProcessor:
    def __init__(self, shared_mem, update_flag, lock, w, h, num_cameras=1):
        self.shared_memories = shared_memories
        self.update_flags = update_flags
        self.lock = lock
        self.w = w
        self.h = h
        self.num_cameras = num_cameras
        self.process_list = []
        self.detector = [solutions.hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        ) for _ in range(num_cameras)]

    def draw_landmarks(frame, hand_landmarks, connections=None, landmark_color=(0, 255, 0), connection_color=(255, 0, 0), radius=4, thickness=2):
        """
        Draw hand landmarks and connections on a given frame.

        :param frame: The input frame (BGR format).
        :param hand_landmarks: List of landmarks (normalized x, y, z coordinates).
        :param connections: List of tuples defining connections between landmarks (e.g., HAND_CONNECTIONS).
        :param landmark_color: Color for landmarks (default: green).
        :param connection_color: Color for connections (default: blue).
        :param radius: Radius of the landmark points.
        :param thickness: Thickness of the connection lines.
        """
        h, w, _ = frame.shape

        # Draw connections
        if connections:
            for start_idx, end_idx in connections:
                start = hand_landmarks[start_idx]
                end = hand_landmarks[end_idx]

                # Convert normalized coordinates to pixel coordinates
                start_point = (int(start.x * w), int(start.y * h))
                end_point = (int(end.x * w), int(end.y * h))

                # Draw the line
                cv2.line(frame, start_point, end_point, connection_color, thickness)

        # Draw landmarks
        for landmark in hand_landmarks:
            # Convert normalized coordinates to pixel coordinates
            x, y = int(landmark.x * w), int(landmark.y * h)
            
            # Draw the point
            cv2.circle(frame, (x, y), radius, landmark_color, cv2.FILLED)

        return frame

    

if __name__ == "__main__":
    