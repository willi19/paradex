import cv2
import mediapipe as mp

class HandKeypointDetector:
    def __init__(self, static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5):
        """
        Initializes the MediaPipe Hands model with the given parameters.
        
        :param static_image_mode: Whether to treat the input images as static.
        :param max_num_hands: Maximum number of hands to detect.
        :param min_detection_confidence: Minimum confidence value for detection.
        """
        self.mp_hands = mp.solutions.hands
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles

        self.hands = self.mp_hands.Hands(
            static_image_mode=static_image_mode,
            max_num_hands=max_num_hands,
            min_detection_confidence=min_detection_confidence
        )

    def detect_keypoints(self, image):
        """
        Detects hand keypoints in a given image.
        
        :param image: The input image (in BGR format) for hand detection.
        :return: A list of detected hand landmarks, or None if no hands are detected.
        """
        # Convert the image to RGB as MediaPipe expects RGB format
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Process the image to detect hands
        results = self.hands.process(image_rgb)

        # Check if hands are detected
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                print("Hand landmarks detected:")
                for idx, landmark in enumerate(hand_landmarks.landmark):
                    # Print the normalized x, y, z coordinates of each keypoint
                    print(f"Landmark {idx}: (x: {landmark.x}, y: {landmark.y}, z: {landmark.z})")

                # Draw the hand landmarks on the image
                self.mp_drawing.draw_landmarks(
                    image,
                    hand_landmarks,
                    self.mp_hands.HAND_CONNECTIONS,
                    self.mp_drawing_styles.get_default_hand_landmarks_style(),
                    self.mp_drawing_styles.get_default_hand_connections_style()
                )
            return results.multi_hand_landmarks
        else:
            return None

    def __del__(self):
        """
        Ensures that the MediaPipe Hands model is released properly.
        """
        self.hands.close()