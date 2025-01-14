import cv2
import mediapipe as mp

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# Set up the MediaPipe Hands model
hands = mp_hands.Hands(
    static_image_mode=True,  # Use True for static images, False for real-time
    max_num_hands=2,        # Maximum number of hands to detect
    min_detection_confidence=0.5  # Detection confidence threshold
)

def detect_keypoint(image):
    # Process the image to detect hands
    results = hands.process(image_rgb)

    # Check if hands are detected
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            print("Hand landmarks detected:")
            for idx, landmark in enumerate(hand_landmarks.landmark):
                # Print the normalized x, y, z coordinates of each keypoint
                print(f"Landmark {idx}: (x: {landmark.x}, y: {landmark.y}, z: {landmark.z})")

            # Draw the hand landmarks on the image
            mp_drawing.draw_landmarks(
                image,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS,
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style()
            )
        return results.multi_hand_landmarks
    else:
        return None

    
