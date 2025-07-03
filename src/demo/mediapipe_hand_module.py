import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# Draw on image
from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2
import numpy as np
import cv2
from pathlib import Path

MEDIAPIPE_HANDDETECTOR = Path(__file__).absolute().parent/'third_party/hand_landmarker.task'

MARGIN = 10  # pixels
FONT_SIZE = 1
FONT_THICKNESS = 1
HANDEDNESS_TEXT_COLOR = (88, 205, 54) # vibrant green


class Hand_Module:
  def __init__(self):
  
    # NOTE: options from github
    # STEP 2: Create an HandLandmarker object.
    # base_options = python.BaseOptions(model_asset_path='hand_landmarker.task')
    # options = vision.HandLandmarkerOptions(base_options=base_options,
    #                                        num_hands=2)
    # detector = vision.HandLandmarker.create_from_options(options)

    BaseOptions = mp.tasks.BaseOptions
    HandLandmarker = mp.tasks.vision.HandLandmarker
    HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
    VisionRunningMode = mp.tasks.vision.RunningMode

    options = HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=MEDIAPIPE_HANDDETECTOR),
        running_mode=VisionRunningMode.IMAGE)

    self.detector = HandLandmarker.create_from_options(options)


  def return_pixel_keypoints(self, detection_result, img_h, img_w):
    hand_landmarks_list = detection_result.hand_landmarks
    handedness_list = detection_result.handedness

    hand_result = {}
    for hand_landmarks, handedness in zip(hand_landmarks_list, handedness_list):
      side = handedness[0].category_name
      result = []
      for land_mark in hand_landmarks:
        if land_mark is not None:
        # check threshold if needed.
          output = solutions.drawing_utils._normalized_to_pixel_coordinates(land_mark.x, land_mark.y, img_w, img_h)
          if output is not None:
            x_px, y_px = output
            result.append([x_px, y_px])
          else:
            result.append([0, 0])
        else:
          result.append([0, 0])
      if len(result)>0:
        hand_result[side] = np.array(result)

    return hand_result


    

  def draw_landmarks_on_image(self, rgb_image, detection_result):
    hand_landmarks_list = detection_result.hand_landmarks
    handedness_list = detection_result.handedness
    annotated_image = np.copy(rgb_image)

    # Loop through the detected hands to visualize.
    for idx in range(len(hand_landmarks_list)):
      hand_landmarks = hand_landmarks_list[idx]
      handedness = handedness_list[idx]

      # Draw the hand landmarks.
      hand_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
      hand_landmarks_proto.landmark.extend([
        landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z) for landmark in hand_landmarks
      ])
      solutions.drawing_utils.draw_landmarks(
            annotated_image,
            hand_landmarks_proto,
            solutions.hands.HAND_CONNECTIONS,
            solutions.drawing_styles.get_default_hand_landmarks_style(),
            solutions.drawing_styles.get_default_hand_connections_style()
        )

      # Get the top left corner of the detected hand's bounding box.
      height, width, _ = annotated_image.shape
      x_coordinates = [landmark.x for landmark in hand_landmarks]
      y_coordinates = [landmark.y for landmark in hand_landmarks]
      text_x = int(min(x_coordinates) * width)
      text_y = int(min(y_coordinates) * height) - MARGIN

      # Draw handedness (left or right hand) on the image.
      cv2.putText(annotated_image, f"{handedness[0].category_name}",
                  (text_x, text_y), cv2.FONT_HERSHEY_DUPLEX,
                  FONT_SIZE, HANDEDNESS_TEXT_COLOR, FONT_THICKNESS, cv2.LINE_AA)

    return annotated_image

  def inference(self, numpy_image: np.ndarray):
    """Run inference on the input image and return the detection results."""
    numpy_image = cv2.cvtColor(numpy_image, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=numpy_image)
    hand_detection_result = self.detector.detect(mp_image)

    img_h, img_w = numpy_image.shape[:2]
    returned_pixels = self.return_pixel_keypoints(hand_detection_result, img_h, img_w)

    # annotated_image = self.draw_landmarks_on_image(mp_image.numpy_view(), hand_detection_result)
    # cv2.imwrite('test.png',cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR))
    # # cv2.waitKey(0)

    return returned_pixels
  
  
if __name__ == "__main__":
    hand_module = Hand_Module()
    image = cv2.imread('/home/jisoo/data2/paradex/test/image_test.png')
    
    returned_pixels = hand_module.inference(image)
    # print(detection_result)
