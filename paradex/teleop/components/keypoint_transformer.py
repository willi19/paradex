from ..utils.vectorops import moving_average, normalize_vector
from ..constants import (
    OCULUS_JOINTS,
)
import numpy as np
from copy import deepcopy as copy
from scipy.spatial.transform import Rotation as R


class WristFrameHandPose:
    # This class extract wrist orientation and transforms the hand keypoints into the wrist frame
    def __init__(
        self,
    ):
        # Keypoint indices for knuckles
        self.knuckle_points = (
            OCULUS_JOINTS["knuckles"][0],
            OCULUS_JOINTS["knuckles"][-1],
        )

        self.step = 0

    # Function to find hand coordinates with respect to the wrist
    def _translate_coords(self, hand_coords):
        return copy(hand_coords) - hand_coords[0]

    # Create a coordinate frame for the hand
    def _get_coord_frame(self, index_knuckle_coord, pinky_knuckle_coord):
        palm_normal = normalize_vector(
            np.cross(index_knuckle_coord, pinky_knuckle_coord)
        )  # Current Z
        palm_direction = normalize_vector(
            index_knuckle_coord + pinky_knuckle_coord
        )  # Current Y
        cross_product = normalize_vector(
            np.cross(palm_direction, palm_normal)
        )  # Current X
        # palm_normal = np.array([1, 0, 0])
        # palm_direction = np.array([0, 1, 0])
        # cross_product = np.array([0, 0, 1])
        return [cross_product, palm_direction, palm_normal]

    # Create a coordinate frame for the arm
    def _get_hand_dir_frame(
        self, origin_coord, index_knuckle_coord, pinky_knuckle_coord
    ):
        # self.step += 1.0
        # self.bias = R.from_euler("xyz", [0, self.step, 0], degrees=True).as_matrix()
        # print(self.bias)
        palm_normal = normalize_vector(
            np.cross(index_knuckle_coord, pinky_knuckle_coord)
        )  # Allegro - X
        palm_direction = normalize_vector(
            index_knuckle_coord + pinky_knuckle_coord
        )  # Allegro - Z
        palm_direction = normalize_vector(
            palm_direction - np.dot(palm_direction, palm_normal) * palm_normal
        )
        cross_product = normalize_vector(
            index_knuckle_coord - pinky_knuckle_coord
        )  # Allegro - Y

        # palm_normal = np.array([1, 0, 0])
        # palm_direction = np.array([0, 1, 0])
        # cross_product = np.array([0, 0, 1])

        # palm_normal = palm_normal @ self.bias
        # palm_direction = palm_direction @ self.bias
        # cross_product = cross_product @ self.bias

        return [origin_coord, palm_normal, cross_product, palm_direction]

    def transform_keypoints(self, hand_coords):
        translated_coords = self._translate_coords(hand_coords)
        original_coord_frame = self._get_coord_frame(
            translated_coords[self.knuckle_points[0]],
            translated_coords[self.knuckle_points[1]],
        )

        # Finding the rotation matrix and rotating the coordinates
        rotation_matrix = np.linalg.solve(original_coord_frame, np.eye(3)).T
        transformed_hand_coords = (rotation_matrix @ translated_coords.T).T

        hand_dir_frame = self._get_hand_dir_frame(
            hand_coords[0],  # wrist point 3D
            translated_coords[self.knuckle_points[0]],
            translated_coords[self.knuckle_points[1]],
        )
        hand_dir_frame = np.array(hand_dir_frame)

        return transformed_hand_coords, hand_dir_frame

    def transform(self, data_type, hand_coords):

        # Shift the points to required axes
        transformed_hand_coords, translated_hand_coord_frame = self.transform_keypoints(
            hand_coords
        )

        # Passing the transformed coords into a moving average
        self.averaged_hand_coords = transformed_hand_coords

        ret = {"hand_coords": self.averaged_hand_coords}

        if data_type == "absolute":
            self.averaged_hand_frame = translated_hand_coord_frame
            ret["hand_frame"] = self.averaged_hand_frame
        return ret
