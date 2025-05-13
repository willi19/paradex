from ..utils.network import (
    create_pull_socket,
)
from ..constants import (
    ARM_LOW_RESOLUTION,
    ARM_HIGH_RESOLUTION,
    ARM_TELEOP_STOP,
    ARM_TELEOP_CONT,
    OCULUS_NUM_KEYPOINTS,
)
import numpy as np


class OculusVRHandReceiver:
    def __init__(
        self,
        host,
        oculus_port,
        button_port,
        teleop_reset_port,
    ):
        # Initializing the network socket for getting the raw right hand keypoints
        self.raw_keypoint_socket = create_pull_socket(host, oculus_port)
        self.button_keypoint_socket = create_pull_socket(host, button_port)
        self.teleop_reset_socket = create_pull_socket(host, teleop_reset_port)

    # Function to process the data token received from the VR
    def _process_data_token(self, data_token):
        return data_token.decode().strip()

    # Function to Extract the Keypoints from the String Token sent by the VR
    def _extract_data_from_token(self, token):
        data = self._process_data_token(token)
        data_type = "absolute" if data.startswith("absolute") else "relative"

        keypoint_vals = []
        vector_strings = data.split(":")[1].strip().split("|")
        for vector_str in vector_strings:
            vector_vals = vector_str.split(",")
            for float_str in vector_vals[:3]:
                keypoint_vals.append(float(float_str))

        return data_type, np.asanyarray(keypoint_vals).reshape(OCULUS_NUM_KEYPOINTS, 3)

    def get_input(self):
        raw_keypoints = self.raw_keypoint_socket.recv()
        # Getting the button feedback
        button_feedback = self.button_keypoint_socket.recv()
        # Getting the Teleop Reset Status
        pause_status = self.teleop_reset_socket.recv()
        if button_feedback == b"Low":
            button_feedback_num = ARM_LOW_RESOLUTION
        else:
            button_feedback_num = ARM_HIGH_RESOLUTION
        # Analyzing the Teleop Reset Status
        if pause_status == b"Low":
            pause_status = ARM_TELEOP_STOP
        else:
            pause_status = ARM_TELEOP_CONT
        # Processing the keypoints and publishing them
        data_type, keypoint = self._extract_data_from_token(raw_keypoints)
        ret = {
            "data_type": data_type,
            "keypoint": keypoint,
            "button_feedback_num": button_feedback_num,
            "pause_status": pause_status,
        }
        return ret

    def close(self):
        self.raw_keypoint_socket.close()
        self.button_keypoint_socket.close()
        self.teleop_reset_socket.close()

        print("Stopping the oculus keypoint extraction process.")
