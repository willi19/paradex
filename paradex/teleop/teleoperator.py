import hydra

from .utils.timer import FrequencyTimer
from .constants import VR_FREQ, ARM_TELEOP_STOP, ARM_TELEOP_CONT
import threading
import numpy as np


class TeleOperator:
    """
    Returns all the teleoperation processes. Start the list of processes
    to run the teleop.
    """

    def __init__(self, configs):
        self.configs = configs
        self.transformer = self._start_component(self.configs.robot.transformer)
        self.receiver = self._start_component(self.configs.robot.receiver)
        self.retargeter = self._start_component(self.configs.robot.retargeter)

        self.timer = FrequencyTimer(VR_FREQ)
        self.retargeted_action = None
        self.valid = False
        self.lock = threading.Lock()

    # Function to start the components
    def _start_component(self, configs):
        component = hydra.utils.instantiate(configs)[0]
        return component

    def get_retargeted_action(self):
        with self.lock:
            if not self.valid:
                print("Invalid")
                return None
            return self.retargeted_action

    def run(self):
        self.thread = threading.Thread(target=self.stream)
        self.thread.start()

    def stream(self):
        while True:
            self.timer.start_loop()

            # Get the input from the receiver
            received_input = self.receiver.get_input()

            # # Transform the input data
            transformed_keypoint = self.transformer.transform(
                received_input["data_type"], received_input["keypoint"]
            )
            if "hand_frame" not in transformed_keypoint or np.all(
                transformed_keypoint["hand_frame"][0] == 0
            ):
                # print("Hand frame not found")
                self.valid = False
                continue
            # # Retarget the transformed data
            retargeted_action = self.retargeter.retarget(
                transformed_keypoint["hand_coords"],
                transformed_keypoint["hand_frame"],
                ARM_TELEOP_CONT if self.valid else ARM_TELEOP_STOP,
            )
            # print(retargeted_action["endeff_coords"], "Retargeted Action")
            self.retargeted_action = retargeted_action
            self.retargeted_action["hand_frame"] = transformed_keypoint["hand_frame"]
            self.valid = True
