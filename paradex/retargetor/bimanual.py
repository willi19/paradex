import numpy as np
from paradex.transforms.coordinate import DEVICE2WRIST, DEVICE2GLOBAL
from paradex.retargetor.hand_regargetor import inspire, allegro, gripper


class Retargetor:
    def __init__(self, arm_name=None, hand_name=None):
        self.arm_name = arm_name
        self.hand_name = hand_name

        valid_arms = [None, "xarm", "franka", "openarm"]
        valid_hands = [None, "inspire", "allegro", "gripper"]
        if arm_name not in valid_arms:
            raise ValueError("Invalid arm name")
        if hand_name not in valid_hands:
            raise ValueError("Invalid hand name")

        self.hand_retargetor = None
        if self.hand_name == "inspire":
            self.hand_retargetor = inspire
        elif self.hand_name == "allegro":
            self.hand_retargetor = allegro
        elif self.hand_name == "gripper":
            self.hand_retargetor = gripper

        if self.arm_name is not None:
            arm_key = "xarm" if self.arm_name == "openarm" else self.arm_name
            # OpenArm uses xarm coordinate mapping until a dedicated calibration is available.
            self.device2wrist = DEVICE2WRIST[arm_key].copy()
            self.device2global = DEVICE2GLOBAL[arm_key].copy()
        else:
            self.device2wrist = DEVICE2WRIST[self.hand_name].copy()
            self.device2global = np.eye(4)

        self.init_human_pose_left = None
        self.init_human_pose_right = None
        self.init_robot_pose_left = None
        self.init_robot_pose_right = None
        self.last_arm_pose_left = None
        self.last_arm_pose_right = None

    def _compute_arm_action(self, human_wrist_pose, init_human_pose, init_robot_pose):
        delta_r = (
            self.device2wrist[:3, :3].T
            @ np.linalg.inv(init_human_pose[:3, :3])
            @ human_wrist_pose[:3, :3]
            @ self.device2wrist[:3, :3]
        )
        delta_t = human_wrist_pose[:3, 3] - init_human_pose[:3, 3]

        robot_wrist_pose = np.zeros((4, 4))
        robot_wrist_pose[:3, :3] = init_robot_pose[:3, :3] @ delta_r
        robot_wrist_pose[:3, 3] = delta_t + init_robot_pose[:3, 3]
        robot_wrist_pose[3, 3] = 1
        return robot_wrist_pose

    def get_action(self, data):
        if data is None or data.get("Left") is None or data.get("Right") is None:
            return None, None

        if self.init_human_pose_left is None:
            self.init_human_pose_left = data["Left"]["wrist"].copy()
        if self.init_human_pose_right is None:
            self.init_human_pose_right = data["Right"]["wrist"].copy()

        left_wrist_pose = self._compute_arm_action(
            data["Left"]["wrist"],
            self.init_human_pose_left,
            self.init_robot_pose_left,
        )
        right_wrist_pose = self._compute_arm_action(
            data["Right"]["wrist"],
            self.init_human_pose_right,
            self.init_robot_pose_right,
        )

        self.last_arm_pose_left = left_wrist_pose.copy()
        self.last_arm_pose_right = right_wrist_pose.copy()

        if self.hand_name is not None:
            left_hand_action = self.hand_retargetor(data["Left"])
            right_hand_action = self.hand_retargetor(data["Right"])
        else:
            left_hand_action = None
            right_hand_action = None

        arm_action = {"Left": left_wrist_pose, "Right": right_wrist_pose}
        hand_action = {"Left": left_hand_action, "Right": right_hand_action}
        return arm_action, hand_action

    def start(self, home_pose_left, home_pose_right=None):
        if home_pose_right is None and isinstance(home_pose_left, dict):
            home_pose_right = home_pose_left.get("Right") or home_pose_left.get("right")
            home_pose_left = home_pose_left.get("Left") or home_pose_left.get("left")

        if home_pose_left is None or home_pose_right is None:
            raise ValueError("Both left and right home poses are required.")

        self.init_robot_pose_left = home_pose_left.copy()
        self.init_robot_pose_right = home_pose_right.copy()
        self.last_arm_pose_left = home_pose_left.copy()
        self.last_arm_pose_right = home_pose_right.copy()
        self.init_human_pose_left = None
        self.init_human_pose_right = None

    def stop(self):
        self.init_human_pose_left = None
        self.init_human_pose_right = None
        if self.last_arm_pose_left is not None:
            self.init_robot_pose_left = self.last_arm_pose_left.copy()
        if self.last_arm_pose_right is not None:
            self.init_robot_pose_right = self.last_arm_pose_right.copy()
