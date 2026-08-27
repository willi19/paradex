from functools import partial

import numpy as np
from paradex.transforms.coordinate import DEVICE2WRIST,  DEVICE2GLOBAL
from paradex.retargetor.hand_regargetor import (
    allegro_v5,
    inspire,
    allegro,
    inspire_f1,
    kistar,
    robotiq_2f85,
    wuji,
    wuji_direct,
    wuji_hybrid,
)
_HAND_RETARGETORS = {
    "inspire": inspire,
    "inspire_dftp": inspire,
    "allegro": allegro,
    "inspire_f1": inspire_f1,
    "kistar": kistar,
    "allegro_v5": allegro_v5,
    "robotiq_2f85": robotiq_2f85,
    "wuji": wuji,
    "wuji_direct": wuji_direct,
    "wuji_hybrid": wuji_hybrid,
}


def _resolve_hand(name, is_right=True, scale=1.0):
    if name is None:
        return None
    if name not in _HAND_RETARGETORS:
        raise ValueError(f"Invalid hand name: {name}")
    fn = _HAND_RETARGETORS[name]
    if name in ("wuji", "wuji_direct", "wuji_hybrid"):
        return partial(fn, is_right=is_right, scale=scale)
    if name in ("inspire", "inspire_dftp"):
        return partial(fn, is_right=is_right)
    if name == "inspire_f1":
        return partial(fn, is_right=is_right)
    return fn


class Retargetor(): # Input is only from Xsens
    def __init__(self, arm_name=None, hand_name=None, hand_side="Right",
                 hand_name_left=None, hand_name_right=None, hand_scale=1.0,
                 teleop_name=None):
        self.arm_name = arm_name
        self.teleop_name = teleop_name
        self.hand_name = hand_name
        self.hand_name_left = hand_name_left
        self.hand_name_right = hand_name_right
        self.hand_scale = float(hand_scale)
        if not np.isfinite(self.hand_scale) or self.hand_scale <= 0.0:
            raise ValueError(f"hand_scale must be a positive finite number, got {hand_scale}")

        if arm_name not in [None, "xarm", "franka"]:
            raise ValueError("Invalid arm name")
        if hand_side not in ["Right", "Left", "Bimanual"]:
            raise ValueError("Invalid hand side")

        self.hand_side = hand_side

        if self.hand_side == "Bimanual":
            left = hand_name_left if hand_name_left is not None else hand_name
            right = hand_name_right if hand_name_right is not None else hand_name
            self.hand_retargetor = {
                "Left": _resolve_hand(
                    left,
                    is_right=False,
                    scale=self.hand_scale,
                ),
                "Right": _resolve_hand(
                    right,
                    is_right=True,
                    scale=self.hand_scale,
                ),
            }
        else:
            self.hand_retargetor = _resolve_hand(
                hand_name,
                is_right=(self.hand_side != "Left"),
                scale=self.hand_scale,
            )

        if self.hand_side == "Bimanual":
            if self.arm_name is not None:
                wrist_key_prefix = (
                    self.arm_name + "_vive"
                    if self.teleop_name == "vive"
                    else self.arm_name
                )
                self.device2wrist = {
                    "Left": DEVICE2WRIST[wrist_key_prefix + "_Left"].copy(),
                    "Right": DEVICE2WRIST[wrist_key_prefix + "_Right"].copy(),
                }
                self.device2global = DEVICE2GLOBAL[self.arm_name].copy()
            else:
                left_name = hand_name_left if hand_name_left is not None else hand_name
                right_name = hand_name_right if hand_name_right is not None else hand_name
                self.device2wrist = {
                    "Left": DEVICE2WRIST[left_name].copy(),
                    "Right": DEVICE2WRIST[right_name].copy(),
                }
                self.device2global = np.eye(4)
        elif self.arm_name is not None:
            wrist_key_prefix = (
                self.arm_name + "_vive"
                if self.teleop_name == "vive"
                else self.arm_name
            )
            self.device2wrist = DEVICE2WRIST[
                wrist_key_prefix + "_" + hand_side
            ].copy()
            self.device2global = DEVICE2GLOBAL[self.arm_name].copy()
        else:
            self.device2wrist = DEVICE2WRIST[self.hand_name].copy()
            self.device2global = np.eye(4)

    def _compute_wrist_pose(self, side, data):
        device2wrist = self.device2wrist[side] if isinstance(self.device2wrist, dict) else self.device2wrist
        init_human = self.init_human_pose[side]
        init_robot = self.init_robot_pose[side]

        if init_human is None:
            init_human = data[side]["wrist"].copy()
            self.init_human_pose[side] = init_human

        delta_R = device2wrist[:3,:3].T @ np.linalg.inv(init_human[:3,:3]) @ data[side]["wrist"][:3,:3] @ device2wrist[:3,:3]
        delta_t_device = data[side]["wrist"][:3,3] - init_human[:3,3]
        delta_t = self.device2global[:3, :3] @ delta_t_device

        robot_wrist_pose = np.zeros((4,4))
        robot_wrist_pose[:3,:3] = init_robot[:3,:3] @ delta_R
        robot_wrist_pose[:3,3] = init_robot[:3,3] + delta_t
        robot_wrist_pose[3,3] = 1

        self.last_arm_pose[side] = robot_wrist_pose.copy()
        return robot_wrist_pose

    def get_action(self, data):
        if self.hand_side == "Bimanual":
            arm_action_left = self._compute_wrist_pose("Left", data)
            arm_action_right = self._compute_wrist_pose("Right", data)

            rt_left = self.hand_retargetor.get("Left")
            rt_right = self.hand_retargetor.get("Right")
            ergonomics = data.get("ergonomics", {})
            left_name = self.hand_name_left or self.hand_name
            right_name = self.hand_name_right or self.hand_name
            left_kwargs = (
                {"ergonomics": ergonomics.get("Left")}
                if left_name in ("inspire", "inspire_dftp", "allegro_v5")
                else {}
            )
            right_kwargs = (
                {"ergonomics": ergonomics.get("Right")}
                if right_name in ("inspire", "inspire_dftp", "allegro_v5")
                else {}
            )
            hand_action_left = (
                rt_left(data["Left"], **left_kwargs)
                if rt_left is not None
                else None
            )
            hand_action_right = (
                rt_right(data["Right"], **right_kwargs)
                if rt_right is not None
                else None
            )

            if (
                left_name in ("inspire", "inspire_dftp")
                and hand_action_left[5] >= 600.0
            ):
                hand_action_left[4] = 1000.0
            if (
                right_name in ("inspire", "inspire_dftp")
                and hand_action_right[5] >= 600.0
            ):
                hand_action_right[4] = 1000.0

            return arm_action_left, arm_action_right, hand_action_left, hand_action_right
        else:
            arm_action = self._compute_wrist_pose(self.hand_side, data)

            if self.hand_retargetor is not None:
                kwargs = {}
                if self.hand_name in ("inspire", "inspire_dftp", "allegro_v5"):
                    kwargs["ergonomics"] = data.get("ergonomics", {}).get(
                        self.hand_side
                    )
                hand_action = self.hand_retargetor(
                    data[self.hand_side],
                    **kwargs,
                )
            else:
                hand_action = None

            return arm_action, hand_action


    def start(self, home_pose):
        def reset_hand_retargeter(retargeter):
            if hasattr(retargeter, "reset"):
                retargeter.reset()

        if isinstance(self.hand_retargetor, dict):
            for retargeter in self.hand_retargetor.values():
                reset_hand_retargeter(retargeter)
        else:
            reset_hand_retargeter(self.hand_retargetor)

        if self.hand_side == "Bimanual":
            self.init_robot_pose = {
                "Left": home_pose["Left"].copy(),
                "Right": home_pose["Right"].copy(),
            }
            self.last_arm_pose = {
                "Left": home_pose["Left"].copy(),
                "Right": home_pose["Right"].copy(),
            }
            self.init_human_pose = {"Left": None, "Right": None}
        else:
            self.init_robot_pose = {self.hand_side: home_pose.copy()}
            self.last_arm_pose = {self.hand_side: home_pose.copy()}
            self.init_human_pose = {self.hand_side: None}

    def stop(self):
        if self.hand_side == "Bimanual":
            self.init_human_pose = {"Left": None, "Right": None}
            self.init_robot_pose = {
                "Left": self.last_arm_pose["Left"].copy(),
                "Right": self.last_arm_pose["Right"].copy(),
            }
        else:
            self.init_human_pose = {self.hand_side: None}
            self.init_robot_pose = {self.hand_side: self.last_arm_pose[self.hand_side].copy()}
