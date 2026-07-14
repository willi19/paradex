from paradex.utils.system import network_info

def get_arm(arm_name, **kwargs):
    # if arm_name == "franka":
    #     from .franka_controller import FrankaController
    #     return FrankaController()
    if arm_name == "xarm":
        # from .xarm_controller import XArmController
        # return XArmController(**network_info[arm_name]["param"])
        from .xarm_controller_ros import XArmControllerROS
        return XArmControllerROS(**{**network_info["xarm"]["param"], **kwargs})
    if arm_name == "xarm_ik":
        from .xarm_controller_ros_ik import XArmControllerROSIK
        return XArmControllerROSIK(**{**network_info["xarm"]["param"], **kwargs})
    if arm_name == "openarm":
        from .openarm_state_receiver import OpenArmStateReceiver
        return OpenArmStateReceiver()

def get_hand(hand_name, tactile = False, ip = False, hand_side="right"):
    if hand_name == "inspire":
        if ip:
            from .deprecated.inspire_controller_ip import InspireControllerIP
            return InspireControllerIP(**network_info["inspire_ip"]["param"], tactile=tactile)
        else:
            from .inspire_controller import InspireController
            return InspireController(**network_info["inspire_usb"]["param"], tactile=tactile)
        
    if hand_name == "inspire_f1":
            # from .inspire_f1_controller import InspireF1Controller
            # return InspireF1Controller(**network_info["inspire_f1"]["param"], tactile=tactile)
        from .inspire_f1_state_receiver import InspireF1Controller
        return InspireF1Controller(hand_side=hand_side)
        
    if hand_name == "allegro":
        from .allegro_controller_ros2 import AllegroController
        return AllegroController(**network_info[hand_name]["param"])

    if hand_name == "allegro_v5":
        from .allegro_v5_controller_ros2 import AllegroController
        return AllegroController(hand_side=hand_side, tactile=tactile)

    if hand_name == "robotiq_2f85":
        from .robotiq_2f85_controller_ros2 import Robotiq2F85ControllerROS2
        params = network_info.get("robotiq_2f85", {}).get("param", {})
        return Robotiq2F85ControllerROS2(**params)

    if hand_name in ("wuji", "wuji_direct", "wuji_hybrid"):
        from .wuji_controller_ros2 import WujiControllerROS2
        params = network_info.get("wuji", {}).get("param", {})
        return WujiControllerROS2(hand_side=hand_side, **params)

    if hand_name == "kistar":
        from .kistarcontroller import KistarController
        return KistarController()
