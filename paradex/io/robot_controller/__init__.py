from paradex.utils.system import network_info

def get_arm(arm_name):
    # if arm_name == "franka":
    #     from .franka_controller import FrankaController
    #     return FrankaController()
    if arm_name == "xarm":
        from .xarm_controller import XArmController
        return XArmController(**network_info[arm_name]["param"])

def get_hand(hand_name):
    if hand_name == "inspire":
        from .inspire_controller import InspireController
        return InspireController(**network_info["inspire_usb"]["param"])
    
    if hand_name == "allegro":
        from .allegro_controller import AllegroController
        return AllegroController(**network_info[hand_name]["param"])