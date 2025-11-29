
from threading import Event
from paradex.utils.keyboard_listener import listen_keyboard
import time
import argparse

from paradex.visualization.skeleton.hand import HandVisualizer

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', required=True, choices=['xsens', 'occulus'])
    
    args = parser.parse_args()

    if args.device == 'xsens': 
        from paradex.io.teleop import XSensReceiver, xsens_joint_name, xsens_joint_parent_name
        receiver = XSensReceiver()
        skeleton_info = {child:parent for child, parent in zip(xsens_joint_name, xsens_joint_parent_name)}
        joint_name_list = xsens_joint_name

    if args.device =='occulus':
        from paradex.io.teleop import OculusReceiver, occulus_hand_joint_name, occulus_hand_joint_parent_name
        receiver = OculusReceiver()
        skeleton_info = {child:parent for child, parent in occulus_hand_joint_parent_name.items()}
        joint_name_list = occulus_hand_joint_name
        
    visualizer = HandVisualizer(skeleton_info)
    stop_event = Event()
    listen_keyboard({"q":stop_event})
    
    init_wrist_mid_pose = None
    
    while not stop_event.is_set():
        start_time = time.time()
        hand_data = receiver.get_data()
        if hand_data['Right'] is None or hand_data['Left'] is None:
            continue
        
        if init_wrist_mid_pose is None:
            init_wrist_mid_pose = (hand_data['Right']['wrist'][:3, 3] + hand_data['Left']['wrist'][:3, 3]) / 2

        for side in ['Left', 'Right']:
            for joint_name in joint_name_list:
                hand_data[side][joint_name][:3, 3] -= init_wrist_mid_pose
            
        visualizer.update_sphere_positions(hand_data)
        
        time_lapse = time.time() - start_time
        if time_lapse > 0.02:
            continue
        time.sleep(max(0.02 - time_lapse, 0))

    visualizer.stop()    
    receiver.quit()
    