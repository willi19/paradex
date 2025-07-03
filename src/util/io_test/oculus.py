from paradex.io.occulus.receiver import OculusReceiver, occulus_joint_name, occulus_joint_parent
from paradex.visualization.hand import HandVisualizer

from threading import Event
from paradex.utils.keyboard_listener import listen_keyboard
import time

if __name__ == "__main__":
    occulus = OculusReceiver()
    visualizer = HandVisualizer(occulus_joint_parent)
    
    stop_event = Event()
    listen_keyboard({"q":stop_event})
    
    init_wrist_pose = None
    
    while not stop_event.is_set():
        hand_data = occulus.get_data()
        if hand_data['Right'] is None:
            continue
        
        if init_wrist_pose is None:
            init_wrist_pose = hand_data['Right'][0,:3,3].copy()
        for i in range(0, len(hand_data['Right'])):
            hand_data['Right'][i][:3,3] -= init_wrist_pose
            
        visualizer.update_sphere_positions(hand_data['Right'])
        time.sleep(0.1)
        
    visualizer.stop()    
    occulus.quit()
    