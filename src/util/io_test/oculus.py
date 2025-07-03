from paradex.io.occulus.receiver import OculusReceiver, occulus_joint_name, occulus_joint_parent
from paradex.visualization.hand import HandVisualizer

from threading import Event
from paradex.utils.keyboard_listener import listen_keyboard
import time
import numpy as np

unity2Euclidian_mat = np.array([
                        [ 0,  0,  1, 0],   # new X = old Z
                        [-1,  0,  0, 0],   # new Y = –old X
                        [ 0,  1,  0, 0],   # new Z = old Y
                        [ 0,  0,  0, 1],
                    ], dtype=float)



if __name__ == "__main__":
    occulus = OculusReceiver()
    visualizer = HandVisualizer(occulus_joint_parent)
    
    stop_event = Event()
    listen_keyboard({"q":stop_event})
    
    init_wrist_mid_pose = None
    
    while not stop_event.is_set():
        start_time = time.time()
        hand_data = occulus.get_data()
        if hand_data['Right'] is None or hand_data['Left'] is None:
            continue

        hand_data_array = np.stack((hand_data['Left'], hand_data['Right']), axis=0)
        hand_data_array = unity2Euclidian_mat.reshape(1, 1, 4, 4) @ hand_data_array @ unity2Euclidian_mat.T.reshape(1, 1, 4, 4)
        
        if init_wrist_mid_pose is None:
            init_wrist_mid_pose = (hand_data_array[0, 0, :3, 3] + hand_data_array[1, 0, :3, 3]) / 2
        hand_data_array[:, :, :3, 3] -= init_wrist_mid_pose
            
        visualizer.update_sphere_positions(hand_data_array)
        # visualizer.tick()
        time_lapse = time.time() - start_time
        if time_lapse > 0.02:
            continue
        time.sleep(max(0.02 - time_lapse, 0))

    visualizer.stop()    
    occulus.quit()
    