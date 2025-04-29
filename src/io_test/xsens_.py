from dex_robot.io.xsens.receiver import XSensReceiver
import time 
from multiprocessing import shared_memory, Lock, Value, Event
from dex_robot.io.xsens.visualizer import XSensVisualizer
import threading
import numpy as np

if __name__ == "__main__":
    host = "192.168.0.2"
    port = 9763

    xsens_updater = XSensReceiver()
    xsens_updater.init_server(host, port)
    
    start_time = time.time()

    # visualizer = XSensVisualizer("render",2048, 1536,[],None)
    # threading.Thread(target=visualizer.run, daemon=True).start()

    while time.time() - start_time < 10:
        data = xsens_updater.get_data()

        pose_data = data["hand_pose"]        
        state = data["state"]
        # print(pose_data[0], pose_data[6][:3,3])
        # for i in range(1,20):
        #     pose_data[i] = np.linalg.inv(pose_data[0]) @ pose_data[i]
        # pose_data[0] = np.eye(4)
        
        print(state)

        # visualizer.update_sphere_positions(pose_data)
        time.sleep(0.01)  # Update at 20 FPS
    
    xsens_updater.quit()
    # visualizer.stop()
    print("Program terminated.")
    exit(0)