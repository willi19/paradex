from dex_robot.io.robot_controller.inspire_controller import InspireController
import numpy as np
import time

if __name__ == "__main__":
    inspire = InspireController()
    inspire.set_homepose(np.zeros(6)+500)
    inspire.home_robot()
    print("Home robot done.")
    start_time = time.time()
    while time.time() - start_time < 5:
        inspire.set_robot_servo(np.zeros(6)+100 * (time.time() - start_time)+500)
        time.sleep(0.01)
    inspire.quit()