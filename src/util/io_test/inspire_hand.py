from paradex.io.robot_controller.inspire_controller import InspireController
import numpy as np
import time

action = np.zeros(6) + 800

inspire = InspireController()

start_time = time.time()
while time.time() - start_time < 10:
    action[:] = 800 + 200 / 10 * (time.time()-start_time)
    inspire.set_target_action(action.copy())
    time.sleep(0.1)