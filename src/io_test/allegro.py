from dex_robot.io.robot_controller.allegro_controller import AllegroController
import numpy as np
import time

if __name__ == "__main__":
    allegro = AllegroController()
    allegro.set_homepose(np.zeros(16))
    allegro.home_robot()
    print("Home robot done.")
    start_time = time.time()
    while time.time() - start_time < 5:
        # allegro.move(np.zeros(16))
        time.sleep(0.01)
    allegro.quit()