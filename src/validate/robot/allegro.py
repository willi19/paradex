from paradex.io.robot_controller import get_hand
import numpy as np
import time

hand = get_hand("allegro")
start_time = time.time()

qpos = np.zeros(16)

while time.time() - start_time < 10:
    if time.time() - start_time > 5:
        v = 0.5 - (time.time() - start_time - 5) / 10 * 1
    else:
        v = (time.time() - start_time) / 10 * 1 + 0.5
    qpos[4] = v
    hand.move(qpos)

time.sleep(2)   

# 0 thumb root
# 1 thumb mid
# 2 thumb asdf
# 3 
# 4 index root