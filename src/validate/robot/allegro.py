from paradex.io.robot_controller import get_hand
import numpy as np
import time

hand = get_hand("allegro")
start_time = time.time()

qpos = np.zeros(16)
hand.start("allegro_order_debug")
for joint_idx in range(16):
    print(f"Moving joint {joint_idx}")
    start_time = time.time()
    
    while time.time() - start_time < 2:  # 각 조인트당 2초
        t = time.time() - start_time
        if t > 1:
            v = 1.0 - (t - 1) * 1.0  # 1초 후 돌아오기
        else:
            v = t * 1.0  # 1초 동안 펴기
        
        qpos[joint_idx] = v
        hand.move(qpos)
    
    qpos[joint_idx] = 0  # 원위치
    hand.move(qpos)
hand.end()
time.sleep(2)

# 0 thumb root
# 1 thumb mid
# 2 thumb asdf
# 3 
# 4 index root