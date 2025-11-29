import argparse
import os
import numpy as np
import time

from paradex.calibration.utils import save_current_camparam
from paradex.calibration.utils import get_handeye_calib_path
from paradex.io.capture_pc.ssh import run_script
from paradex.io.robot_controller.xarm_controller import XArmController
from paradex.utils.system import network_info

if __name__ == "__main__":
    controller = XArmController(**network_info["xarm"]["param"])
    controller.connect_event.wait()
    
    # 현재 위치
    start_data = controller.get_data()
    # 목표: 왼쪽으로 100mm
    speed = 700.0  # mm/s
    direction = np.array([0.0, 1.0, 0.0])  
    init_time = start_time = time.time()
    
    # risky_target = start_data["position"].copy()
    # risky_target[:3, 3] += direction * 0.5
    # print("moving to risky target:", risky_target)
    # controller.move(risky_target)
    # time.sleep(0.05)
    while time.time() - init_time < 5.0:
        elapsed = time.time() - start_time
        start_time = time.time()
        
        cur_y = controller.get_data()["position"]
        cur_y[:3, 3] += direction * speed * elapsed / 1000
        controller.move(cur_y)
        
        cur_y = controller.get_data()["position"][1, 3] 
        if cur_y > 0.5:
            direction = np.array([0.0, -1.0, 0.0])  # 오른쪽으로 변경
        elif cur_y < -0.5:
            direction = np.array([0.0, 1.0, 0.0])   # 왼쪽으로 변경
        time.sleep(0.01)
    
    controller.end(False)
    