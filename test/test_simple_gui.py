import os
import numpy as np
from paradex.io.robot_controller import get_arm, get_hand
from paradex.io.robot_controller.gui_controller_simple import QposTrajectoryGUIController

# arm+hand qpos trajectory 로드 (예: (T, 22))
traj_qpos = np.load("경로/your_traj_qpos.npy")   # shape: (T, 22)

# 홈 포즈: 예를 들어 trajectory 첫 스텝을 홈으로 쓰거나,
# 별도의 (22,) npy를 저장해 둔 걸 불러와도 됨.
home_qpos = traj_qpos[0]  # 또는 np.load("home_qpos.npy")

arm = get_arm("xarm")
hand = get_hand("allegro")

gui = QposTrajectoryGUIController(
    robot_controller=arm,
    hand_controller=hand,
    home_qpos=home_qpos,
    traj_qpos=traj_qpos,
    dt=0.01,  # 필요하면 조절
)
gui.run()