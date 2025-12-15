import tkinter as tk
import time
import numpy as np
from threading import Thread
from scipy.spatial.transform import Rotation

from paradex.io.robot_controller import get_arm, get_hand
from paradex.io.robot_controller.gui_controller import RobotGUIController

arm = get_arm("xarm")
rgc = RobotGUIController(arm)

rgc.run()
print("GUI 종료됨")