import time
import numpy as np
from paradex.io.robot_controller.inspire_f1_controller_ethercat import InspireF1Controller


ctrl = InspireF1Controller("enp7s0f3", fps=200)

time.sleep(5)
ctrl.close()