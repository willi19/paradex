import time

from paradex.io.camera_system.signal_generator import UTGE900
from paradex.utils.system import network_info

param = network_info.get("signal_generator")["param"]
signal_gen = UTGE900(**param)
signal_gen.start(fps=30)
time.sleep(5)
signal_gen.stop()
signal_gen.end()