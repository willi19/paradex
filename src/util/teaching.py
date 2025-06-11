import os
import sys
import time

sys.path.append(os.path.join(os.path.dirname(__file__), '../../..'))

from xarm.wrapper import XArmAPI

ip = "192.168.1.221"
arm = XArmAPI(ip, is_radian=True)
arm.motion_enable(enable=True)
arm.set_mode(0)
arm.set_state(state=0)


# # Turn on manual mode before recording
arm.set_mode(2)
arm.set_state(0)

# arm.start_record_trajectory()

# # Analog recording process, here with delay instead
time.sleep(5)

# arm.stop_record_trajectory()
# arm.save_record_trajectory('test.traj')

# Turn off manual mode after recording
arm.set_mode(0)
arm.set_state(0)

arm.motion_enable(enable=False)
arm.disconnect()