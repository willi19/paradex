from paradex.io.robot_controller.openarm_state_receiver import OpenArmStateReceiver
import time

arm = OpenArmStateReceiver()
arm.start("/home/temp_id/shared_data/test_openarm")
time.sleep(10)
arm.stop()
arm.end()