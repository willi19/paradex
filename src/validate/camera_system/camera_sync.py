from paradex.io.camera_system.camera import Camera
from paradex.io.camera_system.pyspin import get_serial_list
import time

# Test image -> full -> image
serial_list = ["22645026"]# get_serial_list()
for serial_num in serial_list:
    camera = Camera("pyspin", serial_num)
    camera.start("acquire", True, fps=30)
    camera.set_sink(video=True, save_path="test.avi")
    time.sleep(5)
    camera.stop()
    camera.end()
    