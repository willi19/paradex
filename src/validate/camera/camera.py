from paradex.io.camera_system.camera import Camera
from paradex.io.camera_system.pyspin import get_serial_list
import time

# Test image -> full -> image
serial_list = get_serial_list()
for serial_num in serial_list:
    camera = Camera("pyspin", serial_num)
    
    camera.start("full", True, f"test1_{serial_num}.avi",fps=30)
    time.sleep(2)
    camera.stop()
    camera.end()
    