from paradex.io.camera_system.camera import Camera
from paradex.io.camera_system.pyspin import get_serial_list
import time

# Test image -> full -> image
serial_list = get_serial_list()
for serial_num in serial_list:
    camera = Camera("pyspin", serial_num)
    
    camera.start("image", False, "test1.png")
    camera.stop()
    
    camera.start("full", False, "test1.avi",fps=30)
    time.sleep(5)
    camera.stop()
    
    time.sleep(1)
    camera.start("image", False,"test2.png",fps=30)
    camera.stop()
    
    camera.start("video", False, "test2.avi",fps=5)
    time.sleep(5)
    
    camera.end()
    