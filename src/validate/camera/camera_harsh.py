from paradex.io.camera_system.camera import Camera
from paradex.io.camera_system.pyspin import get_serial_list
import time

# Test image -> full -> image
serial_list = get_serial_list()
for serial_num in serial_list:
    camera = Camera("pyspin", serial_num)
    for i in range(1000):
        print(f"=== Test round {i} ===")
        camera.start("image", False, "test1.png")
        camera.stop()
        
    camera.end()
    