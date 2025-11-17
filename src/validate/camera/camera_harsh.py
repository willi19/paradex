from paradex.io.camera_system.camera import Camera
from paradex.io.camera_system.pyspin import get_serial_list
import time

# Test image -> full -> image
serial_list = get_serial_list()
for serial_num in serial_list:
    camera = Camera("pyspin", serial_num)
    for i in range(5):
        print(f"=== Test round {i} ===")
        camera.start("image", False, f"test_{i}.png")
        camera.stop()
        print(f"Image capture {i} complete.")

        camera.start("full", False, f"test_{i}.avi",fps=30)
        time.sleep(1)
        camera.stop()
        print(f"Full capture {i} complete.")
        
    camera.end()
    