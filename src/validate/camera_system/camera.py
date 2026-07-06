from paradex.io.camera_system.camera import Camera
from paradex.io.camera_system.pyspin import get_serial_list
import time

# Test image -> full -> image
serial_list = ["22645026"]# get_serial_list()
for serial_num in serial_list:
    camera = Camera("pyspin", serial_num)
    for i in range(2):
        print(f"=== Test round {i} ===")
        camera.start("image", False, f"test_{i}.png")
        camera.stop()
        print(f"Image capture {i} complete.")

        camera.start("acquire", False, fps=30)
        camera.set_sink(video=True, stream=True, save_path=f"test_{i}.avi")
        time.sleep(1)
        camera.stop()
        print(f"Full capture {i} complete.")
        
    camera.end()
    