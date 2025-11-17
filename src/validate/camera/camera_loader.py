from paradex.io.camera_system.camera_loader import CameraLoader
import time

camera = CameraLoader()
camera.load_pyspin_camera()

for i in range(2):
    camera.start("full", False, "test_camloader",fps=30)
    time.sleep(1 * 2)
    camera.stop()

    camera.start("image", False, "test_camloader")
    camera.stop()
    
camera.end()
