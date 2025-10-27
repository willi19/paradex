from paradex.io.camera_system.camera_loader import CameraLoader
import time

camera = CameraLoader()
camera.load_pyspin_camera()

camera.start("image", False, "test")
camera.stop()

camera.start("full", False, "test",fps=30)
time.sleep(5)
camera.stop()
camera.end()
