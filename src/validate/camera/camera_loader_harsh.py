from paradex.io.camera_system.camera_loader import CameraLoader
import time

camera = CameraLoader()
camera.load_pyspin_camera()

for i in range(100):
    print(f"=== Test round {i} ===")
    camera.start("image", False, "test")
    camera.stop()

    camera.start("stream", False, "test",fps=30)
    time.sleep(1)
    camera.stop()

camera.end()
