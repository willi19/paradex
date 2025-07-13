import time

from paradex.io.capture_pc.camera_local import CameraCommandReceiver

camera_loader = CameraCommandReceiver()

while not camera_loader.exit:
    time.sleep(1)
    