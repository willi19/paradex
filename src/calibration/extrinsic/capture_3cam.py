import time
import os

from paradex.io.camera_system.remote_camera_controller import remote_camera_controller
from paradex.io.capture_pc.ssh import run_script
from paradex.calibration.utils import extrinsic_dir

TARGET_PCS = ["capture18", "capture14", "capture12"]

filename = time.strftime("%Y%m%d_%H%M%S", time.localtime())
os.makedirs(os.path.join(extrinsic_dir, filename), exist_ok=True)

# Clean stale remote client processes from previous interrupted runs.
run_script("pkill -f 'src/calibration/extrinsic/client_3cam.py' || true", pc_list=TARGET_PCS)
time.sleep(0.5)
run_script("python src/calibration/extrinsic/client_3cam.py", pc_list=TARGET_PCS)

rcc = remote_camera_controller("extrinsic_calibration", pc_list=TARGET_PCS)
rcc.start("stream", False, fps=30)

try:
    while True:
        time.sleep(1.0)
except KeyboardInterrupt:
    pass
finally:
    print("Stopping capture...")

    # Cleanup
    rcc.stop()
    rcc.end()

    print("Stream stopped.")
