from paradex.inference.object_6d import get_current_object_6d
from paradex.io.capture_pc.camera_main import RemoteCameraController
from paradex.io.capture_pc.connect import git_pull, run_script
from paradex.utils.env import get_pcinfo
from paradex.io.signal_generator.UTGE900 import UTGE900
import time

pc_info = get_pcinfo()
pc_list = list(pc_info.keys())

signal_generator = UTGE900()

for _ in range(5):
    run_script(f"python src/capture/camera/video_client.py", pc_list)
    camera = RemoteCameraController("video", serial_list=None, sync=True)
    camera.start(f"erasethis/videos")
    signal_generator.on(1)
    time.sleep(1)
    camera.end()
    time.sleep(1)
    signal_generator.off(1)
    
    camera.quit()
    
    print("Run")
    pick_6D = get_current_object_6d("pringles")