import os
from paradex.utils.file_io import shared_dir
from paradex.visualization_.viewer import ViserViewer
import time

if __name__ == "__main__":
    scene_path = os.path.join(shared_dir, "capture_", "lookup", "pringles", "stand_free", "0")
    viewer = ViserViewer(scene_path, "pringles")
    
    while True:
        viewer.update()
        time.sleep(0.03)
