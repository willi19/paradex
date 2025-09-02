import os
from paradex.utils.file_io import shared_dir
from paradex.visualization_.viewer import ViserViewer
import time

if __name__ == "__main__":
    scene_path = os.path.join(shared_dir, "capture", "lookup", "pringles_light", "18")
    viewer = ViserViewer(scene_path, "pringles_light",hand_nm="allegro")
    
    while True:
        viewer.update()
        time.sleep(0.03)
