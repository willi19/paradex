from pathlib import Path
import os, sys
sys.path.append(str(Path(__file__).parent.parent.parent))
from paradex.io.capture_pc.connect import run_script
from paradex.utils.env import get_pcinfo

pc_info = get_pcinfo()
pc_list = list(pc_info.keys())

run_script("pkill -f src/capture/camera/image_client.py", pc_list)
run_script("pkill -f src/calibration/handeyecalibration/client.py", pc_list)
run_script("pkill -f paradex/object_detection/client.py", pc_list)