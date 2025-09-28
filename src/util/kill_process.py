from paradex.io.capture_pc.connect import run_script
from paradex.utils.env import get_pcinfo

pc_info = get_pcinfo()
pc_list = list(pc_info.keys())

run_script("pkill -f src/capture/camera/video_client.py", pc_list)