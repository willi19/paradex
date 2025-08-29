import os
import time

from paradex.utils.file_io import shared_dir
from paradex.io.capture_pc.connect import run_script
from paradex.utils.env import get_pcinfo
from paradex.process.processor import ProcessorMain

pc_info = get_pcinfo()
pc_list = list(pc_info.keys())
    
# run_script(f"python src/debug/inference/sync/process/process_client.py", pc_list)

root_dir = os.path.join(shared_dir, "debug_", "inference")
process_list = []
for index in os.listdir(root_dir):
    process_list.append(os.path.join(root_dir, index))

p = ProcessorMain(process_list)
while not p.finish:
    time.sleep(1)