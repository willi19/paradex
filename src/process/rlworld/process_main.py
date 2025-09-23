import os
import time
import argparse

from paradex.utils.file_io import shared_dir
from paradex.io.capture_pc.connect import run_script
from paradex.utils.env import get_pcinfo
from paradex.process.processor import ProcessorMain
from util import get_process_list, get_argument, get_path

pc_info = get_pcinfo()
pc_list = list(pc_info.keys())

    

args = get_argument()
process_list = get_process_list(args)

run_script(f"python src/process/rlworld/process_client.py", pc_list)

p = ProcessorMain(process_list)
while not p.finish:
    time.sleep(1)