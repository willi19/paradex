import os
from multiprocessing import Pool
import time
import numpy as np

from paradex.io.capture_pc.raw_video_processor import ProgressMonitor
from paradex.io.capture_pc.connect import git_pull, run_script
from paradex.utils.env import get_pcinfo

pc_info = get_pcinfo()
pc_list = list(pc_info.keys())
run_script("python src/util/raw_video/client.py",pc_list)

monitor = ProgressMonitor()
monitor.monitor()