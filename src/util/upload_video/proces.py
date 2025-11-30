import os
from multiprocessing import Pool
import time
import numpy as np

from paradex.io.capture_pc.raw_video_processor import ProgressMonitor
from paradex.io.capture_pc.ssh import run_script

run_script("python src/util/raw_video/client.py")

monitor = ProgressMonitor()
monitor.monitor()