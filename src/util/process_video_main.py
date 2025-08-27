import os
from multiprocessing import Pool
import time
import numpy as np

from paradex.io.capture_pc.raw_video_processor import ProgressMonitor
from paradex.io.capture_pc.connect import git_pull, run_script

run_script()

monitor = ProgressMonitor()
monitor.monitor()