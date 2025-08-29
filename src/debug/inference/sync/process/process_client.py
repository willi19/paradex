import os
import shutil
import time

from paradex.utils.file_io import shared_dir, download_dir
from paradex.process.util import merge, overlay, match_sync
from paradex.process.processor import ProcessorLocal

def process(root_dir, logger=None):
    # match_sync(root_dir, logger=None)
    # overlay(root_dir, logger=None)
    # merge(root_dir, logger=None)
    
    # download_root_dir = root_dir.replace(shared_dir, download_dir)
    # if os.path.exists(download_root_dir):
    #     shutil.rmtree(download_root_dir)
    print(root_dir)
    
    for i in range(1000):
        time.sleep(0.01)
        # print(root_dir, time.time(), i)
        logger.append({"root_dir":root_dir, "time":time.time(), "state":"processing", "msg":str(i), "type":"process_msg"})
        
p = ProcessorLocal(process)
while not p.finished:
    time.sleep(2)