import os
import shutil
import time

from paradex.utils.file_io import shared_dir, download_dir
from paradex.process.util import merge, overlay, match_sync, get_object6D
from paradex.process.processor import ProcessorLocal

def process(root_dir, logger=[]):
    match_sync(root_dir, logger, overwrite=True)
    
    get_object6D(root_dir, logger, overwrite=True)
    
    overlay(root_dir, logger, overwrite=True)
    
    merge(root_dir, logger, overwrite=True)
    
    download_root_dir = root_dir.replace(shared_dir, download_dir)
    if os.path.exists(download_root_dir):
        shutil.rmtree(download_root_dir)
    
p = ProcessorLocal(process)
while not p.finished:
    time.sleep(2)