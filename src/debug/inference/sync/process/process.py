import os
import shutil

from paradex.utils.file_io import shared_dir, download_dir
from paradex.process.util import merge, overlay, match_sync

def process(root_dir, logger=None):
    match_sync(root_dir, logger=None)
    overlay(root_dir, logger=None)
    merge(root_dir, logger=None)
    
    download_root_dir = root_dir.replace(shared_dir, download_dir)
    if os.path.exists(download_root_dir):
        shutil.rmtree(download_root_dir)
    

root_dir = os.path.join(shared_dir, "debug_", "inference")
for index in os.listdir(root_dir)[0]:
    process(os.path.join(root_dir, index), logger=None)