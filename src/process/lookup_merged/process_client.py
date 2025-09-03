import os
import shutil
import time

from paradex.utils.file_io import shared_dir, download_dir
from paradex.process.util import merge, overlay, match_sync, get_object6D
from paradex.process.processor import ProcessorLocal
from paradex.process.lookup import generate_lookup_table, visualize_lookup_table

def process(root_dir, logger=[]):
    match_sync(root_dir, logger)
    
    get_object6D(root_dir, logger, overwrite=False)
    
    overlay(root_dir, logger, overwrite=False)
    
    merge(root_dir, logger, overwrite=False)
    
    generate_lookup_table(root_dir)
    visualize_lookup_table(root_dir, logger, overwrite=True)
    
    download_root_dir = root_dir.replace(shared_dir, download_dir)
    if os.path.exists(download_root_dir):
        shutil.rmtree(download_root_dir)
    
p = ProcessorLocal(process)
while not p.finished:
    time.sleep(2)