import os
import shutil
from util import get_process_list, get_argument, get_path

from paradex.utils.file_io import shared_dir, download_dir
from paradex.process.util import merge, overlay, match_sync, get_object6D
from paradex.process.lookup import generate_lookup_table, visualize_lookup_table

def process(root_dir, logger=[]):
    match_sync(root_dir, logger)
    print(logger)
    logger = []
    
    get_object6D(root_dir, logger, overwrite=False)
    print(logger)
    logger = []
    
    overlay(root_dir, logger, overwrite=False)
    print(logger)
    logger = []
    
    merge(root_dir, logger, overwrite=False)
    print(logger)
    logger = []
    
    # generate_lookup_table(root_dir)
    # visualize_lookup_table(root_dir, logger, overwrite=True)
    
    download_root_dir = root_dir.replace(shared_dir, download_dir)
    if os.path.exists(download_root_dir):
        shutil.rmtree(download_root_dir)
    

args = get_argument()
process_list = get_process_list(args)

for process_path in process_list:
    root_dir = os.path.join(shared_dir, process_path)
    process(root_dir, [])