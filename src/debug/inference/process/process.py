import os

from paradex.utils.file_io import shared_dir
from paradex.process.util import merge, overlay, match_sync

def process(root_dir):
    match_sync(root_dir)
    overlay(root_dir)
    merge(root_dir)
    

root_dir = os.path.join(shared_dir, "debug_", "inference")
for index in os.listdir(root_dir):
    process(os.path.join(root_dir, index))