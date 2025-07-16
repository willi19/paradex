from paradex.video.raw_video_processor import RawVideoProcessor
from paradex.utils.file_io import home_path
import os
from multiprocessing import Pool
import time

def process_frame():
    pass

def process_result():
    pass

def load_info(video_path):
    pass

start_time = time.time()
rvp = RawVideoProcessor("lookup")
while not rvp.finished():
    time.sleep(0.1)
    
print("consumed time : ", time.time()-start_time)
for msg in rvp.log:
    print(msg)