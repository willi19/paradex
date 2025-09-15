import cv2
import numpy as np
import os
import json
from multiprocessing import Pool, shared_memory, Manager, Value
from paradex.utils.file_io import home_path
from paradex.video.raw_video import *

# Assert there are no other files under ~/captures1, ~/captures2.
# Every files will be deleted during processing

class RawVideoProcessor():
    def __init__(self, save_path, load_info=None, process_result=None, process_frame=None, preserve=True, overwrite=True):
        self.save_path = save_path
        self.manager = Manager()
        
        self.log = []
        self.video_path_list = get_videopath_list(f"{home_path}/captures1/{save_path}") + \
                                get_videopath_list(f"{home_path}/captures2/{save_path}")
        self.remove_invalid()
        
        # self.cur_state = self.manager.dict({vid_path:0 for vid_path in self.valid_video_path_list})
        self.total_frame = {vid_path:len(load_timestamp(vid_path)["frameID"]) for vid_path in self.valid_video_path_list}
        
        self.valid_video_path_list.sort()
        
        self.process_frame = process_frame # function that gets video_path & image for data processing and saving
        self.load_info = load_info
        self.process_result = process_result
               
        self.preserve = preserve
        self.overwrite = overwrite
        
        self.frame_counter = {vid_path:self.manager.Value('i', 0) for vid_path in self.valid_video_path_list}
        print(self.process_list)
        self.pool = Pool()
        self.process()
        
    def remove_invalid(self):
        self.valid_video_path_list = []
        for video_path in self.video_path_list:
            timestamp_path = get_timestamp_path(video_path)
            if not os.path.exists(timestamp_path):
                self.log.append(f"{video_path} stream terminated unexpectedly")
                os.remove(video_path)
                continue
            try:
                timestamp = json.load(open(timestamp_path))
            except:
                self.log.append(f"{video_path} : timestamp file not avaiable")
                os.remove(video_path)
                os.remove(timestamp_path)            
                continue
            
            if not check_valid(timestamp):
                self.log.append(f"{video_path} : sync or lan cable connection unstable")
                os.remove(video_path)
                os.remove(timestamp_path)            
                continue
            self.valid_video_path_list.append(video_path)
    
    
    def async_callback(self, result):
        self.log.append(result)
        
    def error_callback(self, e):
        self.log.append("ERROR in process:", e)
        
    def process(self):
        self.process_list = [self.pool.apply_async(fill_dropped_frames, args=(vid_path, self.load_info, self.process_frame, self.process_result, self.preserve, self.overwrite, self.frame_counter[vid_path]), callback=self.async_callback, error_callback=self.error_callback)
                        for vid_path in self.valid_video_path_list]    
       
    def finished(self):
        return all(r.ready() for r in self.process_list)

    