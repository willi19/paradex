from threading import Event, Thread
import time
import copy
import cv2

class Camera():
    def __init__(
        self,
        type,
        name
    ):
        self.event = {
            "start": Event(),
            "exit": Event(),
            
            "connection": Event(),
            "acquisition": Event(),
            "release": Event(),
            "stop": Event()
        }
        self.data_lock = Thread.Lock()
        
        self.type = type
        self.name = name
        
        self.capture_thread = Thread(target=self.run) 
        self.capture_thread.start()  
        
    def start(self, mode, syncMode, save_path=None):
        self.mode = mode
        self.syncMode = syncMode
        
        self.save_path = save_path
        
        self.data = {
            "frame": None,
            "frame_data": None
        }  
             
        self.event["start"].set()     
               
    def stop(self):
        self.event["start"].clear()
        
    def end(self):
        self.event["exit"].set()
    
    def get_data(self):
        with self.data_lock:
            return copy.deepcopy(self.data)
     
    def run(self):
        # Establish connection
        if self.type == "pyspin":
            from paradex.io.camera.pyspin import load_camera
        
        self.camera = load_camera(self.name)
        self.event["connection"].set()
        
        while not self.event["exit"].is_set(): # we should maintain the connection until exit
            if self.event["start"].is_set(): # Start data acquisition
                # reset camera
                self.camera.change_mode(self.mode, self.syncMode)
                
                if self.mode == "video":
                    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
                    video_writer = cv2.VideoWriter(self.save_path, fourcc, fps=30, frameSize=(1536, 2048) )
                        
                self.camera.start()
                self.event["acquisition"].set()
                
                while self.event["start"].is_set() and not self.event["exit"].is_set():
                    frame, frame_data = self.camera.get_image()
                    
                    if self.mode == "video":
                        video_writer.write(frame)
                    
                    elif self.mode == "image":
                        cv2.imwrite(self.save_path, frame)
                        break
                    
                    elif self.mode == "stream":
                        with self.data_lock:
                            self.data["frame"] = frame.copy()
                            self.data["frame_data"] = copy.deepcopy(frame_data)
                
                self.camera.stop()    
            time.sleep(0.001)
            self.event["acquisition"].clear()
            self.event["stop"].set()
            
            if self.mode == "video":
                video_writer.release()
        self.camera.release()
        self.event["release"].set()