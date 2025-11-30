from threading import Event, Thread
import time
import numpy as np
import os
import traceback

class TimestampMonitor():
    def __init__(self, cam_type, name):
        self.event = {
            "start": Event(),
            "exit": Event(),
            "error": Event(),
            "error_reset": Event(),
            
            "connection": Event(),
            "acquisition": Event(),
            "release": Event(),
            "stop": Event()
        }
        
        self.event["error_reset"].set()
        
        self.type = cam_type
        self.name = name

        self.last_error = None
        self.last_traceback = None
        
        self.capture_thread = Thread(target=self.run) 
        self.capture_thread.start()  
        
        self.event["connection"].wait()

    def start(self, save_path=None):
        if self.event["start"].is_set():
            self.event["error"].set()
            self.event["error_reset"].clear()
            
            self.last_error = "Acquisition is already running."
            self.last_traceback = ""
            
        if self.event["error"].is_set():
            print(f"[WARNING] Camera {self.name} is in ERROR state. Resetting error state.")
            return
        
        self.last_frame_id = 0 # Start with 1
        self.last_timestamp = 0.0
        
        self.save_path = None
        
        if save_path is not None:
            os.makedirs(save_path, exist_ok=True)
            self.data = {
                "frame_id": [],
                "timestamp": []
            }
            self.save_path = save_path
            
        self.event["stop"].clear()
        self.event["start"].set()  
        
        self.event["acquisition"].wait()   
        
    def error_reset(self):
        self.last_error = None
        self.last_traceback = None
        
        self.event["error"].clear()
        self.event["error_reset"].set()
        
    def is_error(self):
        return self.event["error"].is_set()

    def stop(self):
        self.event["start"].clear()
        
        if self.event["error"].is_set():
            self.error_reset()
            
        self.event["stop"].wait()
                   
    def end(self):
        if self.event["start"].is_set():
            self.stop()
        
        self.event["exit"].set()
        self.capture_thread.join()
    
    def continuous_acquire(self):
        save_timestamps = (self.save_path is not None)
        
        try:
            self.camera.start()
        
        except Exception as e:
            self.event["error"].set()
            self.event["error_reset"].clear() 
               
            self.last_error = str(e)
            self.last_traceback = traceback.format_exc()
            
            print(f"[ERROR] Camera {self.name} exception occurred:")
            print(f"Exception Type: {type(e).__name__}")
            print(f"Exception Message: {str(e)}")
            print(self.last_traceback)

            self.event["acquisition"].set()  # To avoid deadlock

            self.event["error_reset"].wait()
            
            self.event["acquisition"].clear()
            self.event["stop"].set()
            
            print(f"[INFO] Camera {self.name} acquisition aborted due to error during start.")
            return

        self.event["acquisition"].set()
                
        while self.event["start"].is_set() and not self.event["exit"].is_set():
            try:
                frame_data = self.camera.get_timestamp()
                self.last_frame_id = frame_data["frameID"]
                self.last_timestamp = frame_data["pc_time"]
                
                if save_timestamps:
                    self.data["timestamp"].append(self.last_timestamp)
                    self.data["frame_id"].append(self.last_frame_id)

            except Exception as e:
                self.event["error"].set()
                self.event["error_reset"].clear() 
                   
                self.last_error = str(e)
                self.last_traceback = traceback.format_exc()
                
                print(f"[ERROR] Camera {self.name} exception occurred during acquisition:")
                print(f"Exception Type: {type(e).__name__}")
                print(f"Exception Message: {str(e)}")
                print(self.last_traceback)

                self.event["error_reset"].wait()
                break

        self.camera.stop()
        self.event["acquisition"].clear()
        
        if save_timestamps:
            for name, value in self.data.items():                     
                np.save(os.path.join(self.save_path, f"{name}.npy"), np.array(value))
            
            self.data = {}
                    
        self.event["stop"].set()
    
    def connect_camera(self):
        # Establish connection
        if self.type == "pyspin":
            from paradex.io.camera_system.pyspin import load_timestamp_monitor
        else:
            raise NotImplementedError(f"Camera type {self.type} is not implemented.")
        
        self.camera = load_timestamp_monitor(self.name)
        self.event["connection"].set()
    
    def release(self):
        self.camera.release()
        self.event["release"].set()

    def get_data(self):
        return {
            'frame_id': self.last_frame_id,
            'time': self.last_timestamp
        }  

    def wait_signal_inactive(self, fps):
        self.camera.start()
        while True:
            frame_data = self.camera.get_timestamp( timeout_ms= int(1000.0 / fps * 1.5) )
            if frame_data is None:
                self.camera.stop()
                return
                
    def run(self):
        self.connect_camera()
        
        while not self.event["exit"].is_set(): # we should maintain the connection until exit
            if self.event["start"].is_set(): # Start data acquisition
                self.continuous_acquire()
                
            time.sleep(0.001)
                    
        self.release()