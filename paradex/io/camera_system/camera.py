from threading import Event, Thread
import time
import cv2
from multiprocessing import shared_memory
import numpy as np
import os
import traceback

class Camera():
    def __init__(self, cam_type, name, frame_shape=(1536, 2048, 3)):
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
        
        self.frame_shape = frame_shape  
        
        self.last_error = None
        self.last_traceback = None
        
        self.load_shared_memory()
        
        self.capture_thread = Thread(target=self.run) 
        self.capture_thread.start()  
        
        self.event["connection"].wait()
    
    def _cleanup_existing_shm(self):
        shm_names = [
            self.name + "_image_a",
            self.name + "_image_b",
            self.name + "_fid_a",
            self.name + "_fid_b",
            self.name + "_flag"
        ]
        
        for shm_name in shm_names:
            try:
                shm = shared_memory.SharedMemory(name=shm_name)
                shm.close()
                shm.unlink()
            except FileNotFoundError:
                pass  # 없으면 패스
    
    def load_shared_memory(self):
        frame_size = np.prod(self.frame_shape)
        self._cleanup_existing_shm()
        # Buffer 2개
        self.image_shm_a = shared_memory.SharedMemory(
            create=True, 
            size=frame_size, 
            name=self.name + "_image_a"
        )
        
        self.image_shm_b = shared_memory.SharedMemory(
            create=True, 
            size=frame_size, 
            name=self.name + "_image_b"
        )
        
        # Frame ID 2개
        self.fid_shm_a = shared_memory.SharedMemory(
            create=True, 
            size=8, 
            name=self.name + "_fid_a"
        )
        self.fid_shm_b = shared_memory.SharedMemory(
            create=True, 
            size=8, 
            name=self.name + "_fid_b"
        )
        
        # Write buffer flag (0 or 1)
        self.write_flag_shm = shared_memory.SharedMemory(
            create=True, 
            size=1, 
            name=self.name + "_flag"
        )
        
        # Arrays
        self.image_array_a = np.ndarray(
            self.frame_shape, dtype=np.uint8, buffer=self.image_shm_a.buf
        )
        self.image_array_b = np.ndarray(
            self.frame_shape, dtype=np.uint8, buffer=self.image_shm_b.buf
        )
        self.fid_array_a = np.ndarray(
            (1,), dtype=np.int64, buffer=self.fid_shm_a.buf
        )
        self.fid_array_b = np.ndarray(
            (1,), dtype=np.int64, buffer=self.fid_shm_b.buf
        )
        self.write_flag = np.ndarray(
            (1,), dtype=np.uint8, buffer=self.write_flag_shm.buf
        )
        self.write_flag[0] = 0
    
    def release_shared_memory(self):
        self.image_array_a = None
        self.image_array_b = None
        self.fid_array_a = None
        self.fid_array_b = None
        self.write_flag   = None

        self.image_shm_a.close()
        self.image_shm_a.unlink()
        
        self.image_shm_b.close()
        self.image_shm_b.unlink()
        
        self.fid_shm_a.close()
        self.fid_shm_a.unlink()
        
        self.fid_shm_b.close()
        self.fid_shm_b.unlink()
        
        self.write_flag_shm.close()
        self.write_flag_shm.unlink()
        
    def clear_shared_memory(self):
        self.fid_array_a[0] = 0
        self.fid_array_b[0] = 0
        self.write_flag[0] = 0
        
        self.image_array_a.fill(0)
        self.image_array_b.fill(0)
        
    def start(self, mode, syncMode, save_path=None, fps=30):
        if fps < 0 and mode in ["video", "full"] and syncMode is False:
            raise ValueError("FPS must be specified for video recording.")
        
        if mode in ["video", "full", "image"] and save_path is None:
            raise ValueError("Save path must be specified for video or image saving.")
        if self.event["start"].is_set():
            raise RuntimeError("Acquisition is already running.")

        if self.event["error"].is_set():
            print(f"[WARNING] Camera {self.name} is in ERROR state. Resetting error state.")
            return
        
        self.mode = mode
        self.syncMode = syncMode
        self.fps = fps
        self.last_frame_id = 0
        
        if save_path is not None:
            _, ext = os.path.splitext(save_path)
            if not ext: 
                default_ext = ".avi" if mode in ["video", "full"] else ".png"
                self.save_path = os.path.join(save_path, f"{self.name}{default_ext}")
            else:  
                self.save_path = save_path
            
            save_dir = os.path.dirname(self.save_path)
            if save_dir:
                os.makedirs(save_dir, exist_ok=True)
            
        self.event["stop"].clear()
        self.event["start"].set()  
        
        self.event["acquisition"].wait()   
    
    def error_reset(self):
        self.last_error = None
        self.last_traceback = None
        
        self.event["error"].clear()
        self.event["error_reset"].set()
               
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
        save_video = (self.mode in ["video", "full"] and self.save_path is not None)
        stream = (self.mode in ["stream", "full"])
        blank_frame = np.zeros(self.frame_shape, dtype=np.uint8)
        
        if save_video:
            fourcc = cv2.VideoWriter_fourcc(*'MJPG')
            video_writer = cv2.VideoWriter(self.save_path, fourcc, fps=self.fps, frameSize=(self.frame_shape[1], self.frame_shape[0]))        
        
        try:
            self.camera.start("continuous", self.syncMode, self.fps)
        
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
            
            if save_video:
                video_writer.release()
            if stream:
                self.clear_shared_memory()
                
            return 
            
                
        self.event["acquisition"].set()
                
        while self.event["start"].is_set() and not self.event["exit"].is_set():
            frame, frame_data = self.camera.get_image()
            if frame is None:
                continue
            current_frame_id = frame_data["frameID"]
            if save_video:
                for _ in range(current_frame_id - self.last_frame_id-1):
                    print(f"frame drop {self.name}: missing frame id", current_frame_id-self.last_frame_id-1)
                    video_writer.write(blank_frame)
                video_writer.write(frame)
            
            if stream:
                # Write to shared memory
                if self.write_flag[0] == 0:
                    np.copyto(self.image_array_a, frame)
                    self.fid_array_a[0] = frame_data["frameID"]
                    self.write_flag[0] = 1
                else:
                    np.copyto(self.image_array_b, frame)
                    self.fid_array_b[0] = frame_data["frameID"]
                    self.write_flag[0] = 0

            self.last_frame_id = current_frame_id

        self.camera.stop()
        self.event["acquisition"].clear()

        if stream:
            self.clear_shared_memory()
        
        if save_video:
            video_writer.release()
        
        self.event["stop"].set()
    
    def single_acquire(self):
        self.camera.start("single", self.syncMode)
        self.event["acquisition"].set()
        
        frame, _ = self.camera.get_image()
        cv2.imwrite(self.save_path, frame)
        
        self.event["acquisition"].clear()
        self.event["start"].clear()
        self.camera.stop()
        self.event["stop"].set()       
    
    def connect_camera(self):
        # Establish connection
        if self.type == "pyspin":
            from paradex.io.camera_system.pyspin import load_camera
        else:
            raise NotImplementedError(f"Camera type {self.type} is not implemented.")
        
        self.camera = load_camera(self.name)
        self.event["connection"].set()
    
    def release(self):
        self.camera.release()
        self.release_shared_memory()
        self.event["release"].set()
    
    def get_state(self):
        if self.event["error"].is_set():
            return f"ERROR: {self.last_error} {self.last_traceback}"
        elif self.event["exit"].is_set():
            return "STOPPED"
        elif self.event["start"].is_set():
            if self.event["acquisition"].is_set():
                return "CAPTURING"
            else:
                return "STARTING"
        elif self.event["connection"].is_set():
            return "READY"
        else:
            return "CONNECTING"

    def get_frame_id(self):
        if self.write_flag[0] == 0:
            return int(self.fid_array_b[0])
        else:
            return int(self.fid_array_a[0])

    def get_status(self):
        return {
            'state': self.get_state(),
            'frame_id': self.get_frame_id(),
            'name': self.name,
            'mode': getattr(self, 'mode', None),
            'fps': getattr(self, 'fps', None),
            'syncMode': getattr(self, 'syncMode', None),
            'save_path': getattr(self, 'save_path', None),
            'time': time.time()
        }  

    def run(self):
        self.connect_camera()
        
        while not self.event["exit"].is_set(): # we should maintain the connection until exit
            if self.event["start"].is_set(): # Start data acquisition
                if self.mode in ["full", "video", "stream"]:
                    self.continuous_acquire()
                else:
                    self.single_acquire()
            
            time.sleep(0.001)
                    
        self.release()