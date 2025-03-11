import threading
import PySpin
import json
import time
from multiprocessing import shared_memory, Lock, Value, Event
import signal
import sys
from ..camera import camera
from ..utils.image_util import spin2cv
import numpy as np
import os

homedir = os.path.expanduser("~")

class CameraManager:
    def __init__(self, num_cameras, duration=0, is_streaming=False, name=None, syncMode=True, shared_memories={}, update_flags={}):
        self.num_cameras = num_cameras
        self.duration = duration

        self.is_streaming = is_streaming
        self.name = name

        self.stop_event = Event()
        self.shared_memories = shared_memories
        self.update_flags = update_flags
        self.locks = {}

        self.capture_threads = []
        self.frame_cnt = 0
        self.syncMode = syncMode


    def configure_camera(self, cam):
        nodemap = cam.GetNodeMap()
        node_acquisition_mode = PySpin.CEnumerationPtr(nodemap.GetNode("AcquisitionMode"))
        node_acquisition_mode_continuous = node_acquisition_mode.GetEntryByName("Continuous")
        node_acquisition_mode.SetIntValue(node_acquisition_mode_continuous.GetValue())
        

    def create_shared_memory(self, camera_index, shape, dtype):
        """
        Creates shared memory and lock for a camera.
        """
        shm_name = f"camera_{camera_index}_shm"
        shm = shared_memory.SharedMemory(create=True, name=shm_name, size=np.prod(shape) * np.dtype(dtype).itemsize)
        self.shared_memories[camera_index] = {
            "name": shm_name,
            "shm": shm,
            "array": np.ndarray(shape, dtype=dtype, buffer=shm.buf),
            "lock": Lock(),
        }
        self.update_flags[camera_index] = Value('i', 0)  # 0: not updated, 1: updated

        print(f"Shared memory created for camera {camera_index}.")

    def capture_video(self, camera_index):
        system = PySpin.System.GetInstance()
        cam_list = system.GetCameras()
        cnt = 0

        if cam_list.GetSize() <= camera_index:
            print(f"Camera index {camera_index} is out of range.")
            cam_list.Clear()
            system.ReleaseInstance()
            return

        camPtr = cam_list.GetByIndex(camera_index)
        lens_info = json.load(open("config/lens.json", "r"))
        cam_info = json.load(open("config/camera.json", "r"))

        if self.is_streaming:
            shm_info = self.shared_memories[camera_index]
            update_flag = self.update_flags[camera_index]
        
        save_dir = f"{homedir}/captures{camera_index // 2 + 1}/{self.name}"
        os.makedirs(save_dir, exist_ok=True)
        
        cam = camera.Camera(camPtr, lens_info, cam_info, save_dir, syncMode=self.syncMode)

        if not self.is_streaming:
            cam.set_record()
        while not self.stop_event.is_set():
            frame, ret = cam.get_capture()
            # cnt += 1
            # print(f"Camera {camera_index} captured frame {cnt}")
            if ret and self.is_streaming:
                img = spin2cv(frame, 1536, 2048)
                with shm_info["lock"]:
                    np.copyto(shm_info["array"], img)
                    update_flag.value = 1  # Mark as updated
        
        if not self.is_streaming:
            cam.set_record()
        cam.stop_camera()
        del camPtr
        cam_list.Clear()
        system.ReleaseInstance()

    def signal_handler(self):
        print("\nSIGINT received. Terminating all processes and threads...")
        self.stop_event.set()

        for shm_info in self.shared_memories.values():
            shm_info["shm"].close()
            shm_info["shm"].unlink()

        sys.exit(0)
    

    def input_listener(self):
        """
        Listens for user input and stops capture when 'q' is pressed.
        """
        while not self.stop_event.is_set():
            user_input = input().strip().lower()
            if user_input == "q":
                print("\n'q' received. Stopping capture...")
                self.stop_event.set()
                break

    def start(self):
        frame_shape = (1536, 2048, 3)  # Example shape for each frame (RGB)
        frame_dtype = np.uint8
        if self.is_streaming:
            for i in range(self.num_cameras):
                self.create_shared_memory(i, frame_shape, frame_dtype)

        self.capture_threads = [
            threading.Thread(target=self.capture_video, args=(i,))
            for i in range(self.num_cameras)
        ]

        if not self.is_streaming:
            self.input_thread = threading.Thread(target=self.input_listener, daemon=True)
            self.input_thread.start()

        signal.signal(signal.SIGINT, lambda sig, frame: self.signal_handler())

        for p in self.capture_threads:
            p.start()

        # Wait for all threads to finish
        for p in self.capture_threads:
            p.join()

        print("All capture threads have stopped.")

        
if __name__ == "__main__":
    manager = CameraManager(num_cameras=4, duration=180, save_dir="/home/capture16/captures1", is_streaming=False)
    manager.start()
