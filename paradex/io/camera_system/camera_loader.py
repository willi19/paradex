from threading import Thread
import os
import time

from paradex.io.camera_system.camera import Camera
from paradex.utils.path import home_path, capture_path_list
from paradex.utils.system import get_camera_list, get_camera_config

RETRY_COUNT = 5
RETRY_DELAY = 2  # seconds

# Fallback per-camera params when a serial is absent from camera.json.
DEFAULT_GAIN = 3.0
DEFAULT_EXPOSURE = 2500.0

class CameraLoader:
    def __init__(self, types=["pyspin"]):
        self.cameralist = []
        self.camera_names = []
        # Per-serial gain/exposure baseline (system/current/camera.json).
        self.cam_config = get_camera_config()

        for cam_type in types:
            if cam_type == "pyspin":
                self.load_pyspin_camera()
    
    def load_pyspin_camera(self, serial_list=None):
        from paradex.io.camera_system.pyspin import get_serial_list, autoforce_ip
        
        expected = len(get_camera_list())
        autoforce_ip()

        if serial_list is None:
            serial_list = get_serial_list()

        # After a power cycle GigE cameras boot slowly and come up on the wrong IP.
        # autoforce_ip() only forces cameras that are already enumerated, so cameras
        # that were not ready at the first call would never be forced. Re-run
        # autoforce_ip() on every retry until the expected count appears.
        if len(serial_list) != expected:
            print(f"[Warning] Configured camera count ({expected}) does not match "
                  f"detected camera count ({len(serial_list)}). Retrying (re-forcing IP)...")
            for _ in range(RETRY_COUNT):
                time.sleep(RETRY_DELAY)
                autoforce_ip()
                serial_list = get_serial_list()
                if len(serial_list) == expected:
                    print("[Info] Camera count matched after retry.")
                    break
            else:
                print(f"[Warning] Still {len(serial_list)}/{expected} cameras after "
                      f"{RETRY_COUNT} retries; proceeding with detected cameras.")
                
        self.cameralist = [Camera("pyspin", serial) for serial in serial_list]
        self.camera_names = self.camera_names + serial_list
    
    def start(self, mode, syncMode, save_path=None, fps=30, exposure_time=None, gain=None):
        if mode == "image":
            save_paths = [os.path.join(home_path, save_path, "images") for _ in self.cameralist]
            print("image save paths:", save_paths)
            for path in save_paths:
                os.makedirs(path, exist_ok=True)

        elif mode in ["video", "full"]:
            save_paths = [os.path.join(capture_path_list[ind % len(capture_path_list)], save_path, "videos") for ind, _ in enumerate(self.cameralist)]
            for path in save_paths:
                os.makedirs(path, exist_ok=True)
            print("video save paths:", save_paths)
            
        else:
            save_paths = [None for _ in self.cameralist]
        print("starting cameras... cameras:", self.camera_names)
        threads = []
        for camera, path in zip(self.cameralist, save_paths):
            # Resolve deterministically: explicit arg > camera.json baseline > default.
            # None means "use the camera.json baseline", never "keep whatever was last
            # set" — so a prior override (e.g. an exposure sweep) can't silently leak
            # into the next capture.
            cfg = self.cam_config.get(camera.name, {})
            cam_exposure = exposure_time if exposure_time is not None else cfg.get("exposure", DEFAULT_EXPOSURE)
            cam_gain = gain if gain is not None else cfg.get("gain", DEFAULT_GAIN)
            t = Thread(target=camera.start, args=(mode, syncMode, path, fps, cam_exposure, cam_gain))
            threads.append(t)
            
        for t in threads:
            t.start()
        
        for t in threads:
            t.join()
        print("all cameras started.")
    
    def _broadcast(self, method_name):
        """Call `method_name` on every camera in parallel and wait for all to finish."""
        threads = [Thread(target=getattr(camera, method_name)) for camera in self.cameralist]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

    def stop(self):
        self._broadcast("stop")

    def end(self):
        self._broadcast("end")

    def get_status_list(self):
        status_list = []
        for camera in self.cameralist:
            status_list.append(camera.get_status())
        return status_list
    
    def get_all_errors(self):
        """모든 카메라의 에러 정보 반환"""
        errors = {}
        for camera in self.cameralist:
            has_error, (error_msg, traceback_msg) = camera.get_error()
            if has_error:
                errors[camera.name] = (error_msg, traceback_msg)
        return errors