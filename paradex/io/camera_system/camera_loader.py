from threading import Thread
import os
import time

from paradex.io.camera_system.camera import Camera
from paradex.utils.path import home_path, capture_path_list
from paradex.utils.system import get_camera_list

RETRY_COUNT = 5
RETRY_DELAY = 2  # seconds

class CameraLoader:
    def __init__(self, types=["pyspin"]):
        self.cameralist = []
        self.camera_names = []
        
        for cam_type in types:
            if cam_type == "pyspin":
                self.load_pyspin_camera()        
    
    def load_pyspin_camera(self, serial_list=None):
        from paradex.io.camera_system.pyspin import get_serial_list, autoforce_ip

        autoforce_ip()

        # env var 필터가 들어왔는지 추적 — 들어왔으면 아래 count-mismatch retry
        # 로직이 env var 필터를 덮어쓰지 않도록 skip.
        env_filtered = False
        if serial_list is None:
            # env var로 명시적 필터 가능 (e.g. PARADEX_CAMERA_SERIALS=25322639,25305465).
            # 일부 카메라가 물리적으로 없거나 다른 곳에서 점유 중이면 daemon이
            # Camera.__init__ 에서 hang하는 걸 방지.
            env_serials = os.environ.get("PARADEX_CAMERA_SERIALS", "").strip()
            if env_serials:
                serial_list = [s.strip() for s in env_serials.split(",") if s.strip()]
                env_filtered = True
                print(f"[CameraLoader] PARADEX_CAMERA_SERIALS 필터: {serial_list}")
            else:
                serial_list = get_serial_list()

        if not env_filtered and len(serial_list) != len(get_camera_list()):
            print(f"[Warning] Configured camera count ({len(get_camera_list())}) does not match detected camera count ({len(serial_list)}). Using detected cameras.")
            for _ in range(RETRY_COUNT):
                time.sleep(RETRY_DELAY)
                serial_list = get_serial_list()
                if len(serial_list) == len(get_camera_list()):
                    print("[Info] Camera count matched after retry.")
                    break
                
        self.cameralist = [Camera("pyspin", serial) for serial in serial_list]
        self.camera_names = self.camera_names + serial_list

    def _already_running(self, mode):
        if not self.cameralist:
            return False
        for camera in self.cameralist:
            is_running = (
                camera.event["start"].is_set()
                and camera.event["acquisition"].is_set()
                and getattr(camera, "mode", None) == mode
            )
            if not is_running:
                return False
        return True

    def _clear_errors(self):
        for camera in self.cameralist:
            if camera.event["error"].is_set():
                camera.error_reset()
    
    def start(self, mode, syncMode, save_path=None, fps=30):
        if mode == "stream" and self._already_running(mode):
            self._clear_errors()
            print("stream cameras already running; reusing existing acquisition.")
            return

        if mode == "image":
            save_paths = [os.path.join(home_path, save_path, "images") for _ in self.cameralist]
            print(save_paths)
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
            t = Thread(target=camera.start, args=(mode, syncMode, path, fps))
            threads.append(t)
            
        for t in threads:
            t.start()
        
        for t in threads:
            t.join()
        print("all cameras started.")

    def record_start(self, save_path, fps=30):
        """Arm .avi recording on all cameras WITHOUT restarting acquisition.

        Cameras must already be running (typically started in "stream" mode);
        the shared-memory stream is unaffected. Video dir layout matches the
        "video"/"full" start() path so rsync tooling keeps working.
        """
        save_paths = [
            os.path.join(capture_path_list[ind % len(capture_path_list)], save_path, "videos")
            for ind, _ in enumerate(self.cameralist)
        ]
        for path in save_paths:
            os.makedirs(path, exist_ok=True)
        print("record save paths:", save_paths)
        threads = [
            Thread(target=camera.record_start, args=(path, fps))
            for camera, path in zip(self.cameralist, save_paths)
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        print("recording armed on all cameras.")

    def record_stop(self):
        """Disarm .avi recording on all cameras (stream keeps running)."""
        threads = [Thread(target=camera.record_stop) for camera in self.cameralist]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        print("recording disarmed on all cameras.")

    def stop(self):
        threads = []
        for camera in self.cameralist:
            t = Thread(target=camera.stop)
            threads.append(t)
            
        for t in threads:
            t.start()
        
        for t in threads:
            t.join()
    
    def end(self):
        threads = []
        for camera in self.cameralist:
            t = Thread(target=camera.end)
            threads.append(t)
            
        for t in threads:
            t.start()
        
        for t in threads:
            t.join()

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
