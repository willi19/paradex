from threading import Event, Thread
import time
import cv2
from multiprocessing import shared_memory
import numpy as np
import os
import traceback

class Camera():
    # JPEG shm geometry: keep MAX large enough for 2048x1536 q=85 worst case
    # (~1 MB). Layout per buffer: [4B little-endian length | JPEG bytes ...].
    JPEG_MAX_BYTES = 1_000_000
    JPEG_QUALITY = 85

    def __init__(self, cam_type, name, frame_shape=(1536, 2048, 3)):
        self.event = {
            "start": Event(),
            "exit": Event(),
            "error": Event(),
            "error_reset": Event(),

            "connection": Event(),
            "acquisition": Event(),
            "release": Event(),
            "stop": Event(),
            # Set whenever NO recording file is open (idle or after the writer
            # has been fully released/flushed). record_stop() waits on this so
            # callers can safely rsync the .avi right after.
            "record_closed": Event(),
            # Runtime recording toggle, independent of the acquisition
            # lifecycle. When set during continuous_acquire(), a VideoWriter is
            # opened mid-stream; when cleared it is closed. This lets a caller
            # record .avi on/off WITHOUT stopping the shared-memory stream
            # (no stop()/start(), no shm unlink).
            "record": Event()
        }

        self.event["error_reset"].set()
        self.event["record_closed"].set()  # no writer open yet

        self.type = cam_type
        self.name = name

        self.frame_shape = frame_shape

        self.last_frame_id = 0
        # Per-episode recording target (set by record_start / start()).
        self.record_save_path = None
        self.record_fps = 30

        self.last_error = None
        self.last_traceback = None
        
        self.load_shared_memory()
        print(f"[INFO] Camera {self.name} shared memory loaded.")
        self.capture_thread = Thread(target=self.run) 
        self.capture_thread.start()  
        
        self.event["connection"].wait()
    
    def _cleanup_existing_shm(self):
        shm_names = [
            self.name + "_image_a",
            self.name + "_image_b",
            self.name + "_fid_a",
            self.name + "_fid_b",
            self.name + "_flag",
            # JPEG-encoded mirror (added for low-bandwidth consumers)
            self.name + "_jpeg_a",
            self.name + "_jpeg_b",
            self.name + "_jlen_a",
            self.name + "_jlen_b",
        ]
        print("unlink existing shm if any:", shm_names)
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

        # JPEG mirror: separate double-buffer alongside raw, sharing the same
        # write_flag so a reader gets matched raw/JPEG for any given frame.
        self.jpeg_shm_a = shared_memory.SharedMemory(
            create=True, size=self.JPEG_MAX_BYTES, name=self.name + "_jpeg_a",
        )
        self.jpeg_shm_b = shared_memory.SharedMemory(
            create=True, size=self.JPEG_MAX_BYTES, name=self.name + "_jpeg_b",
        )
        self.jlen_shm_a = shared_memory.SharedMemory(
            create=True, size=4, name=self.name + "_jlen_a",
        )
        self.jlen_shm_b = shared_memory.SharedMemory(
            create=True, size=4, name=self.name + "_jlen_b",
        )
        self.jpeg_buf_a = np.ndarray(
            (self.JPEG_MAX_BYTES,), dtype=np.uint8, buffer=self.jpeg_shm_a.buf,
        )
        self.jpeg_buf_b = np.ndarray(
            (self.JPEG_MAX_BYTES,), dtype=np.uint8, buffer=self.jpeg_shm_b.buf,
        )
        self.jlen_a = np.ndarray((1,), dtype=np.int32, buffer=self.jlen_shm_a.buf)
        self.jlen_b = np.ndarray((1,), dtype=np.int32, buffer=self.jlen_shm_b.buf)
        self.jlen_a[0] = 0
        self.jlen_b[0] = 0
    
    def release_shared_memory(self):
        self.image_array_a = None
        self.image_array_b = None
        self.fid_array_a = None
        self.fid_array_b = None
        self.write_flag   = None
        self.jpeg_buf_a = None
        self.jpeg_buf_b = None
        self.jlen_a = None
        self.jlen_b = None

        for attr in (
            "image_shm_a", "image_shm_b", "fid_shm_a", "fid_shm_b",
            "write_flag_shm", "jpeg_shm_a", "jpeg_shm_b",
            "jlen_shm_a", "jlen_shm_b",
        ):
            shm = getattr(self, attr, None)
            if shm is None:
                continue
            try:
                shm.close()
            except Exception:
                pass
            try:
                shm.unlink()
            except FileNotFoundError:
                pass
            setattr(self, attr, None)
        print(f"[INFO] Camera {self.name} shared memory released.")

    def clear_shared_memory(self):
        self.fid_array_a[0] = 0
        self.fid_array_b[0] = 0
        self.write_flag[0] = 0

        self.image_array_a.fill(0)
        self.image_array_b.fill(0)
        self.jlen_a[0] = 0
        self.jlen_b[0] = 0

    def _resolve_video_path(self, save_path):
        """Resolve a save_path (dir or file) to a concrete .avi file path."""
        _, ext = os.path.splitext(save_path)
        if not ext:
            return os.path.join(save_path, f"{self.name}.avi")
        return save_path

    def record_start(self, save_path, fps=30):
        """Begin writing .avi mid-stream (no acquisition restart).

        Safe to call while continuous_acquire() is running in any mode; the
        acquisition loop picks up event['record'] on its next iteration and
        opens the VideoWriter. The shared-memory stream is unaffected.
        """
        if save_path is None:
            print(f"[WARNING] Camera {self.name} record_start ignored: save_path is None")
            return
        self.record_save_path = self._resolve_video_path(save_path)
        self.record_fps = fps
        save_dir = os.path.dirname(self.record_save_path)
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
        self.event["record"].set()
        print(f"[INFO] Camera {self.name} recording armed -> {self.record_save_path}")

    def record_stop(self, flush_timeout=5.0):
        """Stop writing .avi mid-stream (stream keeps running).

        Blocks until the acquisition loop has actually released the
        VideoWriter (file flushed/finalized on disk), so the caller can rsync
        the .avi immediately afterwards without racing a half-written file.
        """
        self.event["record"].clear()
        # Only wait if a capture loop is running to act on it; otherwise the
        # writer is already closed (record_closed stays set).
        if self.event["acquisition"].is_set():
            if not self.event["record_closed"].wait(timeout=flush_timeout):
                print(f"[WARNING] Camera {self.name} record file not confirmed "
                      f"closed within {flush_timeout}s")
        print(f"[INFO] Camera {self.name} recording disarmed.")

    def start(self, mode, syncMode, save_path=None, fps=30):
        if fps < 0 and mode in ["video", "full"] and syncMode is False:
            self.event["error"].set()
            self.event["error_reset"].clear()
            
            self.last_error = "FPS must be non-negative for video or full mode when syncMode is False."
            self.last_traceback = ""
        
        if mode in ["video", "full", "image"] and save_path is None:
            self.event["error"].set()
            self.event["error_reset"].clear()
            
            self.last_error = "Save path must be specified for video or image saving."
            self.last_traceback = ""
            
        if self.event["start"].is_set():
            self.event["error"].set()
            self.event["error_reset"].clear()
            
            self.last_error = "Acquisition is already running."
            self.last_traceback = ""
            print(f"[WARNING] Camera {self.name} acquisition is already running.")
            
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

        # Backward compat: "video"/"full" record from the start. We express
        # this through the same record-toggle path so continuous_acquire() has
        # a single code path. "stream" starts disarmed (recording toggled at
        # runtime via record_start/record_stop). "image" is unaffected.
        if mode in ["video", "full"] and save_path is not None:
            self.record_save_path = self.save_path
            self.record_fps = fps
            self.event["record"].set()
        elif mode == "stream":
            self.event["record"].clear()

        self.event["stop"].clear()
        self.event["start"].set()
        print(f"[INFO] Camera {self.name} will start.")  
        
        self.event["acquisition"].wait()   
        print(f"[INFO] Camera {self.name} acquisition started.")
    
    def error_reset(self):
        self.last_error = None
        self.last_traceback = None
        
        self.event["error"].clear()
        self.event["error_reset"].set()
        
    def get_error(self):
        if self.event["error"].is_set():
            return True, (self.last_error, self.last_traceback)
        return False, (None, None)

    def stop(self):
        self.event["start"].clear()
        
        if self.event["error"].is_set():
            self.error_reset()
            
        self.event["stop"].wait()
        print(f"[INFO] Camera {self.name} has been stopped.")
           
    def end(self):
        if self.event["start"].is_set():
            self.stop()
        
        self.event["exit"].set()
        self.capture_thread.join()
        print(f"[INFO] Camera {self.name} has been successfully ended.")
    
    def continuous_acquire(self):
        stream = (self.mode in ["stream", "full"])
        blank_frame = np.zeros(self.frame_shape, dtype=np.uint8)
        blank_frame[::2, ::2] = 255  # checkerboard pattern for dropped frames

        # Recording is driven entirely by self.event["record"] (armed at
        # start() for video/full, or toggled at runtime via record_start/
        # record_stop). The writer is created/closed inside the loop so it can
        # turn on/off WITHOUT interrupting acquisition or the shm stream.
        video_writer = None
        rec_prev_fid = None  # episode-local frame id for blank-fill

        def _open_writer():
            if self.record_save_path is None:
                print(f"[WARNING] Camera {self.name} record requested but no save_path; ignoring.")
                return None
            fourcc = cv2.VideoWriter_fourcc(*'MJPG')
            w = cv2.VideoWriter(
                self.record_save_path, fourcc, fps=self.record_fps,
                frameSize=(self.frame_shape[1], self.frame_shape[0]),
            )
            self.event["record_closed"].clear()  # a file is now open
            print(f"[INFO] Camera {self.name} recording -> {self.record_save_path}")
            return w

        def _close_writer(w):
            if w is not None:
                w.release()
                print(f"[INFO] Camera {self.name} recording file closed.")
            # Signal record_stop() that the .avi is fully flushed.
            self.event["record_closed"].set()

        print(f"[INFO] Camera {self.name} starting continuous acquisition.")

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

            if stream:
                self.clear_shared_memory()
            print(f"[INFO] Camera {self.name} acquisition aborted due to error during start.")
            return


        self.event["acquisition"].set()

        while self.event["start"].is_set() and not self.event["exit"].is_set():
            try:
                # Toggle the recording file on/off mid-stream.
                want_record = self.event["record"].is_set()
                if want_record and video_writer is None:
                    video_writer = _open_writer()
                    rec_prev_fid = None
                elif not want_record and video_writer is not None:
                    _close_writer(video_writer)
                    video_writer = None
                    rec_prev_fid = None

                frame, frame_data = self.camera.get_image()
                if frame is None:
                    continue

                current_frame_id = frame_data["frameID"]
                if self.last_frame_id > current_frame_id:
                    continue  # Skip out-of-order frames

                if video_writer is not None:
                    if rec_prev_fid is None:
                        # First frame of this recording episode.
                        rec_prev_fid = current_frame_id - 1
                    for _ in range(current_frame_id - rec_prev_fid - 1):
                        print(f"frame drop {self.name}: missing frame id", current_frame_id - rec_prev_fid - 1)
                        video_writer.write(blank_frame)
                    video_writer.write(frame)
                    rec_prev_fid = current_frame_id

                if stream:
                    # Encode once; raw + JPEG share the same flag toggle so
                    # downstream readers always see a matched (raw, jpeg) pair.
                    ok, jpg = cv2.imencode(
                        '.jpg', frame,
                        [cv2.IMWRITE_JPEG_QUALITY, self.JPEG_QUALITY],
                    )
                    jlen = int(jpg.size) if ok else 0
                    if jlen > self.JPEG_MAX_BYTES:
                        # Bigger than the slot — keep raw only, clear length so
                        # readers skip this frame on the JPEG path.
                        jlen = 0

                    # Write to shared memory
                    if self.write_flag[0] == 0:
                        np.copyto(self.image_array_a, frame)
                        self.fid_array_a[0] = current_frame_id
                        if jlen > 0:
                            self.jpeg_buf_a[:jlen] = jpg.reshape(-1)
                        self.jlen_a[0] = jlen
                        self.write_flag[0] = 1
                    else:
                        np.copyto(self.image_array_b, frame)
                        self.fid_array_b[0] = current_frame_id
                        if jlen > 0:
                            self.jpeg_buf_b[:jlen] = jpg.reshape(-1)
                        self.jlen_b[0] = jlen
                        self.write_flag[0] = 0

                self.last_frame_id = current_frame_id

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

        try:
            self.camera.stop()
        except Exception as e:
            self.event["error"].set()
            self.event["error_reset"].clear()
            self.last_error = str(e)
            self.last_traceback = traceback.format_exc()
            print(f"[ERROR] Camera {self.name} exception occurred during stop:")
            print(f"Exception Type: {type(e).__name__}")
            print(f"Exception Message: {str(e)}")
            print(self.last_traceback)
        finally:
            self.event["acquisition"].clear()

            if stream:
                self.clear_shared_memory()

            _close_writer(video_writer)
            video_writer = None

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
        print(f"[INFO] Camera {self.name} shared memory released.")
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
        return self.last_frame_id

    def get_status(self):
        return {
            'state': self.get_state(),
            'frame_id': self.get_frame_id(),
            'name': self.name,
            'mode': getattr(self, 'mode', None),
            'fps': getattr(self, 'fps', None),
            'syncMode': getattr(self, 'syncMode', None),
            'save_path': getattr(self, 'save_path', None),
            'recording': self.event["record"].is_set(),
            'record_writer_open': not self.event["record_closed"].is_set(),
            'record_save_path': self.record_save_path,
            'time': time.time()
        }

    def run(self):
        self.connect_camera()
        
        while not self.event["exit"].is_set(): # we should maintain the connection until exit
            if self.event["start"].is_set(): # Start data acquisition
                if self.mode in ["full", "video", "stream"]:
                    print(f"[INFO] Camera {self.name} entering continuous acquisition mode.")
                    self.continuous_acquire()
                else:
                    self.single_acquire()
            
            time.sleep(0.001)
                    
        self.release()
        
