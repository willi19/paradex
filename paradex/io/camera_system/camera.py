from threading import Event, Thread
import time
import cv2
from multiprocessing import shared_memory
import numpy as np
import os
import traceback


class Camera():
    """One FLIR camera plus its dedicated capture thread.

    A ``Camera`` owns a single hardware camera (via :class:`PyspinCamera`) and a
    background **capture thread** (:meth:`run`). The thread and the calling thread
    coordinate through a set of :class:`threading.Event` objects (``start``,
    ``stop``, ``exit``, ``acquisition`` ...) — a handshake, not plain flags.

    Frames are routed to *sinks* depending on ``mode``:

    - ``stream`` / ``full`` → a double-buffered shared-memory block read by
      :class:`MultiCameraReader`.
    - ``video`` / ``full`` → an ``.avi`` file on disk.
    - ``image`` → a single still.

    Parameters
    ----------
    cam_type : str
        Backend, currently only ``"pyspin"``.
    name : str
        Camera serial number (used as the SHM key and file stem).
    frame_shape : tuple of int, optional
        ``(height, width, channels)`` of a frame, by default ``(1536, 2048, 3)``.
    """

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

        self.last_frame_id = 0

        self.last_error = None
        self.last_traceback = None

        self.load_shared_memory()
        print(f"[INFO] Camera {self.name} shared memory loaded.")
        self.capture_thread = Thread(target=self.run)
        self.capture_thread.start()

        self.event["connection"].wait()

    def _cleanup_existing_shm(self):
        """Unlink any leftover shared-memory blocks for this serial.

        A process that was ``SIGKILL``-ed cannot free its SHM, so the segments
        leak. This is called before allocating fresh ones so a restart cleans up
        after a previous crash.
        """
        shm_names = [
            self.name + "_image_a",
            self.name + "_image_b",
            self.name + "_fid_a",
            self.name + "_fid_b",
            self.name + "_flag"
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
        """Allocate the double-buffered shared memory for streaming.

        Creates two image buffers (``_image_a`` / ``_image_b``), a frame-id per
        buffer, and a ``_flag`` selecting the buffer being written. Readers use
        the flag to pick the stable buffer. Called once from ``__init__``.
        """
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
        """Close and unlink every shared-memory block (frees the OS segments)."""
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
        print(f"[INFO] Camera {self.name} shared memory released.")

    def clear_shared_memory(self):
        """Zero the frame ids, write flag, and both image buffers."""
        self.fid_array_a[0] = 0
        self.fid_array_b[0] = 0
        self.write_flag[0] = 0

        self.image_array_a.fill(0)
        self.image_array_b.fill(0)

    def start(self, mode, syncMode, save_path=None, fps=30, exposure_time=None, gain=None, timeout=10.0):
        """Begin capture; blocks until the camera is actually acquiring.

        Validates the arguments (sets the error state on a bad combination),
        stores the parameters, resolves ``save_path`` to a per-camera file, then
        signals the capture thread via ``event["start"]`` and waits on
        ``event["acquisition"]`` (the handshake).

        Parameters
        ----------
        mode : str
            ``image`` / ``video`` / ``stream`` / ``full``.
        syncMode : bool
            Wait on the hardware trigger if ``True``.
        save_path : str, optional
            Output dir/file for ``video``/``image`` (required for those modes).
        fps : int, optional
            Frame rate for free-run, by default 30.
        exposure_time, gain : float, optional
            Overrides; ``None`` keeps the camera.json baseline.
        timeout : float, optional
            Max seconds to wait for the capture thread to begin acquiring before
            logging a warning and returning, by default 10.0.
        """
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
        self.exposure_time = exposure_time
        self.gain = gain
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
        print(f"[INFO] Camera {self.name} will start.")

        if not self.event["acquisition"].wait(timeout=timeout):
            # Surface the failure instead of silently returning "started": clear
            # start and set the error state so CameraLoader.get_all_errors and the
            # daemon report a start failure.
            print(f"[WARN] Camera {self.name} start() timed out after {timeout}s "
                  f"waiting for acquisition to begin.")
            self.event["start"].clear()
            self.last_error = f"start() timed out after {timeout}s waiting for acquisition"
            self.last_traceback = ""
            self.event["error"].set()
            self.event["error_reset"].clear()
            return
        print(f"[INFO] Camera {self.name} acquisition started.")

    def error_reset(self):
        """Clear the error state and release anyone waiting on ``error_reset``."""
        self.last_error = None
        self.last_traceback = None

        self.event["error"].clear()
        self.event["error_reset"].set()

    def get_error(self):
        """Return the current error state.

        Returns
        -------
        tuple
            ``(has_error, (message, traceback))``; ``(False, (None, None))`` when
            there is no error.
        """
        if self.event["error"].is_set():
            return True, (self.last_error, self.last_traceback)
        return False, (None, None)

    def stop(self, timeout=5.0):
        """Stop the current capture.

        Clears ``event["start"]`` (so the acquire loop exits on its next
        iteration) and waits up to ``timeout`` for ``event["stop"]``. If the wait
        times out (a wedged acquire thread) it logs and returns instead of
        blocking forever.

        Parameters
        ----------
        timeout : float, optional
            Max seconds to wait for the thread to acknowledge, by default 5.0.
        """
        self.event["start"].clear()

        if self.event["error"].is_set():
            self.error_reset()

        if not self.event["stop"].wait(timeout=timeout):
            print(f"[WARN] Camera {self.name} stop() timed out after {timeout}s "
                  f"(acquire thread may be stuck); continuing without blocking.")
            return
        print(f"[INFO] Camera {self.name} has been stopped.")

    def end(self, timeout=5.0):
        """Stop, then release the camera (DeInit + free SHM) and join the thread.

        Parameters
        ----------
        timeout : float, optional
            Per-step wait bound (stop + thread join), by default 5.0. If the
            thread is still alive after the join it logs and returns rather than
            blocking the daemon.
        """
        if self.event["start"].is_set():
            self.stop(timeout=timeout)

        self.event["exit"].set()
        self.capture_thread.join(timeout=timeout)
        if self.capture_thread.is_alive():
            print(f"[WARN] Camera {self.name} end(): capture thread still alive after "
                  f"{timeout}s; not blocking daemon.")
            return
        print(f"[INFO] Camera {self.name} has been successfully ended.")

    def continuous_acquire(self):
        """Capture loop for ``video`` / ``stream`` / ``full`` modes.

        Arms the PySpin camera (``BeginAcquisition``), signals ``acquisition``,
        then loops grabbing frames until ``start`` clears or ``exit`` sets. Each
        frame is written to the video sink (with blank frames inserted for drops)
        and/or the SHM double buffer. On any exception it records the error and
        waits for ``error_reset``. Runs on the capture thread only.
        """
        save_video = (self.mode in ["video", "full"] and self.save_path is not None)
        stream = (self.mode in ["stream", "full"])
        blank_frame = np.zeros(self.frame_shape, dtype=np.uint8)
        blank_frame[::2, ::2] = 255  # checkerboard pattern for dropped frames

        print(f"[INFO] Camera {self.name} starting continuous acquisition.")
        if save_video:
            fourcc = cv2.VideoWriter_fourcc(*'MJPG')
            video_writer = cv2.VideoWriter(self.save_path, fourcc, fps=self.fps, frameSize=(self.frame_shape[1], self.frame_shape[0]))

        try:
            self.camera.start("continuous", self.syncMode, self.fps, gain=self.gain, exposure_time=self.exposure_time)

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
            print(f"[INFO] Camera {self.name} acquisition aborted due to error during start.")
            return


        self.event["acquisition"].set()

        while self.event["start"].is_set() and not self.event["exit"].is_set():
            try:
                frame, frame_data = self.camera.get_image()
                if frame is None:
                    continue

                current_frame_id = frame_data["frameID"]
                if self.last_frame_id > current_frame_id:
                    continue  # Skip out-of-order frames

                if save_video:
                    for _ in range(current_frame_id - self.last_frame_id-1):
                        print(f"frame drop {self.name}: missing frame id", current_frame_id-self.last_frame_id-1)
                        video_writer.write(blank_frame)
                    video_writer.write(frame)

                if stream:
                    # Write to shared memory
                    if self.write_flag[0] == 0:
                        np.copyto(self.image_array_a, frame)
                        self.fid_array_a[0] = current_frame_id
                        self.write_flag[0] = 1
                    else:
                        np.copyto(self.image_array_b, frame)
                        self.fid_array_b[0] = current_frame_id
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
            print(f"[WARN] Camera {self.name} continuous_acquire stop() failed: {e}")
        self.event["acquisition"].clear()

        if stream:
            self.clear_shared_memory()

        if save_video:
            video_writer.release()

        self.event["stop"].set()

    def single_acquire(self, max_attempts=5):
        """Capture one frame for ``image`` mode and write it to disk.

        Because ``get_image()`` now times out instead of blocking, this retries a
        few times so a transient miss still yields a frame without hanging on a
        dead link.

        Parameters
        ----------
        max_attempts : int, optional
            Grab attempts before giving up (and skipping the write), by default 5.
        """
        try:
            self.camera.start("single", self.syncMode, gain=self.gain, exposure_time=self.exposure_time)
            self.event["acquisition"].set()

            # get_image() now times out instead of blocking forever, so retry a few
            # times to stay reliable on a transient miss without hanging on a dead link.
            frame = None
            for _ in range(max_attempts):
                if self.event["exit"].is_set():
                    break
                frame, _ = self.camera.get_image()
                if frame is not None and getattr(frame, "size", 0) > 0:
                    break

            if frame is not None and getattr(frame, "size", 0) > 0:
                cv2.imwrite(self.save_path, frame)
            else:
                print(f"[WARN] Camera {self.name}: single_acquire got no frame after "
                      f"{max_attempts} attempts, skipping write")

        except Exception as e:
            self.event["error"].set()
            self.event["error_reset"].clear()
            self.last_error = str(e)
            self.last_traceback = traceback.format_exc()
            print(f"[ERROR] Camera {self.name} single_acquire error: {e}")
            print(self.last_traceback)
            self.event["acquisition"].set()   # release Camera.start()'s wait on failure

        finally:
            # Always leave a deterministic state: acquisition/start cleared, camera
            # acquisition ended (guarded), and stop signalled so stop()/end() return.
            self.event["acquisition"].clear()
            self.event["start"].clear()
            try:
                self.camera.stop()
            except Exception as e:
                print(f"[WARN] Camera {self.name} single_acquire stop() failed: {e}")
            self.event["stop"].set()

    def connect_camera(self):
        """Open the underlying hardware camera and signal ``connection``.

        Raises
        ------
        NotImplementedError
            If ``cam_type`` is not ``"pyspin"``.
        """
        # Establish connection
        if self.type == "pyspin":
            from paradex.io.camera_system.pyspin import load_camera
        else:
            raise NotImplementedError(f"Camera type {self.type} is not implemented.")

        self.camera = load_camera(self.name)
        self.event["connection"].set()

    def release(self):
        """Disconnect the camera (``DeInit``) and free its shared memory."""
        self.camera.release()
        self.release_shared_memory()
        print(f"[INFO] Camera {self.name} shared memory released.")
        self.event["release"].set()

    def get_state(self):
        """Return the lifecycle state as a string.

        Returns
        -------
        str
            One of ``CONNECTING``, ``READY``, ``STARTING``, ``CAPTURING``,
            ``STOPPED``, or ``ERROR: <message>``.
        """
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
        """Return the id of the last frame processed by the capture loop."""
        return self.last_frame_id

    def get_status(self):
        """Return a status snapshot dict.

        Returns
        -------
        dict
            Keys: ``state``, ``frame_id``, ``name``, ``mode``, ``fps``,
            ``syncMode``, ``save_path``, ``time``.
        """
        has_error, (err_msg, _tb) = self.get_error()
        return {
            'state': "ERROR" if has_error else self.get_state(),
            'error': err_msg if has_error else None,
            'frame_id': self.get_frame_id(),
            'name': self.name,
            'mode': getattr(self, 'mode', None),
            'fps': getattr(self, 'fps', None),
            'syncMode': getattr(self, 'syncMode', None),
            'save_path': getattr(self, 'save_path', None),
            'time': time.time()
        }

    def run(self):
        """Capture-thread main loop.

        Connects the camera, then until ``exit`` is set: when ``start`` is set,
        dispatches to :meth:`continuous_acquire` (video/stream/full) or
        :meth:`single_acquire` (image). Releases the camera on exit.
        """
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
