from threading import Event, Thread, Lock
import time
import cv2
from multiprocessing import shared_memory
import numpy as np
import os
import traceback

from paradex.utils.log import get_logger

logger = get_logger("camera")


class Camera():
    """One FLIR camera plus its dedicated capture thread.

    A ``Camera`` owns a single hardware camera (via :class:`PyspinCamera`) and a
    background **capture thread** (:meth:`run`). The thread and the calling thread
    coordinate through a set of :class:`threading.Event` objects (``start``,
    ``stop``, ``exit``, ``acquisition`` ...) — a handshake, not plain flags.

    Two acquisition modes: ``image`` (one single-frame still) and ``acquire``
    (continuous). In ``acquire`` mode frames are routed to whichever *sinks* are
    enabled — toggled independently at runtime via :meth:`set_sink`:

    - **stream** sink → a double-buffered shared-memory block read by
      :class:`MultiCameraReader`.
    - **video** sink → an ``.avi`` file on disk.
    - **snapshot** sink → the next N frames written as images.

    (``video`` / ``stream`` / ``full`` are no longer modes — that's just
    ``acquire`` with the corresponding sink(s) on.)

    Parameters
    ----------
    cam_type : str
        Backend, currently only ``"pyspin"``.
    name : str
        Camera serial number (used as the SHM key and file stem).
    frame_shape : tuple of int, optional
        ``(height, width, channels)`` of a frame, by default ``(1536, 2048, 3)``.
    cfg : dict, optional
        Per-serial ``camera.json`` entry passed through to the PySpin backend.
    """

    def __init__(self, cam_type, name, frame_shape=(1536, 2048, 3), cfg=None):
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
        self.event["stop"].set()

        self.type = cam_type
        self.name = name
        self.cfg = cfg or {}

        self.frame_shape = frame_shape

        self.last_frame_id = 0

        # ── runtime sinks (decoupled from a fixed capture mode) ──────────────
        # Desired sink state; the acquire loop reconciles the real VideoWriter /
        # SHM against these under _sink_lock, so sinks can be toggled mid-capture.
        self._sink_lock = Lock()
        self._want_video = False           # .avi video sink
        self._video_path = None            # dir or file for the video sink
        self._want_stream = False          # shared-memory double-buffer sink
        self._snapshot = None              # {'path','count','remaining'} one-shot image sink
        self._pending_param = None         # {'gain','exposure'} to apply live on the capture thread

        self.last_error = None
        self.last_traceback = None

        self.load_shared_memory()
        logger.info(f"Camera {self.name} shared memory loaded.")
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
        logger.info(f"unlink existing shm if any: {shm_names}")
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
        logger.info(f"Camera {self.name} shared memory released.")

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
            ``image`` (single-frame still) or ``acquire`` (continuous; enable
            outputs with :meth:`set_sink`).
        syncMode : bool
            Wait on the hardware trigger if ``True``.
        save_path : str, optional
            Output dir/file for ``image`` (required for ``image`` mode).
        fps : int, optional
            Frame rate for free-run, by default 30.
        exposure_time, gain : float, optional
            Overrides; ``None`` keeps the camera.json baseline.
        timeout : float, optional
            Max seconds to wait for the capture thread to begin acquiring before
            logging a warning and returning, by default 10.0.
        """
        if mode not in ("image", "acquire"):
            self.event["error"].set()
            self.event["error_reset"].clear()

            self.last_error = (f"invalid mode '{mode}': use 'image' (single frame) or "
                               f"'acquire' + set_sink()/set_record()/set_stream().")
            self.last_traceback = ""

        if fps < 0 and mode == "acquire" and syncMode is False:
            self.event["error"].set()
            self.event["error_reset"].clear()

            self.last_error = "FPS must be non-negative for free-run acquisition."
            self.last_traceback = ""

        if mode == "image" and save_path is None:
            self.event["error"].set()
            self.event["error_reset"].clear()

            self.last_error = "Save path must be specified for image capture."
            self.last_traceback = ""

        if self.event["start"].is_set():
            self.event["error"].set()
            self.event["error_reset"].clear()

            self.last_error = "Acquisition is already running."
            self.last_traceback = ""
            logger.warning(f"Camera {self.name} acquisition is already running.")

        if self.event["error"].is_set():
            logger.warning(f"Camera {self.name} is in ERROR state. Resetting error state.")
            self.event["stop"].set()
            return

        self.mode = mode
        self.syncMode = syncMode
        self.fps = fps
        self.exposure_time = exposure_time
        self.gain = gain
        self.last_frame_id = 0

        # Modes are only "image" (single-frame one-shot) or "acquire" (continuous).
        # Acquisition begins with all sinks OFF; enable them at runtime via
        # set_sink()/set_record()/set_stream(). video/stream/full are gone.
        with self._sink_lock:
            self._want_video = False
            self._want_stream = False
            self._video_path = None
            self._snapshot = None

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
        logger.info(f"Camera {self.name} will start.")

        if not self.event["acquisition"].wait(timeout=timeout):
            # Surface the failure instead of silently returning "started": clear
            # start and set the error state so CameraLoader.get_all_errors and the
            # daemon report a start failure.
            logger.warning(f"Camera {self.name} start() timed out after {timeout}s "
                           f"waiting for acquisition to begin.")
            self.event["start"].clear()
            self.last_error = f"start() timed out after {timeout}s waiting for acquisition"
            self.last_traceback = ""
            self.event["error"].set()
            self.event["error_reset"].clear()
            self.event["stop"].set()
            return
        logger.info(f"Camera {self.name} acquisition started.")

    def _resolve_save_path(self, save_path, default_ext):
        """Turn a dir-or-file ``save_path`` into a concrete per-camera file path.

        A path with no extension is treated as a directory and the file becomes
        ``<save_path>/<serial><default_ext>``; a path with an extension is used
        verbatim. Returns ``None`` if ``save_path`` is ``None``.
        """
        if save_path is None:
            return None
        _, ext = os.path.splitext(save_path)
        if not ext:
            return os.path.join(save_path, f"{self.name}{default_ext}")
        return save_path

    def set_sink(self, video=None, stream=None, save_path=None, snapshot=None):
        """Toggle output sinks at runtime — safe to call while capturing.

        Records only the *desired* sink state; the acquire loop opens/closes the
        real ``VideoWriter`` and SHM on the capture thread, so nothing here
        touches OpenCV/SHM from the caller's thread.

        Parameters
        ----------
        video : bool, optional
            Turn the ``.avi`` video sink on/off. Use ``save_path`` to set where.
        stream : bool, optional
            Turn the shared-memory stream sink on/off.
        save_path : str, optional
            Destination dir/file for the video sink (applied when it turns on).
        snapshot : tuple, optional
            ``(path, count)`` — write the next ``count`` frames as images.
        """
        with self._sink_lock:
            if save_path is not None:
                self._video_path = save_path
            if video is not None:
                self._want_video = bool(video)
            if stream is not None:
                self._want_stream = bool(stream)
            if snapshot is not None:
                path, count = snapshot
                self._snapshot = {'path': path, 'count': int(count), 'remaining': int(count)}

    def set_param(self, gain=None, exposure=None):
        """Apply new gain (dB) / exposure (microseconds) to the camera **live**.

        Recorded here and applied on the capture thread at the next acquire
        iteration (FLIR ``Gain``/``ExposureTime`` are writable during
        acquisition), so it is safe to call while capturing. Takes effect only in
        ``acquire`` mode while running.

        Parameters
        ----------
        gain : float, optional
            New gain in dB.
        exposure : float, optional
            New exposure time in microseconds.
        """
        with self._sink_lock:
            p = dict(self._pending_param or {})
            if gain is not None:
                p['gain'] = float(gain)
            if exposure is not None:
                p['exposure'] = float(exposure)
            self._pending_param = p

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
        was_active = self.event["start"].is_set() or self.event["acquisition"].is_set()
        self.event["start"].clear()

        if self.event["error"].is_set() and was_active:
            self.error_reset()

        if not self.event["stop"].wait(timeout=timeout):
            logger.warning(f"Camera {self.name} stop() timed out after {timeout}s "
                           f"(acquire thread may be stuck); continuing without blocking.")
            return
        logger.info(f"Camera {self.name} has been stopped.")

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
            logger.warning(f"Camera {self.name} end(): capture thread still alive after "
                           f"{timeout}s; not blocking daemon.")
            return
        logger.info(f"Camera {self.name} has been successfully ended.")

    def acquire(self):
        """Unified capture loop: acquire continuously, route to runtime sinks.

        Arms the PySpin camera (``BeginAcquisition``), signals ``acquisition``,
        then loops grabbing frames until ``start`` clears or ``exit`` sets. Each
        frame is routed to whichever sinks are currently enabled — video, SHM
        stream, one-shot snapshot — read fresh every iteration from the
        :meth:`set_sink` state, so sinks can be toggled live. The real
        ``VideoWriter`` is opened/closed *here* (capture thread) when the video
        sink flips. On any exception it records the error and waits for
        ``error_reset``. Runs on the capture thread only.
        """
        blank_frame = np.zeros(self.frame_shape, dtype=np.uint8)
        blank_frame[::2, ::2] = 255  # checkerboard pattern for dropped frames
        fourcc = cv2.VideoWriter_fourcc(*'MJPG')
        video_writer = None
        video_last_fid = None        # drop-accounting baseline, reset when the sink opens

        logger.info(f"Camera {self.name} starting continuous acquisition.")
        try:
            self.camera.start("continuous", self.syncMode, self.fps, gain=self.gain, exposure_time=self.exposure_time)

        except Exception as e:
            self.event["error"].set()
            self.event["error_reset"].clear()

            self.last_error = str(e)
            self.last_traceback = traceback.format_exc()

            logger.error(f"Camera {self.name} exception occurred:\n"
                         f"Exception Type: {type(e).__name__}\n"
                         f"Exception Message: {str(e)}\n"
                         f"{self.last_traceback}")

            self.event["acquisition"].set()  # To avoid deadlock

            self.event["error_reset"].wait()

            self.event["acquisition"].clear()
            self.event["stop"].set()

            self.clear_shared_memory()
            logger.info(f"Camera {self.name} acquisition aborted due to error during start.")
            return


        self.event["acquisition"].set()

        while self.event["start"].is_set() and not self.event["exit"].is_set():
            try:
                # Read the desired sink + param state once per iteration.
                with self._sink_lock:
                    want_video = self._want_video
                    video_path = self._video_path
                    want_stream = self._want_stream
                    snap = self._snapshot
                    pending_param = self._pending_param
                    self._pending_param = None

                # Apply a live gain/exposure change on this (capture) thread.
                if pending_param:
                    try:
                        if 'gain' in pending_param:
                            self.camera.set_gain(pending_param['gain'])
                            self.gain = pending_param['gain']
                        if 'exposure' in pending_param:
                            self.camera.set_exposure(pending_param['exposure'])
                            self.exposure_time = pending_param['exposure']
                        logger.info(f"Camera {self.name} live param -> {pending_param}")
                    except Exception as e:
                        logger.warning(f"Camera {self.name} set_param failed: {e}")

                # Reconcile the video sink on this (capture) thread.
                if want_video and video_writer is None:
                    resolved = self._resolve_save_path(video_path, ".avi")
                    if resolved is None:
                        logger.warning(f"Camera {self.name} video sink requested without a save_path; ignoring.")
                    else:
                        d = os.path.dirname(resolved)
                        if d:
                            os.makedirs(d, exist_ok=True)
                        video_writer = cv2.VideoWriter(resolved, fourcc, self.fps,
                                                       (self.frame_shape[1], self.frame_shape[0]))
                        video_last_fid = None
                        logger.info(f"Camera {self.name} video sink open -> {resolved}")
                elif not want_video and video_writer is not None:
                    video_writer.release()
                    video_writer = None
                    logger.info(f"Camera {self.name} video sink closed")

                frame, frame_data = self.camera.get_image()
                if frame is None:
                    continue

                current_frame_id = frame_data["frameID"]
                if self.last_frame_id > current_frame_id:
                    continue  # Skip out-of-order frames

                # Video sink: insert blanks for frames dropped since the sink opened.
                if video_writer is not None:
                    if video_last_fid is not None:
                        for _ in range(current_frame_id - video_last_fid - 1):
                            video_writer.write(blank_frame)
                    video_writer.write(frame)
                    video_last_fid = current_frame_id

                # Stream sink: shared-memory double buffer.
                if want_stream:
                    if self.write_flag[0] == 0:
                        np.copyto(self.image_array_a, frame)
                        self.fid_array_a[0] = current_frame_id
                        self.write_flag[0] = 1
                    else:
                        np.copyto(self.image_array_b, frame)
                        self.fid_array_b[0] = current_frame_id
                        self.write_flag[0] = 0

                # Snapshot sink: write the next N frames as images, then clear.
                if snap is not None and snap['remaining'] > 0:
                    base = self._resolve_save_path(snap['path'], ".png")
                    if base is not None:
                        if snap['count'] > 1:
                            stem, ext = os.path.splitext(base)
                            base = f"{stem}_{snap['count'] - snap['remaining']}{ext}"
                        d = os.path.dirname(base)
                        if d:
                            os.makedirs(d, exist_ok=True)
                        cv2.imwrite(base, frame)
                    with self._sink_lock:
                        if self._snapshot is not None:
                            self._snapshot['remaining'] -= 1
                            if self._snapshot['remaining'] <= 0:
                                self._snapshot = None

                self.last_frame_id = current_frame_id

            except Exception as e:
                self.event["error"].set()
                self.event["error_reset"].clear()

                self.last_error = str(e)
                self.last_traceback = traceback.format_exc()

                logger.error(f"Camera {self.name} exception occurred during acquisition:\n"
                             f"Exception Type: {type(e).__name__}\n"
                             f"Exception Message: {str(e)}\n"
                             f"{self.last_traceback}")

                self.event["error_reset"].wait()
                break

        try:
            self.camera.stop()
        except Exception as e:
            logger.warning(f"Camera {self.name} acquire stop() failed: {e}")
        self.event["acquisition"].clear()

        if video_writer is not None:
            video_writer.release()
        self.clear_shared_memory()

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
                logger.warning(f"Camera {self.name}: single_acquire got no frame after "
                               f"{max_attempts} attempts, skipping write")

        except Exception as e:
            self.event["error"].set()
            self.event["error_reset"].clear()
            self.last_error = str(e)
            self.last_traceback = traceback.format_exc()
            logger.error(f"Camera {self.name} single_acquire error: {e}\n"
                         f"{self.last_traceback}")
            self.event["acquisition"].set()   # release Camera.start()'s wait on failure

        finally:
            # Always leave a deterministic state: acquisition/start cleared, camera
            # acquisition ended (guarded), and stop signalled so stop()/end() return.
            self.event["acquisition"].clear()
            self.event["start"].clear()
            try:
                self.camera.stop()
            except Exception as e:
                logger.warning(f"Camera {self.name} single_acquire stop() failed: {e}")
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

        try:
            self.camera = load_camera(self.name, cfg=self.cfg)
        except Exception as e:
            self.last_error = str(e)
            self.last_traceback = traceback.format_exc()
            self.event["error"].set()
            self.event["error_reset"].clear()
            logger.error(f"Camera {self.name} connection failed: {e}\n"
                         f"{self.last_traceback}")
        finally:
            self.event["connection"].set()

    def release(self):
        """Disconnect the camera (``DeInit``) and free its shared memory."""
        if hasattr(self, "camera"):
            try:
                self.camera.release()
            except Exception as e:
                logger.warning(f"Camera {self.name} release() failed: {e}")
        self.release_shared_memory()
        logger.info(f"Camera {self.name} shared memory released.")
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
            'sinks': {'video': self._want_video, 'stream': self._want_stream},
            'fps': getattr(self, 'fps', None),
            'syncMode': getattr(self, 'syncMode', None),
            'save_path': getattr(self, 'save_path', None),
            'time': time.time()
        }

    def run(self):
        """Capture-thread main loop.

        Connects the camera, then until ``exit`` is set: when ``start`` is set,
        dispatches to :meth:`acquire` (mode ``acquire`` — continuous + runtime
        sinks) or :meth:`single_acquire` (mode ``image``). Releases on exit.
        """
        self.connect_camera()

        while not self.event["exit"].is_set(): # we should maintain the connection until exit
            if self.event["start"].is_set(): # Start data acquisition
                # "image" keeps the legacy one-shot path; every other mode
                # (video/stream/full/acquire) uses the unified sink-driven loop.
                if getattr(self, "mode", None) == "image":
                    self.single_acquire()
                else:
                    logger.info(f"Camera {self.name} entering continuous acquisition mode.")
                    self.acquire()

            time.sleep(0.001)

        self.release()
