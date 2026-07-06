from threading import Thread
import os
import time

from paradex.io.camera_system.camera import Camera
from paradex.utils.path import home_path, capture_path_list
from paradex.utils.system import get_camera_list, get_camera_config
from paradex.utils.log import get_logger

logger = get_logger("camera")

RETRY_COUNT = 5
RETRY_DELAY = 2  # seconds

# Fallback per-camera params when a serial is absent from camera.json.
DEFAULT_GAIN = 3.0
DEFAULT_EXPOSURE = 2500.0

class CameraLoader:
    """Owns every :class:`Camera` on this PC and drives them as one group.

    A ``CameraLoader`` enumerates the local cameras, builds one :class:`Camera`
    per detected serial, and fans lifecycle calls (``start`` / ``stop`` / ``end``)
    out across all of them in parallel threads. It also holds the per-serial
    gain/exposure baseline read from ``system/current/camera.json``, used to
    resolve capture parameters when the caller does not pass explicit overrides.

    Parameters
    ----------
    types : list of str, optional
        Camera backends to load, by default ``["pyspin"]``. Only ``"pyspin"``
        is currently supported; each listed backend triggers its loader
        (``"pyspin"`` calls :meth:`load_pyspin_camera`).
    """

    def __init__(self, types=["pyspin"]):
        self.cameralist = []
        self.camera_names = []
        self.expected_camera_count = len(get_camera_list())
        # Per-serial gain/exposure baseline (system/current/camera.json).
        self.cam_config = get_camera_config()

        for cam_type in types:
            if cam_type == "pyspin":
                self.load_pyspin_camera()
    
    def load_pyspin_camera(self, serial_list=None):
        """Enumerate PySpin (GigE) cameras and build a :class:`Camera` for each.

        Forces every detected camera onto its expected IP (``autoforce_ip``),
        then compares the detected serial count against the configured count
        (:func:`get_camera_list`). On a mismatch it retries up to ``RETRY_COUNT``
        times — re-running ``autoforce_ip`` each time, since after a power cycle
        GigE cameras boot slowly and may not be enumerated on the first pass —
        before proceeding with whatever cameras are present. The resulting
        :class:`Camera` objects and their serials are appended to
        ``self.cameralist`` / ``self.camera_names``.

        Parameters
        ----------
        serial_list : list of str, optional
            Explicit serials to load. When ``None`` (default) the serials are
            discovered via ``get_serial_list()``.
        """
        from paradex.io.camera_system.pyspin import get_serial_list, autoforce_ip
        
        expected = self.expected_camera_count
        autoforce_ip()

        if serial_list is None:
            serial_list = get_serial_list()

        # After a power cycle GigE cameras boot slowly and come up on the wrong IP.
        # autoforce_ip() only forces cameras that are already enumerated, so cameras
        # that were not ready at the first call would never be forced. Re-run
        # autoforce_ip() on every retry until the expected count appears.
        if len(serial_list) != expected:
            logger.warning(f"Configured camera count ({expected}) does not match "
                           f"detected camera count ({len(serial_list)}). Retrying (re-forcing IP)...")
            for _ in range(RETRY_COUNT):
                time.sleep(RETRY_DELAY)
                autoforce_ip()
                serial_list = get_serial_list()
                if len(serial_list) == expected:
                    logger.info("Camera count matched after retry.")
                    break
            else:
                logger.warning(f"Still {len(serial_list)}/{expected} cameras after "
                               f"{RETRY_COUNT} retries; proceeding with detected cameras.")
                
        self.cameralist = [
            Camera("pyspin", serial, cfg=self.cam_config.get(serial, {}))
            for serial in serial_list
        ]
        self.camera_names = self.camera_names + serial_list
    
    def start(self, mode, syncMode, save_path=None, fps=30, exposure_time=None, gain=None):
        """Start capture on every camera in parallel and block until all started.

        Resolves per-camera output for ``image`` mode (under ``home_path``);
        ``acquire`` mode uses no start-time path (the video sink's path is set
        later via :meth:`set_sink`). Modes: ``image`` or ``acquire``.
        For each camera the gain and exposure are resolved deterministically as
        **explicit arg > camera.json baseline > module default**: an explicit
        ``exposure_time`` / ``gain`` wins, otherwise the per-serial value from
        ``camera.json`` is used, otherwise ``DEFAULT_EXPOSURE`` /
        ``DEFAULT_GAIN``. Each camera's :meth:`Camera.start` runs on its own
        thread; the method joins all of them before returning.

        Parameters
        ----------
        mode : str
            ``image`` (single-frame) or ``acquire`` (continuous; sinks via
            :meth:`set_sink`).
        syncMode : bool
            Whether cameras wait on the hardware trigger.
        save_path : str, optional
            Session-relative output path for ``image`` / ``video`` / ``full``;
            ignored for ``stream``.
        fps : int, optional
            Frame rate for free-run capture, by default 30.
        exposure_time : float, optional
            Exposure override applied to all cameras; ``None`` (default) falls
            back to the camera.json baseline then ``DEFAULT_EXPOSURE``.
        gain : float, optional
            Gain override applied to all cameras; ``None`` (default) falls back
            to the camera.json baseline then ``DEFAULT_GAIN``.
        """
        if mode == "image":
            save_paths = [os.path.join(home_path, save_path, "images") for _ in self.cameralist]
            logger.info(f"image save paths: {save_paths}")
            for path in save_paths:
                os.makedirs(path, exist_ok=True)
        else:
            # "acquire": no sink at start; the video sink's path is resolved later
            # in set_sink() (spread across capture_path_list), snapshots under home.
            save_paths = [None for _ in self.cameralist]
        logger.info(f"starting cameras... cameras: {self.camera_names}")
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
        errors = self.get_all_errors()
        if errors:
            logger.warning(f"camera start completed with errors: {errors}")
        else:
            logger.info("all cameras started.")
    
    def set_sink(self, video=None, stream=None, save_path=None, snapshot=None):
        """Toggle output sinks on every camera at runtime (see :meth:`Camera.set_sink`).

        Resolves per-camera destinations the same way :meth:`start` does — the
        video sink spreads across ``capture_path_list``; snapshots land under
        ``home_path`` — then flips each camera's desired sink state. Cheap and
        non-blocking: the real VideoWriter opens/closes on each capture thread.

        Parameters
        ----------
        video : bool, optional
            Turn the video (.avi) sink on/off on all cameras.
        stream : bool, optional
            Turn the shared-memory stream sink on/off on all cameras.
        save_path : str, optional
            Session-relative dir for the video sink (applied when it turns on).
        snapshot : tuple, optional
            ``(save_path, count)`` — write the next ``count`` frames as images.
        """
        for ind, camera in enumerate(self.cameralist):
            kw = {}
            if video is not None:
                kw['video'] = video
            if stream is not None:
                kw['stream'] = stream
            if save_path is not None:
                kw['save_path'] = os.path.join(
                    capture_path_list[ind % len(capture_path_list)], save_path, "videos")
            if snapshot is not None:
                spath, count = snapshot
                kw['snapshot'] = (os.path.join(home_path, spath, "images"), count)
            if kw:
                camera.set_sink(**kw)

    def set_param(self, gain=None, exposure=None):
        """Apply gain/exposure to cameras live (see :meth:`Camera.set_param`).

        Each of ``gain`` / ``exposure`` may be a scalar (applied to every camera)
        or a ``{serial: value}`` dict (per-camera); ``None`` leaves it unchanged.

        Parameters
        ----------
        gain : float or dict, optional
            New gain in dB — scalar for all, or ``{serial: dB}``.
        exposure : float or dict, optional
            New exposure in microseconds — scalar for all, or ``{serial: us}``.
        """
        for camera in self.cameralist:
            g = gain.get(camera.name) if isinstance(gain, dict) else gain
            e = exposure.get(camera.name) if isinstance(exposure, dict) else exposure
            if g is not None or e is not None:
                camera.set_param(gain=g, exposure=e)

    def _broadcast(self, method_name):
        """Call ``method_name`` on every camera in parallel and wait for all.

        Spawns one thread per camera, each invoking the named zero-argument
        :class:`Camera` method, then joins every thread before returning.

        Parameters
        ----------
        method_name : str
            Name of the :class:`Camera` method to invoke on each camera.
        """
        threads = [Thread(target=getattr(camera, method_name)) for camera in self.cameralist]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

    def stop(self):
        """Stop capture on all cameras in parallel (broadcasts :meth:`Camera.stop`)."""
        self._broadcast("stop")

    def end(self):
        """Stop and release all cameras in parallel (broadcasts :meth:`Camera.end`)."""
        self._broadcast("end")

    def get_status_list(self):
        """Return a status snapshot for every camera.

        Returns
        -------
        list of dict
            One :meth:`Camera.get_status` dict per camera, in ``cameralist``
            order.
        """
        status_list = []
        for camera in self.cameralist:
            status_list.append(camera.get_status())
        return status_list

    def get_summary(self):
        """Return compact group-level camera state for daemon responses."""
        status_list = self.get_status_list()
        return {
            "expected_camera_count": self.expected_camera_count,
            "detected_camera_count": len(self.cameralist),
            "camera_names": list(self.camera_names),
            "frame_ids": {s["name"]: s["frame_id"] for s in status_list},
            "states": {s["name"]: s["state"] for s in status_list},
            "errors": {s["name"]: s.get("error") for s in status_list if s.get("error")},
        }
    
    def get_all_errors(self):
        """Return error information for every camera currently in an error state.

        Returns
        -------
        dict
            Maps camera serial (``str``) to an ``(error_msg, traceback_msg)``
            tuple, including only cameras whose :meth:`Camera.get_error` reports
            an active error. Empty when no camera is in error.
        """
        errors = {}
        for camera in self.cameralist:
            has_error, (error_msg, traceback_msg) = camera.get_error()
            if has_error:
                errors[camera.name] = (error_msg, traceback_msg)
        return errors
