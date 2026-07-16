"""External-trigger Aravis + GStreamer camera backend for capture PCs.

This is intentionally an adapter for the existing ``CameraLoader`` contract:
the ZMQ command protocol and CaptureSession stay unchanged, while each camera
is prepared locally as::

    Aravis stream -> appsrc -> Bayer conversion -> JPEG -> AVI mux -> filesink

The only trigger owner remains the main PC's UTG900E.  A capture PC returns
from ``start`` only after its cameras were configured for ``Line0`` and their
GStreamer pipelines have been brought up; it never opens a USB trigger.
"""

from __future__ import annotations

import ipaddress
import logging
import os
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Tuple

from paradex.io.camera_system.aravis_addressing import CameraAddressing
from paradex.utils.path import capture_path_list, home_path
from paradex.utils.system import get_camera_config, get_camera_list


log = logging.getLogger(__name__)


class AravisGStreamerError(RuntimeError):
    """A capture PC could not prepare, run, or finalize a camera pipeline."""


def _env_int(name: str, default: int) -> int:
    value = os.getenv(name)
    return default if value is None else int(value)


def _env_float(name: str, default: float) -> float:
    value = os.getenv(name)
    return default if value is None else float(value)


@dataclass(frozen=True)
class AravisGStreamerSettings:
    width: int = 2048
    height: int = 1536
    bayer_format: str = "rggb"
    pixel_format: str = "BayerRG8"
    packet_size: int = 9000
    heartbeat_timeout_ms: int = 10000
    trigger_source: str = "Line0"
    trigger_activation: str = "RisingEdge"
    trigger_overlap: str = "ReadOut"
    jpeg_quality: int = 95
    queue_buffers: int = 60
    startup_timeout_seconds: float = 5.0
    first_frame_timeout_seconds: float = 10.0
    eos_timeout_seconds: float = 15.0
    aravis_buffer_count: int = 64
    stream_poll_timeout_us: int = 200000
    preview_width: int = 640
    preview_fps: int = 5
    preview_jpeg_quality: int = 70

    @classmethod
    def from_environment(cls) -> "AravisGStreamerSettings":
        return cls(
            width=_env_int("PARADEX_CAMERA_WIDTH", 2048),
            height=_env_int("PARADEX_CAMERA_HEIGHT", 1536),
            bayer_format=os.getenv("PARADEX_BAYER_FORMAT", "rggb"),
            pixel_format=os.getenv("PARADEX_PIXEL_FORMAT", "BayerRG8"),
            packet_size=_env_int("PARADEX_GIGE_PACKET_SIZE", 9000),
            heartbeat_timeout_ms=_env_int("PARADEX_GIGE_HEARTBEAT_MS", 10000),
            trigger_source=os.getenv("PARADEX_TRIGGER_SOURCE", "Line0"),
            trigger_activation=os.getenv("PARADEX_TRIGGER_ACTIVATION", "RisingEdge"),
            trigger_overlap=os.getenv("PARADEX_TRIGGER_OVERLAP", "ReadOut"),
            jpeg_quality=_env_int("PARADEX_JPEG_QUALITY", 95),
            queue_buffers=_env_int("PARADEX_GST_QUEUE_BUFFERS", 60),
            first_frame_timeout_seconds=_env_float("PARADEX_FIRST_FRAME_TIMEOUT", 10.0),
            aravis_buffer_count=_env_int("PARADEX_ARAVIS_BUFFERS", 64),
            stream_poll_timeout_us=_env_int("PARADEX_STREAM_POLL_TIMEOUT_US", 200000),
            preview_width=_env_int("PARADEX_PREVIEW_WIDTH", 640),
            preview_fps=_env_int("PARADEX_PREVIEW_FPS", 5),
            preview_jpeg_quality=_env_int("PARADEX_PREVIEW_JPEG_QUALITY", 70),
        )


def trigger_features(settings: AravisGStreamerSettings, sync_mode: bool) -> str:
    """Return aravissrc's post-caps feature string.

    aravissrc can switch a camera back to free-run while it negotiates caps.
    Applying these values through its ``features`` property happens after
    that transition and therefore keeps the external hardware trigger armed.
    """

    if not sync_mode:
        return "TriggerMode=Off"
    return " ".join(
        (
            "TriggerSelector=FrameStart",
            "TriggerSource={}".format(settings.trigger_source),
            "TriggerActivation={}".format(settings.trigger_activation),
            "TriggerOverlap={}".format(settings.trigger_overlap),
            "TriggerMode=On",
        )
    )


def camera_caps(settings: AravisGStreamerSettings, fps: int, sync_mode: bool) -> str:
    """Build caps without invoking Aravis free-run rate control in HT mode.

    On the deployed Aravis 0.8.20 plugin, a caps framerate makes aravissrc
    call ``arv_camera_set_frame_rate()``.  That path also calls the
    USB3Vision-only ``arv_camera_uv_set_usb_mode()`` even for GigE cameras,
    producing a critical assertion and making each state transition slow.
    UTG supplies the cadence in sync mode, so no framerate cap is required.
    """

    base = "video/x-bayer,format={},width={},height={}".format(
        settings.bayer_format,
        settings.width,
        settings.height,
    )
    if sync_mode:
        # Unlike the display-only Aravis viewer, Paradex terminates in
        # avimux, which needs a concrete rate for stream negotiation and AVI
        # headers. This is appsrc caps only; it does not program the camera.
        return "{},framerate={}/1".format(base, fps)
    return "{},framerate={}/1".format(base, fps)


def _load_gst():
    try:
        import gi

        gi.require_version("Gst", "1.0")
        from gi.repository import Gst
    except (ImportError, ValueError) as exc:
        raise AravisGStreamerError(
            "GStreamer GI bindings are unavailable. Install python3-gi, "
            "gir1.2-gstreamer-1.0, gstreamer plugins, and gir1.2-aravis-0.8."
        ) from exc
    Gst.init(None)
    return Gst


def _load_aravis():
    try:
        import gi

        gi.require_version("Aravis", "0.8")
        from gi.repository import Aravis
    except (ImportError, ValueError) as exc:
        raise AravisGStreamerError(
            "Aravis GI bindings are unavailable. Install gir1.2-aravis-0.8."
        ) from exc
    return Aravis


def _feature_equal(actual, expected) -> bool:
    if isinstance(expected, float):
        return abs(float(actual) - expected) <= max(0.1, abs(expected) * 0.01)
    return actual == expected


def _write_feature(device, name: str, value) -> None:
    """Set a GenICam feature and fail before recording if readback differs."""

    if isinstance(value, bool):
        kind = "boolean"
    elif isinstance(value, int):
        kind = "integer"
    elif isinstance(value, float):
        kind = "float"
    elif isinstance(value, str):
        kind = "string"
    else:
        raise TypeError("Unsupported GenICam value for {}: {!r}".format(name, value))

    getattr(device, "set_{}_feature_value".format(kind))(name, value)
    actual = getattr(device, "get_{}_feature_value".format(kind))(name)
    if not _feature_equal(actual, value):
        raise AravisGStreamerError(
            "Camera feature {} did not persist: requested {!r}, got {!r}".format(name, value, actual)
        )


def _write_optional_feature(device, names: Iterable[str], value) -> Optional[str]:
    """Write a firmware-specific GenICam alias, or skip if none exist."""

    attempted = []
    for name in names:
        attempted.append(name)
        try:
            _write_feature(device, name, value)
            return name
        except Exception as exc:
            if "node '{}' not found".format(name) not in str(exc):
                raise
    log.warning("Optional GenICam nodes not found: %s; skipping", attempted)
    return None


class AravisGStreamerCamera:
    """One FLIR GigE camera with a native GStreamer recording pipeline."""

    def __init__(
        self,
        serial: str,
        settings: Optional[AravisGStreamerSettings] = None,
        camera_config: Optional[Dict[str, dict]] = None,
        device_id: Optional[str] = None,
    ) -> None:
        self.name = str(serial)
        self.device_id = str(device_id or "FLIR-{}".format(self.name))
        self.settings = settings or AravisGStreamerSettings.from_environment()
        self.camera_config = camera_config if camera_config is not None else get_camera_config()
        self._lock = threading.RLock()
        self._pipeline = None
        self._gst = None
        self._appsrc = None
        self._aravis = None
        self._aravis_camera = None
        self._aravis_stream = None
        self._stream_thread = None
        self._stream_stop = threading.Event()
        self._first_timestamp_ns = None
        self._state = "READY"
        self._last_error: Optional[str] = None
        self._last_traceback: Optional[str] = None
        self._mode: Optional[str] = None
        self._fps: Optional[int] = None
        self._sync_mode: Optional[bool] = None
        self._save_path: Optional[str] = None
        self._started_at: Optional[float] = None
        self._frame_count = 0
        self._frame_count_lock = threading.Lock()
        self._preview_lock = threading.Lock()
        self._latest_preview: Optional[bytes] = None

    def _camera_values(self, fps: int) -> Tuple[float, float]:
        values = self.camera_config.get(self.name, {})
        gain = float(values.get("gain", 1.0))
        exposure = float(values.get("exposure", values.get("exposure_time", 2200.0)))
        if fps <= 0:
            raise AravisGStreamerError("FPS must be positive for {}".format(self.name))
        return gain, exposure

    def _configure_camera(self, fps: int, sync_mode: bool):
        Aravis = _load_aravis()
        camera = Aravis.Camera.new(self.device_id)
        if camera is None:
            raise AravisGStreamerError(
                "Camera {} was not found by Aravis using device id {!r}".format(
                    self.name, self.device_id
                )
            )
        device = camera.get_device()
        gain, exposure = self._camera_values(fps)
        try:
            _write_feature(device, "GevHeartbeatTimeout", self.settings.heartbeat_timeout_ms)
            _write_feature(device, "PixelFormat", self.settings.pixel_format)
            _write_feature(device, "GevSCPSPacketSize", self.settings.packet_size)

            _write_feature(device, "TriggerMode", "Off")
            if sync_mode:
                _write_feature(device, "TriggerSelector", "FrameStart")
                _write_feature(device, "TriggerSource", self.settings.trigger_source)
                _write_feature(device, "TriggerActivation", self.settings.trigger_activation)
                _write_feature(device, "TriggerOverlap", self.settings.trigger_overlap)

            # ParaOffice programs the AFR cap before arming the trigger.  Some
            # deployed BFS XMLs expose older aliases (or no enable node), so
            # preserve that order while accepting the firmware variants.
            _write_optional_feature(device, ("AcquisitionFrameRateAuto",), "Off")
            _write_optional_feature(
                device,
                ("AcquisitionFrameRateEnable", "AcquisitionFrameRateEnabled"),
                True,
            )
            _write_optional_feature(
                device,
                ("AcquisitionFrameRate", "AcquisitionFrameRateAbs"),
                float(fps),
            )
            _write_feature(device, "ExposureAuto", "Off")
            _write_feature(device, "ExposureTime", exposure)
            _write_feature(device, "GainAuto", "Off")
            _write_feature(device, "Gain", gain)

            if sync_mode:
                _write_feature(device, "TriggerMode", "On")
            return Aravis, camera
        except Exception:
            del device
            del camera
            raise

    @staticmethod
    def _make(Gst, factory: str, name: str):
        element = Gst.ElementFactory.make(factory, name)
        if element is None:
            raise AravisGStreamerError(
                "GStreamer element {!r} is unavailable; install the required plugins".format(factory)
            )
        return element

    def _build_pipeline(self, save_path: Optional[str], fps: int, sync_mode: bool):
        Gst = _load_gst()
        settings = self.settings
        pipeline = Gst.Pipeline.new("camera_{}".format(self.name))
        if pipeline is None:
            raise AravisGStreamerError("Could not create GStreamer pipeline for {}".format(self.name))

        # Use Aravis' stream API directly and feed GStreamer through appsrc.
        # aravissrc treats a 2-second no-frame interval as fatal and also
        # executes a USB3Vision-only setup path on these GigE cameras.  The
        # direct stream loop can wait indefinitely for the first UTG pulse,
        # preserving the original Paradex arm-all-then-trigger contract.
        source = self._make(Gst, "appsrc", "src_{}".format(self.name))
        source.set_property("is-live", True)
        source.set_property("block", True)
        source.set_property("format", Gst.Format.TIME)
        source.set_property("do-timestamp", True)

        capsfilter = self._make(Gst, "capsfilter", "caps_{}".format(self.name))
        capsfilter.set_property(
            "caps",
            Gst.Caps.from_string(camera_caps(settings, fps, sync_mode)),
        )
        source.set_property("caps", Gst.Caps.from_string(camera_caps(settings, fps, sync_mode)))
        raw_queue = self._make(Gst, "queue", "raw_queue_{}".format(self.name))
        raw_queue.set_property("max-size-buffers", settings.queue_buffers)
        raw_queue.set_property("max-size-bytes", 0)
        raw_queue.set_property("max-size-time", 0)
        bayer = self._make(Gst, "bayer2rgb", "bayer_{}".format(self.name))
        convert = self._make(Gst, "videoconvert", "convert_{}".format(self.name))
        tee = self._make(Gst, "tee", "tee_{}".format(self.name))
        record_queue = self._make(Gst, "queue", "record_queue_{}".format(self.name))
        record_queue.set_property("max-size-buffers", settings.queue_buffers)
        record_queue.set_property("max-size-bytes", 0)
        record_queue.set_property("max-size-time", 0)
        jpegenc = self._make(Gst, "jpegenc", "jpeg_{}".format(self.name))
        jpegenc.set_property("quality", settings.jpeg_quality)
        encoded_pad = jpegenc.get_static_pad("src")
        if encoded_pad is None:
            raise AravisGStreamerError(
                "Could not access encoded frame-count pad for {}".format(self.name)
            )

        def count_frame(_pad, _info):
            with self._frame_count_lock:
                self._frame_count += 1
            return Gst.PadProbeReturn.OK

        encoded_pad.add_probe(Gst.PadProbeType.BUFFER, count_frame)

        preview_queue = self._make(Gst, "queue", "preview_queue_{}".format(self.name))
        preview_queue.set_property("leaky", 2)
        preview_queue.set_property("max-size-buffers", 1)
        preview_queue.set_property("max-size-bytes", 0)
        preview_queue.set_property("max-size-time", 0)
        preview_rate = self._make(Gst, "videorate", "preview_rate_{}".format(self.name))
        preview_rate.set_property("drop-only", True)
        preview_scale = self._make(Gst, "videoscale", "preview_scale_{}".format(self.name))
        preview_caps = self._make(Gst, "capsfilter", "preview_caps_{}".format(self.name))
        preview_height = max(
            2,
            int(round(settings.preview_width * settings.height / settings.width / 2.0) * 2),
        )
        preview_caps.set_property(
            "caps",
            Gst.Caps.from_string(
                "video/x-raw,width={},height={},framerate={}/1".format(
                    settings.preview_width,
                    preview_height,
                    settings.preview_fps,
                )
            ),
        )
        preview_convert = self._make(
            Gst, "videoconvert", "preview_convert_{}".format(self.name)
        )
        preview_jpeg = self._make(Gst, "jpegenc", "preview_jpeg_{}".format(self.name))
        preview_jpeg.set_property("quality", settings.preview_jpeg_quality)
        preview_sink = self._make(Gst, "appsink", "preview_sink_{}".format(self.name))
        preview_sink.set_property("emit-signals", True)
        preview_sink.set_property("drop", True)
        preview_sink.set_property("max-buffers", 1)
        preview_sink.set_property("sync", False)

        def update_preview(sink):
            try:
                sample = sink.emit("pull-sample")
                if sample is None:
                    return Gst.FlowReturn.OK
                buffer = sample.get_buffer()
                if buffer is None:
                    return Gst.FlowReturn.OK
                jpeg = buffer.extract_dup(0, buffer.get_size())
                with self._preview_lock:
                    self._latest_preview = bytes(jpeg)
            except Exception:
                log.exception("Preview callback failed for camera %s", self.name)
            return Gst.FlowReturn.OK

        preview_sink.connect("new-sample", update_preview)

        common = [source, capsfilter, raw_queue, bayer, convert, tee]
        record = [record_queue, jpegenc]
        if save_path is not None:
            mux_queue = self._make(Gst, "queue", "mux_queue_{}".format(self.name))
            mux_queue.set_property("max-size-buffers", settings.queue_buffers)
            mux_queue.set_property("max-size-bytes", 0)
            mux_queue.set_property("max-size-time", 0)
            mux = self._make(Gst, "avimux", "mux_{}".format(self.name))
            sink = self._make(Gst, "filesink", "sink_{}".format(self.name))
            sink.set_property("location", save_path)
            sink.set_property("sync", False)
            sink.set_property("async", False)
            record.extend([mux_queue, mux, sink])
        else:
            sink = self._make(Gst, "fakesink", "sink_{}".format(self.name))
            sink.set_property("sync", False)
            record.append(sink)

        preview = [
            preview_queue,
            preview_rate,
            preview_scale,
            preview_caps,
            preview_convert,
            preview_jpeg,
            preview_sink,
        ]
        elements = common + record + preview

        for element in elements:
            pipeline.add(element)

        def link_chain(chain):
            for left, right in zip(chain, chain[1:]):
                if left.link(right):
                    continue
                raise AravisGStreamerError(
                    "Could not link {} -> {} for camera {}".format(
                        left.get_name(), right.get_name(), self.name
                    )
                )

        link_chain(common)
        link_chain([tee] + record)
        link_chain([tee] + preview)
        self._gst = Gst
        self._appsrc = source
        return pipeline

    def get_preview(self) -> Optional[bytes]:
        """Return the newest low-bandwidth JPEG without blocking recording."""

        with self._preview_lock:
            return self._latest_preview

    def _create_aravis_stream(self) -> None:
        assert self._aravis is not None
        assert self._aravis_camera is not None
        stream = self._aravis_camera.create_stream(None, None)
        if stream is None:
            raise AravisGStreamerError(
                "Aravis could not create a stream for camera {}".format(self.name)
            )
        payload = int(self._aravis_camera.get_payload())
        if payload <= 0:
            raise AravisGStreamerError(
                "Camera {} reported invalid payload {}".format(self.name, payload)
            )
        for _ in range(self.settings.aravis_buffer_count):
            stream.push_buffer(self._aravis.Buffer.new_allocate(payload))
        self._aravis_camera.set_acquisition_mode(
            self._aravis.AcquisitionMode.CONTINUOUS
        )
        self._aravis_stream = stream

    def prepare_hardware(self, fps: int = 30, sync_mode: bool = True) -> None:
        """Configure and retain the camera/stream for the daemon lifetime."""

        with self._lock:
            if self._aravis_camera is not None and self._aravis_stream is not None:
                return
            self._state = "PREPARING_HARDWARE"
            try:
                self._aravis, self._aravis_camera = self._configure_camera(
                    fps, sync_mode
                )
                self._create_aravis_stream()
                self._fps = fps
                self._sync_mode = sync_mode
                self._state = "PREPARED"
            except Exception as exc:
                self._last_error = str(exc)
                self._state = "ERROR"
                self._teardown_pipeline(release_hardware=True)
                raise

    def _stream_frames(self) -> None:
        assert self._aravis_stream is not None
        assert self._appsrc is not None
        assert self._gst is not None
        try:
            while not self._stream_stop.is_set():
                arv_buffer = self._aravis_stream.timeout_pop_buffer(
                    self.settings.stream_poll_timeout_us
                )
                if arv_buffer is None:
                    # No external trigger yet is normal in Paradex.
                    continue
                try:
                    if arv_buffer.get_status() != self._aravis.BufferStatus.SUCCESS:
                        continue
                    data = bytes(arv_buffer.get_data())
                    gst_buffer = self._gst.Buffer.new_allocate(None, len(data), None)
                    gst_buffer.fill(0, data)
                    # Match Aravis' official viewer: appsrc timestamps buffers
                    # against the running GStreamer clock. Camera timestamps
                    # are not assumed to share an epoch across devices.
                    result = self._appsrc.emit("push-buffer", gst_buffer)
                    if result != self._gst.FlowReturn.OK:
                        if not self._stream_stop.is_set():
                            raise AravisGStreamerError(
                                "GStreamer appsrc rejected frame for {}: {}".format(
                                    self.name, result
                                )
                            )
                        return
                finally:
                    self._aravis_stream.push_buffer(arv_buffer)
        except Exception as exc:
            self._last_error = str(exc)
            self._state = "ERROR"
            log.exception("Camera %s stream loop failed", self.name)

    def _drain_aravis_output(self) -> None:
        """Return completed buffers left over from the previous session."""

        if self._aravis_stream is None:
            return
        try_pop = getattr(self._aravis_stream, "try_pop_buffer", None)
        if try_pop is None:
            return
        while True:
            buffer = try_pop()
            if buffer is None:
                return
            self._aravis_stream.push_buffer(buffer)

    def _raise_if_bus_error(self, timeout_seconds: float = 0.0) -> None:
        if self._pipeline is None or self._gst is None:
            return
        message = self._pipeline.get_bus().timed_pop_filtered(
            int(timeout_seconds * self._gst.SECOND), self._gst.MessageType.ERROR
        )
        if message is None:
            return
        error, debug = message.parse_error()
        raise AravisGStreamerError("GStreamer error for {}: {} ({})".format(self.name, error, debug))

    def _prepare_playing_state(self) -> None:
        assert self._pipeline is not None
        assert self._gst is not None
        result = self._pipeline.set_state(self._gst.State.PLAYING)
        if result == self._gst.StateChangeReturn.FAILURE:
            self._raise_if_bus_error(self.settings.startup_timeout_seconds)
            raise AravisGStreamerError("GStreamer could not start camera {}".format(self.name))

        # A hardware-triggered appsrc pipeline legitimately has no preroll
        # before the UTG emits its first pulse.
        self._raise_if_bus_error(0.05)

    def prepare(self, mode: str, sync_mode: bool, save_path: Optional[str], fps: int) -> None:
        """Configure the camera and build its appsrc pipeline.

        Aravis' process-global discovery/device machinery is not exercised
        concurrently here. This mirrors ParaOffice's sequential camera
        configure/build phase and avoids racing Camera.new() across cameras.
        """

        if mode != "video":
            raise AravisGStreamerError(
                "Aravis/GStreamer backend is the CaptureSession video backend; "
                "use the pyspin agent for {!r} mode".format(mode)
            )
        if not save_path:
            raise AravisGStreamerError("A save path is required for video mode")

        with self._lock:
            if self._pipeline is not None:
                raise AravisGStreamerError("Camera {} is already recording".format(self.name))
            self._state = "STARTING"
            self._last_error = None
            self._last_traceback = None
            self._mode = mode
            self._fps = fps
            self._sync_mode = sync_mode
            self._save_path = save_path
            with self._frame_count_lock:
                self._frame_count = 0
            with self._preview_lock:
                self._latest_preview = None
            try:
                self.prepare_hardware(fps, sync_mode)
                self._pipeline = self._build_pipeline(save_path, fps, sync_mode)
                self._state = "PREPARED"
            except Exception as exc:
                self._last_error = str(exc)
                self._state = "ERROR"
                self._teardown_pipeline(release_hardware=False)
                raise

    def start_prepared(self) -> None:
        """Open a pipeline that was built by :meth:`prepare`."""

        with self._lock:
            if self._pipeline is None or self._state != "PREPARED":
                raise AravisGStreamerError("Camera {} is not prepared".format(self.name))
            try:
                self._prepare_playing_state()
                self._started_at = time.time()
                self._state = "CAPTURING"
            except Exception as exc:
                self._last_error = str(exc)
                self._state = "ERROR"
                self._teardown_pipeline()
                raise

    def request_playing(self) -> None:
        """Request PLAYING without serializing camera state transitions."""

        with self._lock:
            if self._pipeline is None or self._state != "PREPARED":
                raise AravisGStreamerError("Camera {} is not prepared".format(self.name))
            assert self._gst is not None
            result = self._pipeline.set_state(self._gst.State.PLAYING)
            if result == self._gst.StateChangeReturn.FAILURE:
                self._raise_if_bus_error(0.0)
                raise AravisGStreamerError(
                    "GStreamer could not start camera {}".format(self.name)
                )

    def confirm_playing(self) -> None:
        with self._lock:
            self._raise_if_bus_error(0.05)
            assert self._aravis_camera is not None
            self._stream_stop.clear()
            self._first_timestamp_ns = None
            self._drain_aravis_output()
            self._aravis_camera.start_acquisition()
            self._stream_thread = threading.Thread(
                target=self._stream_frames,
                name="aravis-stream-{}".format(self.name),
                daemon=True,
            )
            self._stream_thread.start()
            self._started_at = time.time()
            self._state = "CAPTURING"

    def start(self, mode: str, sync_mode: bool, save_path: Optional[str], fps: int) -> None:
        """Compatibility entry point for starting one camera directly."""

        self.prepare(mode, sync_mode, save_path, fps)
        self.start_prepared()

    def abort(self) -> None:
        """Immediately release a partially prepared/started pipeline."""

        with self._lock:
            self._teardown_pipeline()
            if self._state != "ERROR":
                self._state = "PREPARED"

    def wait_for_first_frame(self, timeout_seconds: Optional[float] = None) -> None:
        """Fail START unless a hardware-triggered frame actually arrives."""

        timeout = (
            self.settings.first_frame_timeout_seconds
            if timeout_seconds is None
            else timeout_seconds
        )
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            self._raise_if_bus_error(0.0)
            with self._frame_count_lock:
                if self._frame_count > 0:
                    return
            time.sleep(0.01)
        self._raise_if_bus_error(0.0)
        raise AravisGStreamerError(
            "Camera {} received no frames within {:.1f}s after UTG start".format(
                self.name, timeout
            )
        )

    def _teardown_pipeline(self, release_hardware: bool = False) -> None:
        self._stream_stop.set()
        if self._aravis_camera is not None:
            try:
                self._aravis_camera.stop_acquisition()
            except Exception:
                pass
        if self._stream_thread is not None:
            self._stream_thread.join(timeout=2.0)
        self._stream_thread = None
        self._drain_aravis_output()
        if self._pipeline is not None and self._gst is not None:
            self._pipeline.set_state(self._gst.State.NULL)
        self._pipeline = None
        self._appsrc = None
        if release_hardware:
            self._aravis_stream = None
            self._aravis_camera = None
            self._aravis = None

    def stop(self) -> None:
        with self._lock:
            if self._pipeline is None:
                if self._state != "ERROR":
                    self._state = (
                        "PREPARED" if self._aravis_stream is not None else "READY"
                    )
                return
            if self._state == "ERROR":
                # The bus/stream error is already recorded and reported by
                # validate/heartbeat. Do not obscure it with a second EOS
                # finalization failure during rollback.
                self._teardown_pipeline(release_hardware=False)
                return
            self._state = "STOPPING"
            try:
                # CaptureSession sends this while UTG pulses still exist.
                # Pulses remain active until AVI EOS/finalization completes.
                self._stream_stop.set()
                if self._aravis_camera is not None:
                    self._aravis_camera.stop_acquisition()
                if self._stream_thread is not None:
                    self._stream_thread.join(timeout=2.0)
                self._appsrc.emit("end-of-stream")
                message = self._pipeline.get_bus().timed_pop_filtered(
                    int(self.settings.eos_timeout_seconds * self._gst.SECOND),
                    self._gst.MessageType.EOS | self._gst.MessageType.ERROR,
                )
                if message is None:
                    raise AravisGStreamerError("Timed out finalizing AVI for {}".format(self.name))
                if message.type == self._gst.MessageType.ERROR:
                    error, debug = message.parse_error()
                    raise AravisGStreamerError(
                        "GStreamer failed while finalizing {}: {} ({})".format(self.name, error, debug)
                    )
                self._state = "PREPARED"
            except Exception as exc:
                self._last_error = str(exc)
                self._state = "ERROR"
                raise
            finally:
                self._teardown_pipeline(release_hardware=False)

    def end(self) -> None:
        try:
            self.stop()
        finally:
            self._teardown_pipeline(release_hardware=True)
            self._state = "STOPPED"

    def get_error(self) -> Tuple[bool, Tuple[Optional[str], Optional[str]]]:
        return self._state == "ERROR", (self._last_error, self._last_traceback)

    def get_status(self) -> dict:
        with self._frame_count_lock:
            frame_count = self._frame_count
        return {
            "state": self._state,
            "frame_id": frame_count,
            "frame_count": frame_count,
            "name": self.name,
            "mode": self._mode,
            "fps": self._fps,
            "syncMode": self._sync_mode,
            "save_path": self._save_path,
            "time": time.time(),
        }


class AravisGStreamerCameraLoader:
    """Drop-in ``CameraLoader`` replacement used by the ZMQ server daemon."""

    def __init__(
        self,
        serial_list: Optional[Iterable[str]] = None,
        settings: Optional[AravisGStreamerSettings] = None,
        addressing: Optional[CameraAddressing] = None,
        camera_factory: Optional[Callable[[str], AravisGStreamerCamera]] = None,
        reconcile_addresses: bool = True,
        prewarm_hardware: bool = True,
    ) -> None:
        self.settings = settings or AravisGStreamerSettings.from_environment()
        self.camera_names = [str(serial) for serial in (serial_list or get_camera_list())]
        if not self.camera_names:
            raise AravisGStreamerError("No camera serials are configured for this capture PC")
        self._configured_camera_count = len(self.camera_names)

        self.camera_inventory: List[dict] = []
        self.device_ids: Dict[str, str] = {}
        if reconcile_addresses:
            address_manager = addressing or CameraAddressing()
            self.camera_names = address_manager.reconcile(
                self.camera_names,
                allow_partial=True,
            )
            for serial in self.camera_names:
                record = address_manager.seen[serial]
                nic_name = "unknown"
                try:
                    camera_ip = ipaddress.IPv4Address(record.ip)
                    nic_name = next(
                        nic.name
                        for nic in address_manager.nic_subnets
                        if camera_ip in nic.network
                    )
                except (ValueError, StopIteration):
                    pass
                self.camera_inventory.append(
                    {
                        "serial": record.serial,
                        "device_id": record.device_id,
                        "ip": record.ip,
                        "mac": record.mac,
                        "nic": nic_name,
                    }
                )
                self.device_ids[serial] = record.device_id

        if camera_factory is None:
            camera_factory = lambda serial: AravisGStreamerCamera(
                serial,
                self.settings,
                device_id=self.device_ids.get(serial),
            )
        self.cameralist = [camera_factory(serial) for serial in self.camera_names]
        self._capture_active = False
        # Match the legacy PySpin CameraLoader: camera connections and receive
        # buffers are established once at daemon boot, not once per capture.
        if prewarm_hardware:
            for camera in self.cameralist:
                prepare_hardware = getattr(camera, "prepare_hardware", None)
                if prepare_hardware is not None:
                    prepare_hardware(fps=30, sync_mode=True)
        self._print_loaded_cameras()

    def _print_loaded_cameras(self) -> None:
        print(
            "[Info] Aravis/GStreamer cameras loaded ({}/{} configured):".format(
                len(self.camera_inventory) or len(self.camera_names),
                self._configured_camera_count,
            )
        )
        if self.camera_inventory:
            for camera in self.camera_inventory:
                print(
                    "  - serial={serial} ip={ip} nic={nic} mac={mac} device={device_id}".format(
                        **camera
                    )
                )
        else:
            for serial in self.camera_names:
                print("  - serial={}".format(serial))

    def _save_paths(self, mode: str, save_path: Optional[str]) -> List[Optional[str]]:
        if mode == "video":
            if save_path is None:
                raise AravisGStreamerError("save_path is required for video recording")
            paths: List[Optional[str]] = []
            for index, camera in enumerate(self.cameralist):
                directory = Path(capture_path_list[index % len(capture_path_list)]) / save_path / "videos"
                directory.mkdir(parents=True, exist_ok=True)
                paths.append(str(directory / "{}.avi".format(camera.name)))
            return paths
        raise AravisGStreamerError(
            "Aravis/GStreamer backend is limited to CaptureSession video mode; "
            "use the pyspin agent for {!r} mode".format(mode)
        )

    def _parallel(self, action: Callable[[AravisGStreamerCamera, Optional[str]], None], paths: List[Optional[str]]) -> None:
        errors: List[Tuple[str, BaseException]] = []
        lock = threading.Lock()

        def run(camera: AravisGStreamerCamera, path: Optional[str]) -> None:
            try:
                action(camera, path)
            except BaseException as exc:
                with lock:
                    errors.append((camera.name, exc))

        threads = [threading.Thread(target=run, args=(camera, path), daemon=False) for camera, path in zip(self.cameralist, paths)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        if errors:
            messages = ["{}: {}".format(name, error) for name, error in errors]
            raise AravisGStreamerError("; ".join(messages))

    def prepare(self, mode: str, syncMode: bool, save_path: Optional[str] = None, fps: int = 30) -> None:
        paths = self._save_paths(mode, save_path)
        print("[Info] Preparing Aravis/GStreamer cameras: {}".format(self.camera_names))
        try:
            # Configure/build sequentially using the exact device ids that
            # address reconciliation discovered and verified at agent boot.
            # Older Aravis releases do not reliably resolve the shorter
            # ``FLIR-{serial}`` alias used by newer ParaOffice installations.
            for camera, path in zip(self.cameralist, paths):
                camera.prepare(mode, syncMode, path, fps)
        except Exception:
            # A pipeline that has not reached PLAYING must not wait for EOS.
            # Hard-null every partial pipeline before returning the start error.
            for camera in self.cameralist:
                try:
                    camera.abort()
                except Exception:
                    log.exception("Failed to roll back camera %s", camera.name)
            raise
        print("[Info] All Aravis/GStreamer cameras PREPARED.")

    def activate(self) -> None:
        """Open every prepared source immediately before the main-PC UTG starts."""

        try:
            # Request every local pipeline concurrently to minimize skew
            # between camera acquisition start times.
            self._parallel(
                lambda camera, _path: camera.request_playing(),
                [None] * len(self.cameralist),
            )
            for camera in self.cameralist:
                camera.confirm_playing()
            self._capture_active = True
        except Exception:
            for camera in self.cameralist:
                try:
                    camera.abort()
                except Exception:
                    log.exception("Failed to roll back camera %s", camera.name)
            raise
        print("[Info] All Aravis/GStreamer cameras ARMED and waiting for UTG.")
        log.info("All Aravis/GStreamer cameras ARMED: %s", self.camera_names)

    def start(self, mode: str, syncMode: bool, save_path: Optional[str] = None, fps: int = 30) -> None:
        """Compatibility path; distributed callers use prepare then activate."""

        self.prepare(mode, syncMode, save_path, fps)
        self.activate()

    def wait_for_first_frames(self, timeout_seconds: Optional[float] = None) -> None:
        if not self._capture_active:
            raise AravisGStreamerError("Cannot validate frames before capture start")
        self._parallel(
            lambda camera, _path: camera.wait_for_first_frame(timeout_seconds),
            [None] * len(self.cameralist),
        )
        print("[Info] First hardware-triggered frame received from every camera.")

    def abort(self) -> None:
        for camera in self.cameralist:
            try:
                camera.abort()
            except Exception:
                log.exception("Failed to abort camera %s", camera.name)
        self._capture_active = False

    def _print_frame_counts(self) -> None:
        counts = []
        print("[Info] Captured frame counts:")
        for camera in self.cameralist:
            status = camera.get_status()
            count = int(status.get("frame_count", status.get("frame_id") or 0))
            counts.append(count)
            print("  - serial={}: {} frames".format(camera.name, count))
        print("[Info] Total captured frames: {}".format(sum(counts)))

    def stop(self) -> None:
        was_active = self._capture_active
        try:
            self._parallel(lambda camera, _path: camera.stop(), [None] * len(self.cameralist))
        finally:
            if was_active:
                self._capture_active = False
                self._print_frame_counts()

    def end(self) -> None:
        was_active = self._capture_active
        try:
            self._parallel(lambda camera, _path: camera.end(), [None] * len(self.cameralist))
        finally:
            if was_active:
                self._capture_active = False
                self._print_frame_counts()

    def get_status_list(self) -> List[dict]:
        return [camera.get_status() for camera in self.cameralist]

    def get_preview(self, serial: str) -> Optional[bytes]:
        for camera in self.cameralist:
            if camera.name == str(serial):
                return camera.get_preview()
        return None

    def get_all_errors(self) -> Dict[str, Tuple[Optional[str], Optional[str]]]:
        errors = {}
        for camera in self.cameralist:
            has_error, error = camera.get_error()
            if has_error:
                errors[camera.name] = error
        return errors
