"""Aravis timestamp-only receiver for the main-PC synchronization camera.

The camera still transmits GVSP image payloads, because frame metadata arrives
with an Aravis buffer, but pixels are never converted, copied, or saved.  Each
completed buffer contributes only its frame id and the PC receive time before
being returned immediately to the reusable stream pool.
"""

from __future__ import annotations

import time
from typing import Optional

from paradex.io.camera_system.aravis_gstreamer import (
    AravisGStreamerError,
    AravisGStreamerSettings,
    _load_aravis,
    _write_feature,
)
from paradex.utils.system import get_camera_config


class AravisTimestampMonitor:
    """Keep one trigger camera connected and expose PySpin-compatible methods."""

    def __init__(self, serial: str, buffer_count: int = 10) -> None:
        self.serial = str(serial)
        self.settings = AravisGStreamerSettings.from_environment()
        self._aravis = _load_aravis()
        self.device_id = self._resolve_device_id()
        self.camera = self._aravis.Camera.new(self.device_id)
        if self.camera is None:
            raise AravisGStreamerError(
                "Timestamp camera {} was not found using {!r}".format(
                    self.serial, self.device_id
                )
            )
        self.device = self.camera.get_device()
        self.stream = None
        self._acquiring = False
        self._configure()
        self._create_stream(buffer_count)

    def _resolve_device_id(self) -> str:
        self._aravis.update_device_list()
        for index in range(self._aravis.get_n_devices()):
            serial = self._aravis.get_device_serial_nbr(index)
            if serial is not None and str(serial) == self.serial:
                return str(self._aravis.get_device_id(index))
        raise AravisGStreamerError(
            "Timestamp camera serial {} was not discovered by Aravis".format(
                self.serial
            )
        )

    def _configure(self) -> None:
        values = get_camera_config().get(self.serial, {})
        gain = float(values.get("gain", 3.0))
        exposure = float(
            values.get("exposure", values.get("exposure_time", 2500.0))
        )

        _write_feature(self.device, "GevHeartbeatTimeout", self.settings.heartbeat_timeout_ms)
        _write_feature(self.device, "GevSCPSPacketSize", self.settings.packet_size)
        _write_feature(self.device, "PixelFormat", self.settings.pixel_format)
        _write_feature(self.device, "TriggerMode", "Off")
        _write_feature(self.device, "TriggerSelector", "FrameStart")
        _write_feature(self.device, "TriggerSource", self.settings.trigger_source)
        _write_feature(self.device, "TriggerActivation", self.settings.trigger_activation)
        _write_feature(self.device, "TriggerOverlap", self.settings.trigger_overlap)
        _write_feature(self.device, "ExposureAuto", "Off")
        _write_feature(self.device, "ExposureTime", exposure)
        _write_feature(self.device, "GainAuto", "Off")
        _write_feature(self.device, "Gain", gain)
        _write_feature(self.device, "TriggerMode", "On")
        self.camera.set_acquisition_mode(self._aravis.AcquisitionMode.CONTINUOUS)

    def _create_stream(self, buffer_count: int) -> None:
        self.stream = self.camera.create_stream(None, None)
        if self.stream is None:
            raise AravisGStreamerError(
                "Could not create timestamp stream for {}".format(self.serial)
            )
        payload = int(self.camera.get_payload())
        for _ in range(buffer_count):
            self.stream.push_buffer(self._aravis.Buffer.new_allocate(payload))

    def _drain(self) -> None:
        if self.stream is None:
            return
        while True:
            buffer = self.stream.try_pop_buffer()
            if buffer is None:
                return
            self.stream.push_buffer(buffer)

    def start(self) -> None:
        if self._acquiring:
            return
        self._drain()
        self.camera.start_acquisition()
        self._acquiring = True

    def get_timestamp(self, timeout_ms: int = 0) -> Optional[dict]:
        if not self._acquiring or self.stream is None:
            return None
        if timeout_ms > 0:
            buffer = self.stream.timeout_pop_buffer(int(timeout_ms * 1000))
        else:
            buffer = self.stream.pop_buffer()
        if buffer is None:
            return None
        try:
            if buffer.get_status() != self._aravis.BufferStatus.SUCCESS:
                return None
            return {"pc_time": time.time(), "frameID": int(buffer.get_frame_id())}
        finally:
            self.stream.push_buffer(buffer)

    def stop(self) -> None:
        if not self._acquiring:
            return
        self.camera.stop_acquisition()
        self._acquiring = False
        self._drain()

    def release(self) -> None:
        self.stop()
        self.stream = None
        self.device = None
        self.camera = None


def load_timestamp_monitor(serial: str) -> AravisTimestampMonitor:
    return AravisTimestampMonitor(serial)
