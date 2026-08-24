"""Hardware-triggered two-camera acquisition through Paradex capture PCs."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import threading
import time
from typing import Iterable
from urllib.request import Request, urlopen

import cv2
import numpy as np

from paradex.inference.act_xarm_allegro.core import CameraBinding
from paradex.utils.system import get_pc_ip, network_info


@dataclass(frozen=True)
class EncodedFrame:
    frame_id: int
    received_monotonic_ns: int
    jpeg: bytes


@dataclass(frozen=True)
class CameraPair:
    images: dict[str, np.ndarray]
    frame_ids: dict[str, int]
    jpeg_bytes: dict[str, bytes]
    received_monotonic_ns: int


class CameraStreamError(RuntimeError):
    pass


class HttpFrameReader:
    """Continuously retain recent full-resolution frames from one daemon."""

    def __init__(
        self,
        binding: CameraBinding,
        *,
        preview_port: int = 5484,
        request_timeout_seconds: float = 1.0,
        buffer_size: int = 16,
    ):
        self.binding = binding
        self.url = (
            f"http://{get_pc_ip(binding.capture_pc)}:{preview_port}"
            f"/frame/{binding.physical_serial}"
        )
        self.request_timeout_seconds = float(request_timeout_seconds)
        self._frames: deque[EncodedFrame] = deque(maxlen=int(buffer_size))
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._last_error: str | None = None

    def start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(
            target=self._run,
            name=f"act-camera-{self.binding.physical_serial}",
            daemon=True,
        )
        self._thread.start()

    def _run(self) -> None:
        last_id = -1
        while not self._stop.is_set():
            try:
                request = Request(self.url, headers={"Cache-Control": "no-cache"})
                with urlopen(request, timeout=self.request_timeout_seconds) as response:
                    frame_id_header = response.headers.get("X-Frame-Id")
                    if frame_id_header is None:
                        raise CameraStreamError(f"{self.url} omitted X-Frame-Id")
                    frame_id = int(frame_id_header)
                    jpeg = response.read()
                if frame_id <= 0 or not jpeg:
                    raise CameraStreamError(f"{self.url} returned an empty frame")
                if frame_id != last_id:
                    frame = EncodedFrame(frame_id, time.monotonic_ns(), jpeg)
                    with self._lock:
                        self._frames.append(frame)
                    last_id = frame_id
                self._last_error = None
            except Exception as exc:
                self._last_error = f"{type(exc).__name__}: {exc}"
                self._stop.wait(0.02)

    def snapshot(self) -> dict[int, EncodedFrame]:
        with self._lock:
            return {frame.frame_id: frame for frame in self._frames}

    @property
    def last_error(self) -> str | None:
        return self._last_error

    def close(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=self.request_timeout_seconds + 1.0)
            self._thread = None


class SynchronizedCameraStream:
    """Own the Paradex arm-all-then-trigger session and return matched frames."""

    def __init__(
        self,
        bindings: Iterable[CameraBinding],
        *,
        fps: float = 30.0,
        frame_timeout_seconds: float = 3.0,
        max_frame_age_ms: float = 100.0,
        manage_capture_session: bool = True,
    ):
        self.bindings = tuple(bindings)
        if len(self.bindings) != 2:
            raise ValueError("ACT runner requires exactly two camera bindings")
        if len({item.policy_key for item in self.bindings}) != len(self.bindings):
            raise ValueError("camera policy keys must be unique")
        if len({item.physical_serial for item in self.bindings}) != len(self.bindings):
            raise ValueError("physical camera serials must be unique")
        for binding in self.bindings:
            capture_pc = network_info.get(binding.capture_pc, {})
            configured_serials = tuple(str(value) for value in capture_pc.get("cam_list", ()))
            if configured_serials and binding.physical_serial not in configured_serials:
                raise ValueError(
                    f"camera {binding.physical_serial} is not configured on {binding.capture_pc}"
                )
        self.fps = float(fps)
        self.frame_timeout_seconds = float(frame_timeout_seconds)
        self.max_frame_age_ms = float(max_frame_age_ms)
        self.manage_capture_session = bool(manage_capture_session)
        self.readers = [HttpFrameReader(binding) for binding in self.bindings]
        self._remote = None
        self._trigger = None
        self._last_pair_id = 0

    def start(self) -> None:
        try:
            if self.manage_capture_session:
                from paradex.io.camera_system.remote_camera_controller import (
                    remote_camera_controller,
                )
                from paradex.io.camera_system.signal_generator import UTGE900

                capture_pcs = list(dict.fromkeys(item.capture_pc for item in self.bindings))
                self._remote = remote_camera_controller(
                    name="act_xarm_allegro",
                    pc_list=capture_pcs,
                )
                self._remote.wait_until_ready()
                self._remote.start("stream", True, None, fps=int(self.fps))

                for reader in self.readers:
                    reader.start()
                self._trigger = UTGE900(**network_info["signal_generator"]["param"])
                self._trigger.start(fps=self.fps)
                self._remote.validate(timeout=10.0)
            else:
                for reader in self.readers:
                    reader.start()
            self.get_pair(timeout_seconds=self.frame_timeout_seconds)
        except Exception:
            self.close()
            raise

    @staticmethod
    def _decode_rgb(jpeg: bytes, height: int, width: int) -> np.ndarray:
        image_bgr = cv2.imdecode(np.frombuffer(jpeg, np.uint8), cv2.IMREAD_COLOR)
        if image_bgr is None:
            raise CameraStreamError("OpenCV could not decode camera JPEG")
        if image_bgr.shape[:2] != (height, width):
            image_bgr = cv2.resize(
                image_bgr,
                (width, height),
                interpolation=cv2.INTER_AREA,
            )
        return cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

    def get_pair(
        self,
        *,
        timeout_seconds: float | None = None,
        height: int = 480,
        width: int = 640,
    ) -> CameraPair:
        timeout = self.frame_timeout_seconds if timeout_seconds is None else float(timeout_seconds)
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            snapshots = [reader.snapshot() for reader in self.readers]
            if all(snapshots):
                common_ids = set(snapshots[0])
                for snapshot in snapshots[1:]:
                    common_ids.intersection_update(snapshot)
                eligible = [frame_id for frame_id in common_ids if frame_id > self._last_pair_id]
                if eligible:
                    frame_id = max(eligible)
                    encoded = [snapshot[frame_id] for snapshot in snapshots]
                    oldest_receive_ns = min(frame.received_monotonic_ns for frame in encoded)
                    if (time.monotonic_ns() - oldest_receive_ns) / 1e6 <= self.max_frame_age_ms:
                        self._last_pair_id = frame_id
                        images = {
                            binding.policy_key: self._decode_rgb(frame.jpeg, height, width)
                            for binding, frame in zip(self.bindings, encoded)
                        }
                        return CameraPair(
                            images=images,
                            frame_ids={binding.policy_key: frame_id for binding in self.bindings},
                            jpeg_bytes={
                                binding.policy_key: frame.jpeg
                                for binding, frame in zip(self.bindings, encoded)
                            },
                            received_monotonic_ns=oldest_receive_ns,
                        )
            time.sleep(0.002)
        errors = {
            reader.binding.physical_serial: reader.last_error
            for reader in self.readers
            if reader.last_error is not None
        }
        raise CameraStreamError(
            f"Timed out waiting for a fresh matched camera pair after {timeout:.2f}s; "
            f"reader_errors={errors}"
        )

    def close(self) -> None:
        if self._trigger is not None:
            try:
                self._trigger.stop()
            except Exception:
                pass
            try:
                self._trigger.end()
            except Exception:
                pass
            self._trigger = None
        for reader in self.readers:
            reader.close()
        if self._remote is not None:
            try:
                self._remote.stop()
            except Exception:
                pass
            try:
                self._remote.end()
            except Exception:
                pass
            self._remote = None

    def __enter__(self) -> "SynchronizedCameraStream":
        self.start()
        return self

    def __exit__(self, _exc_type, _exc, _traceback) -> None:
        self.close()
