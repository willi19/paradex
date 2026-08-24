"""Non-recording live preview for the capture-PC camera daemon endpoints."""

from __future__ import annotations

import threading
import time
import os
from pathlib import Path
from typing import Callable, Iterable, Optional

# The OpenCV wheel bundles Qt but not its font directory.  Tell Qt to use the
# system DejaVu fonts before importing cv2 so it does not print a warning for
# every preview window created.
_SYSTEM_QT_FONT_DIR = "/usr/share/fonts/truetype/dejavu"
if Path(_SYSTEM_QT_FONT_DIR).is_dir():
    os.environ["QT_QPA_FONTDIR"] = _SYSTEM_QT_FONT_DIR

import cv2

# ``cv2`` can replace QT_QPA_FONTDIR with its own wheel-relative path during
# import, even when that directory was omitted from the wheel.
if Path(_SYSTEM_QT_FONT_DIR).is_dir():
    os.environ["QT_QPA_FONTDIR"] = _SYSTEM_QT_FONT_DIR
import numpy as np

from paradex.image.merge import merge_image
from paradex.io.camera_system.camera_daemon_reader import (
    CameraDaemonReader,
    CameraDaemonReaderError,
)
from paradex.utils.system import get_pc_ip


class CapturePcPreviewGui:
    """Display camera-daemon frames without participating in capture control."""

    def __init__(
        self,
        pc_list: Iterable[str],
        port: int = 5484,
        refresh_interval: float = 1.0 / 30.0,
        request_timeout: float = 1.5,
        window_name: str = "Capture PC Preview",
        reader_factory: Callable = CameraDaemonReader,
        host_lookup: Callable[[str], str] = get_pc_ip,
        cv2_module=cv2,
        side_panel_provider: Optional[Callable[[int, int], Optional[np.ndarray]]] = None,
        side_panel_refresh_interval: float = 1.0 / 30.0,
    ) -> None:
        self.pc_list = list(pc_list)
        self.port = int(port)
        self.refresh_interval = float(refresh_interval)
        if self.refresh_interval <= 0.0:
            raise ValueError("refresh_interval must be positive")
        self.request_timeout = float(request_timeout)
        self.window_name = window_name
        self._reader_factory = reader_factory
        self._host_lookup = host_lookup
        self._cv2 = cv2_module
        self._side_panel_provider = side_panel_provider
        self.side_panel_refresh_interval = float(side_panel_refresh_interval)
        if self.side_panel_refresh_interval <= 0.0:
            raise ValueError("side_panel_refresh_interval must be positive")

        self._readers = {}
        self._retry_after = {}
        self._last_failure_log = {}
        self._last_images = {}
        self._display_lock = threading.Lock()
        self._latest_display = None
        self._latest_camera_grid = None
        self._preview_closed = False
        self._next_gui_update = 0.0
        self._stop_event = threading.Event()
        self._thread = None
        self._side_thread = None
        self._side_lock = threading.Lock()
        self._side_panel_size = None
        self._latest_side_panel = None

    def start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return

        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._run,
            name="capture-pc-preview",
            daemon=True,
        )
        self._thread.start()
        if self._side_panel_provider is not None:
            self._side_thread = threading.Thread(
                target=self._run_side_panel,
                name="capture-pc-preview-side-panel",
                daemon=True,
            )
            self._side_thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=max(2.0, self.request_timeout + 1.0))
            self._thread = None
        if self._side_thread is not None:
            self._side_thread.join(timeout=max(2.0, self.request_timeout + 1.0))
            self._side_thread = None

    def close(self) -> None:
        """Stop preview I/O and close the window from the main thread."""

        self.stop()
        try:
            self._cv2.destroyWindow(self.window_name)
        except Exception:
            pass

    def show(self) -> None:
        """Process GUI events from the caller's main thread without blocking capture."""

        if self._preview_closed or time.monotonic() < self._next_gui_update:
            return
        self._next_gui_update = time.monotonic() + 1.0 / 30.0
        with self._display_lock:
            display = self._latest_display
        if display is None:
            return

        self._cv2.imshow(self.window_name, display)
        if self._cv2.waitKey(1) & 0xFF == 27:
            self._preview_closed = True
            self.stop()
            try:
                self._cv2.destroyWindow(self.window_name)
            except Exception:
                pass
            print("[Camera preview] Window closed; capture continues without preview.")

    def _connect_readers(self) -> None:
        now = time.monotonic()
        for pc_name in self.pc_list:
            if pc_name in self._readers or now < self._retry_after.get(pc_name, 0.0):
                continue

            try:
                self._readers[pc_name] = self._reader_factory(
                    self._host_lookup(pc_name),
                    port=self.port,
                    request_timeout=self.request_timeout,
                )
                print(f"[Camera preview] Connected to {pc_name}:{self.port}")
            except (CameraDaemonReaderError, OSError) as exc:
                self._retry_after[pc_name] = now + 2.0
                print(f"[Camera preview] {pc_name} unavailable: {exc}")

    def _collect_frames(self):
        updated_images = {}
        for pc_name, reader in list(self._readers.items()):
            for camera_name in reader.camera_names:
                display_name = f"{pc_name}:{camera_name}"
                try:
                    image = reader.get_preview(camera_name)
                except (CameraDaemonReaderError, OSError) as exc:
                    self._log_request_failure(pc_name, exc)
                    # A network failure is normally PC-wide.  Do not wait for
                    # every remaining camera to hit the same timeout.
                    break
                if image is not None:
                    updated_images[display_name] = image

        self._last_images.update(updated_images)
        frame_text = {
            display_name: "preview" if display_name in updated_images else "preview (stale)"
            for display_name in self._last_images
        }
        return dict(self._last_images), frame_text

    def _log_request_failure(self, pc_name: str, exc: Exception) -> None:
        """Report transient errors without turning them into a disconnect."""

        now = time.monotonic()
        if now < self._last_failure_log.get(pc_name, 0.0) + 2.0:
            return
        self._last_failure_log[pc_name] = now
        print(
            f"[Camera preview] {pc_name} request failed; retaining the last frame: {exc}"
        )

    def _status_image(self) -> np.ndarray:
        image = np.zeros((240, 720, 3), dtype=np.uint8)
        message = "Waiting for capture-PC previews on port {}".format(self.port)
        self._cv2.putText(
            image,
            message,
            (20, 120),
            self._cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 255, 255),
            1,
            self._cv2.LINE_AA,
        )
        return image

    def _waiting_side_panel(self, height: int, width: int) -> np.ndarray:
        image = np.zeros((height, width, 3), dtype=np.uint8)
        self._cv2.putText(
            image,
            "Open Viser at http://localhost:8080",
            (20, max(40, height // 2)),
            self._cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            1,
            self._cv2.LINE_AA,
        )
        return image

    def _compose_side_panel(self, display: np.ndarray) -> np.ndarray:
        if self._side_panel_provider is None:
            return display
        height, width = display.shape[:2]
        with self._side_lock:
            self._side_panel_size = (height, width)
            side_panel = self._latest_side_panel
        if side_panel is None:
            side_panel = self._waiting_side_panel(height, width)
        elif side_panel.shape[:2] != (height, width):
            side_panel = self._cv2.resize(side_panel, (width, height))
        separator = np.full((height, 10, 3), 255, dtype=np.uint8)
        return np.concatenate((display, separator, side_panel), axis=1)

    def _publish_camera_grid(self, camera_grid: np.ndarray) -> None:
        """Cache a camera frame and publish its current side-panel composite."""
        with self._display_lock:
            self._latest_camera_grid = camera_grid
        self._refresh_composite(camera_grid)

    def _refresh_composite(self, camera_grid: Optional[np.ndarray] = None) -> None:
        """Rebuild the display without waiting for another camera request."""
        if camera_grid is None:
            with self._display_lock:
                camera_grid = self._latest_camera_grid
        if camera_grid is None:
            return

        display = self._compose_side_panel(camera_grid)
        with self._display_lock:
            # Do not replace a newer camera frame if collection completed while
            # the side panel was being resized and concatenated.
            if camera_grid is self._latest_camera_grid:
                self._latest_display = display

    def _run_side_panel(self) -> None:
        """Fetch Viser renders without ever blocking camera collection."""
        while not self._stop_event.is_set():
            with self._side_lock:
                panel_size = self._side_panel_size
            if panel_size is not None:
                height, width = panel_size
                try:
                    image = self._side_panel_provider(height, width)
                except Exception as exc:
                    print(f"[Camera preview] Viser render failed: {exc}")
                    image = None
                if image is not None:
                    with self._side_lock:
                        self._latest_side_panel = image
                    self._refresh_composite()
            self._stop_event.wait(self.side_panel_refresh_interval)

    def _run(self) -> None:
        try:
            while not self._stop_event.is_set():
                started = time.monotonic()
                self._connect_readers()
                images, frame_text = self._collect_frames()
                display = (
                    merge_image(
                        images,
                        frame_text,
                        grid_cols=4,
                        preserve_aspect=True,
                        # Match the previous 5x4 preview's total height:
                        # 1200px image area + three 10px row gaps.
                        target_height=1230,
                    )
                    if images
                    else self._status_image()
                )
                self._publish_camera_grid(display)
                remaining = self.refresh_interval - (time.monotonic() - started)
                self._stop_event.wait(max(0.0, remaining))
        except Exception as exc:
            print(f"[Camera preview] Disabled after GUI error: {exc}")
