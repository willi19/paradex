"""Read decoded frames from capture-PC camera daemon HTTP endpoints."""

from __future__ import annotations

import json
import time
from typing import Dict, Iterable, Optional, Tuple
from urllib.error import HTTPError, URLError
from urllib.parse import quote
from urllib.request import Request, urlopen

import cv2
import numpy as np

from paradex.utils.system import get_pc_ip


class CameraDaemonReaderError(RuntimeError):
    """A camera daemon did not provide a usable frame."""


class CameraDaemonReader:
    """Poll full-resolution JPEG frames from one camera daemon."""

    def __init__(
        self,
        host: str,
        port: int = 5484,
        request_timeout: float = 2.0,
    ) -> None:
        self.base_url = "http://{}:{}".format(host, int(port))
        self.request_timeout = float(request_timeout)
        status = self._get_json("/cameras")
        self.backend = str(status.get("backend", "unknown"))
        self.camera_names = [
            str(camera["name"])
            for camera in status.get("cameras", [])
            if camera.get("name") is not None
        ]
        if not self.camera_names:
            raise CameraDaemonReaderError(
                "No cameras reported by {}".format(self.base_url)
            )

    def _open(self, path: str):
        request = Request(self.base_url + path, headers={"Cache-Control": "no-cache"})
        return urlopen(request, timeout=self.request_timeout)

    def _get_json(self, path: str) -> dict:
        try:
            with self._open(path) as response:
                return json.loads(response.read().decode("utf-8"))
        except (HTTPError, URLError, ValueError) as exc:
            raise CameraDaemonReaderError(
                "Could not read {}{}: {}".format(self.base_url, path, exc)
            )

    def get_image(self, camera_name: str) -> Optional[Tuple[np.ndarray, int]]:
        path = "/frame/{}".format(quote(str(camera_name), safe=""))
        try:
            with self._open(path) as response:
                encoded = response.read()
                frame_id = int(response.headers.get("X-Frame-Id", "0"))
        except HTTPError as exc:
            if exc.code == 404:
                return None
            raise CameraDaemonReaderError(
                "Could not read {}{}: {}".format(self.base_url, path, exc)
            )
        except (URLError, ValueError) as exc:
            raise CameraDaemonReaderError(
                "Could not read {}{}: {}".format(self.base_url, path, exc)
            )

        image = cv2.imdecode(np.frombuffer(encoded, dtype=np.uint8), cv2.IMREAD_COLOR)
        if image is None:
            raise CameraDaemonReaderError(
                "Camera {} returned an invalid JPEG".format(camera_name)
            )
        return image, frame_id

    def get_images(self, copy: bool = True) -> Dict[str, Tuple[np.ndarray, int]]:
        images = {}
        for camera_name in self.camera_names:
            result = self.get_image(camera_name)
            if result is None:
                continue
            image, frame_id = result
            images[camera_name] = (image.copy() if copy else image, frame_id)
        return images


class MultiCameraDaemonReader:
    """Aggregate daemon frames from all selected capture PCs."""

    def __init__(
        self,
        pc_list: Iterable[str],
        port: int = 5484,
        request_timeout: float = 2.0,
    ) -> None:
        self.readers = [
            CameraDaemonReader(get_pc_ip(pc), port, request_timeout)
            for pc in pc_list
        ]
        self.camera_names = [
            camera_name
            for reader in self.readers
            for camera_name in reader.camera_names
        ]
        self.backends = {reader.backend for reader in self.readers}
        if len(set(self.camera_names)) != len(self.camera_names):
            raise CameraDaemonReaderError(
                "Camera serials must be unique across capture PCs: {}".format(
                    self.camera_names
                )
            )

    def get_images(self, copy: bool = True) -> Dict[str, Tuple[np.ndarray, int]]:
        images = {}
        for reader in self.readers:
            images.update(reader.get_images(copy=copy))
        return images

    def wait_for_new_frames(
        self,
        last_frame_ids: Optional[Dict[str, int]] = None,
        timeout: float = 5.0,
    ) -> Dict[str, Tuple[np.ndarray, int]]:
        previous = last_frame_ids or {}
        deadline = time.monotonic() + float(timeout)
        latest = {}
        while time.monotonic() < deadline:
            latest = self.get_images(copy=False)
            if all(
                name in latest and latest[name][1] > previous.get(name, -1)
                for name in self.camera_names
            ):
                return latest
            time.sleep(0.02)

        missing = [
            name
            for name in self.camera_names
            if name not in latest or latest[name][1] <= previous.get(name, -1)
        ]
        raise CameraDaemonReaderError(
            "Timed out waiting for new frames from: {}".format(missing)
        )
