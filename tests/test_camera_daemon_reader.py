import json
import unittest
from unittest.mock import patch

import cv2
import numpy as np

from paradex.io.camera_system.camera_daemon_reader import (
    CameraDaemonReader,
    CameraDaemonReaderError,
    MultiCameraDaemonReader,
)


class FakeResponse:
    def __init__(self, body, headers=None):
        self.body = body
        self.headers = headers or {}

    def __enter__(self):
        return self

    def __exit__(self, _exc_type, _exc, _traceback):
        return False

    def read(self):
        return self.body


class CameraDaemonReaderTests(unittest.TestCase):
    def test_reader_discovers_and_decodes_full_resolution_frame(self):
        image = np.full((6, 8, 3), 127, dtype=np.uint8)
        ok, encoded = cv2.imencode(".jpg", image)
        self.assertTrue(ok)
        responses = [
            FakeResponse(
                json.dumps(
                    {
                        "backend": "aravis-gstreamer",
                        "cameras": [{"name": "cam-a"}],
                    }
                ).encode("utf-8")
            ),
            FakeResponse(encoded.tobytes(), {"X-Frame-Id": "42"}),
        ]

        with patch(
            "paradex.io.camera_system.camera_daemon_reader.urlopen",
            side_effect=responses,
        ):
            reader = CameraDaemonReader("capture-a")
            frames = reader.get_images()

        self.assertEqual(reader.camera_names, ["cam-a"])
        self.assertEqual(reader.backend, "aravis-gstreamer")
        decoded, frame_id = frames["cam-a"]
        self.assertEqual(decoded.shape, image.shape)
        self.assertEqual(frame_id, 42)

    def test_reader_rejects_daemon_without_cameras(self):
        response = FakeResponse(json.dumps({"cameras": []}).encode("utf-8"))
        with patch(
            "paradex.io.camera_system.camera_daemon_reader.urlopen",
            return_value=response,
        ), self.assertRaisesRegex(CameraDaemonReaderError, "No cameras"):
            CameraDaemonReader("capture-a")

    def test_multi_reader_waits_until_every_camera_is_new(self):
        reader = MultiCameraDaemonReader.__new__(MultiCameraDaemonReader)
        reader.camera_names = ["cam-a", "cam-b"]
        reader.readers = []
        old = {
            "cam-a": (np.zeros((1, 1, 3), dtype=np.uint8), 2),
            "cam-b": (np.zeros((1, 1, 3), dtype=np.uint8), 4),
        }
        new = {
            "cam-a": (np.zeros((1, 1, 3), dtype=np.uint8), 3),
            "cam-b": (np.zeros((1, 1, 3), dtype=np.uint8), 5),
        }

        with patch.object(reader, "get_images", side_effect=[old, new]):
            frames = reader.wait_for_new_frames(
                {"cam-a": 2, "cam-b": 4}, timeout=1.0
            )

        self.assertEqual(frames["cam-a"][1], 3)
        self.assertEqual(frames["cam-b"][1], 5)


if __name__ == "__main__":
    unittest.main()
