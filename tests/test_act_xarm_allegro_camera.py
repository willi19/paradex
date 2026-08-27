import time
import threading

import cv2
import numpy as np
import pytest

from paradex.inference.act_xarm_allegro.camera import (
    CameraStreamError,
    EncodedFrame,
    SynchronizedCameraStream,
)
from paradex.inference.act_xarm_allegro.core import DEFAULT_CAMERA_BINDINGS
from paradex.io.camera_system.aravis_gstreamer import AravisGStreamerCamera


class FakeReader:
    def __init__(self, binding, frames):
        self.binding = binding
        self.frames = frames
        self.last_error = None

    def snapshot(self):
        return dict(self.frames)

    def close(self):
        pass


def jpeg(color_bgr):
    image = np.full((24, 32, 3), color_bgr, dtype=np.uint8)
    ok, encoded = cv2.imencode(".jpg", image)
    assert ok
    return encoded.tobytes()


def test_camera_alias_matching_rgb_and_resize():
    stream = SynchronizedCameraStream(DEFAULT_CAMERA_BINDINGS, manage_capture_session=False)
    now = time.monotonic_ns()
    period_ns = int(1e9 / 30)
    stream.readers = [
        FakeReader(
            DEFAULT_CAMERA_BINDINGS[0],
            {
                raw_id: EncodedFrame(raw_id, now - (9 - raw_id) * period_ns, jpeg((0, 0, 255)))
                for raw_id in (7, 8, 9)
            },
        ),
        FakeReader(
            DEFAULT_CAMERA_BINDINGS[1],
            {
                raw_id: EncodedFrame(
                    raw_id,
                    now - (10 - raw_id) * period_ns + 1_000_000,
                    jpeg((0, 255, 0)),
                )
                for raw_id in (8, 9, 10)
            },
        ),
    ]
    pair = stream.get_pair(timeout_seconds=0.05)
    assert pair.frame_ids == {
        "observation.images.cam_23029839": 3,
        "observation.images.cam_25452066": 3,
    }
    assert pair.raw_frame_ids == {
        "observation.images.cam_23029839": 9,
        "observation.images.cam_25452066": 10,
    }
    assert all(image.shape == (480, 640, 3) for image in pair.images.values())
    assert pair.images["observation.images.cam_23029839"][0, 0, 0] > 240


def test_camera_pair_must_advance_and_match():
    stream = SynchronizedCameraStream(DEFAULT_CAMERA_BINDINGS, manage_capture_session=False)
    now = time.monotonic_ns()
    period_ns = int(1e9 / 30)
    stream.readers = [
        FakeReader(
            DEFAULT_CAMERA_BINDINGS[0],
            {
                raw_id: EncodedFrame(raw_id, now - (3 - raw_id) * period_ns, jpeg((0, 0, 0)))
                for raw_id in (1, 2, 3)
            },
        ),
        FakeReader(
            DEFAULT_CAMERA_BINDINGS[1],
            {
                raw_id: EncodedFrame(
                    raw_id,
                    now - (4 - raw_id) * period_ns + 1_000_000,
                    jpeg((0, 0, 0)),
                )
                for raw_id in (2, 3, 4)
            },
        ),
    ]
    stream.get_pair(timeout_seconds=0.05)
    stream.readers[0].frames[5] = EncodedFrame(5, time.monotonic_ns(), jpeg((0, 0, 0)))
    stream.readers[1].frames[7] = EncodedFrame(7, time.monotonic_ns(), jpeg((0, 0, 0)))
    with pytest.raises(CameraStreamError):
        stream.get_pair(timeout_seconds=0.01)


def test_daemon_frame_endpoint_uses_hardware_frame_id():
    camera = AravisGStreamerCamera.__new__(AravisGStreamerCamera)
    camera._frame_count_lock = threading.Lock()
    camera._frame_count = 0
    camera._latest_frame = None
    camera._update_encoded_frame(b"jpeg", frame_id=123456)
    assert camera._frame_count == 1
    assert camera.get_frame() == (123456, b"jpeg")
