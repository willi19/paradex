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
    stream.readers = [
        FakeReader(DEFAULT_CAMERA_BINDINGS[0], {7: EncodedFrame(7, now, jpeg((0, 0, 255)))}),
        FakeReader(DEFAULT_CAMERA_BINDINGS[1], {7: EncodedFrame(7, now, jpeg((0, 255, 0)))}),
    ]
    pair = stream.get_pair(timeout_seconds=0.05)
    assert pair.frame_ids == {
        "observation.images.cam_23029839": 7,
        "observation.images.cam_25452066": 7,
    }
    assert all(image.shape == (480, 640, 3) for image in pair.images.values())
    assert pair.images["observation.images.cam_23029839"][0, 0, 0] > 240


def test_camera_pair_must_advance_and_match():
    stream = SynchronizedCameraStream(DEFAULT_CAMERA_BINDINGS, manage_capture_session=False)
    now = time.monotonic_ns()
    stream.readers = [
        FakeReader(DEFAULT_CAMERA_BINDINGS[0], {8: EncodedFrame(8, now, jpeg((0, 0, 0)))}),
        FakeReader(DEFAULT_CAMERA_BINDINGS[1], {9: EncodedFrame(9, now, jpeg((0, 0, 0)))}),
    ]
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
