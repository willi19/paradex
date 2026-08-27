import cv2
import numpy as np

from paradex.inference.act_xarm_allegro.core import DEFAULT_CAMERA_BINDINGS, ObservationPacket
from paradex.inference.act_xarm_allegro.telemetry import RunLogger, iter_replay_packets


def test_telemetry_replay_round_trip(tmp_path):
    keys = [binding.policy_key for binding in DEFAULT_CAMERA_BINDINGS]
    full_resolution_jpegs = {}
    for index, key in enumerate(keys):
        ok, encoded = cv2.imencode(
            ".jpg", np.full((1200, 1600, 3), index * 40, dtype=np.uint8)
        )
        assert ok
        full_resolution_jpegs[key] = encoded.tobytes()
    packet = ObservationPacket(
        images={key: np.full((480, 640, 3), index * 40, dtype=np.uint8) for index, key in enumerate(keys)},
        state=np.arange(22, dtype=np.float64),
        frame_ids={key: 42 for key in keys},
        captured_monotonic_ns=100,
        state_monotonic_ns=101,
        jpeg_bytes=full_resolution_jpegs,
        raw_frame_ids={key: 100 + index for index, key in enumerate(keys)},
    )
    with RunLogger(tmp_path, run_name="test-run") as logger:
        logger.inference_boundary(packet, np.zeros((10, 25)), 5.5)
        run_dir = logger.run_dir
    replayed = list(iter_replay_packets(run_dir))
    assert len(replayed) == 1
    np.testing.assert_array_equal(replayed[0].state, packet.state)
    assert replayed[0].frame_ids == packet.frame_ids
    assert replayed[0].raw_frame_ids == packet.raw_frame_ids
    assert set(replayed[0].images) == set(keys)
    assert all(image.shape == (480, 640, 3) for image in replayed[0].images.values())
