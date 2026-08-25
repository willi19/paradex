import json

import cv2
import numpy as np

from paradex.inference.act_xarm_allegro.core import (
    DEFAULT_CAMERA_BINDINGS,
    ObservationPacket,
)
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
        images={
            key: np.full((480, 640, 3), index * 40, dtype=np.uint8)
            for index, key in enumerate(keys)
        },
        state=np.arange(22, dtype=np.float64),
        frame_ids={key: 42 for key in keys},
        captured_monotonic_ns=100,
        state_monotonic_ns=101,
        jpeg_bytes=full_resolution_jpegs,
    )
    selected_actions = np.arange(10 * 25, dtype=np.float64).reshape(10, 25)
    full_action_chunk = np.arange(50 * 25, dtype=np.float64).reshape(50, 25)
    full_action_chunk[:10] = selected_actions
    with RunLogger(tmp_path, run_name="test-run") as logger:
        artifact = logger.inference_boundary(
            packet,
            selected_actions,
            full_action_chunk,
            5.5,
        )
        run_dir = logger.run_dir
    arrays = np.load(artifact / "observation_action.npz")
    np.testing.assert_array_equal(arrays["raw_actions"], selected_actions)
    np.testing.assert_array_equal(arrays["selected_actions"], selected_actions)
    np.testing.assert_array_equal(arrays["full_action_chunk"], full_action_chunk)
    metadata = json.loads((artifact / "metadata.json").read_text())
    assert metadata["selected_action_steps"] == 10
    assert metadata["predicted_chunk_steps"] == 50
    assert metadata["action_selection"] == "prefix"
    replayed = list(iter_replay_packets(run_dir))
    assert len(replayed) == 1
    np.testing.assert_array_equal(replayed[0].state, packet.state)
    assert replayed[0].frame_ids == packet.frame_ids
    assert set(replayed[0].images) == set(keys)
    assert all(image.shape == (480, 640, 3) for image in replayed[0].images.values())


def test_telemetry_rejects_selected_actions_that_are_not_chunk_prefix(tmp_path):
    packet = ObservationPacket(
        images={},
        state=np.zeros(22),
        frame_ids={},
        captured_monotonic_ns=100,
        state_monotonic_ns=101,
    )
    with RunLogger(tmp_path, run_name="invalid-run") as logger:
        try:
            logger.inference_boundary(packet, np.ones((2, 25)), np.zeros((5, 25)), 1.0)
        except ValueError as error:
            assert "exact prefix" in str(error)
        else:
            raise AssertionError("Expected mismatched action chunks to be rejected")


def test_telemetry_records_temporal_ensemble_actions(tmp_path):
    packet = ObservationPacket(
        images={},
        state=np.zeros(22),
        frame_ids={},
        captured_monotonic_ns=100,
        state_monotonic_ns=101,
    )
    full_action_chunk = np.zeros((50, 25))
    selected_actions = np.ones((10, 25))
    with RunLogger(tmp_path, run_name="ensemble-run") as logger:
        artifact = logger.inference_boundary(
            packet,
            selected_actions,
            full_action_chunk,
            1.0,
            action_selection="temporal_ensemble",
            ensemble_contributors=np.full(10, 2),
        )
    arrays = np.load(artifact / "observation_action.npz")
    np.testing.assert_array_equal(arrays["selected_actions"], selected_actions)
    np.testing.assert_array_equal(arrays["full_action_chunk"], full_action_chunk)
    np.testing.assert_array_equal(arrays["ensemble_contributors"], np.full(10, 2))
    metadata = json.loads((artifact / "metadata.json").read_text())
    assert metadata["action_selection"] == "temporal_ensemble"
