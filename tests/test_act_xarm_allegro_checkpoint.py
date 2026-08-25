import os

import numpy as np
import pytest

from paradex.inference.act_xarm_allegro.core import RunnerConfig
from paradex.inference.act_xarm_allegro.policy import load_policy


@pytest.mark.skipif(
    os.environ.get("PARADEX_RUN_ACT_CHECKPOINT_TEST") != "1",
    reason="set PARADEX_RUN_ACT_CHECKPOINT_TEST=1 in the lerobot CUDA environment",
)
def test_checkpoint_saved_processors_and_synthetic_inference():
    config = RunnerConfig(mode="contract", device="cuda", action_steps=10)
    loaded = load_policy(config)
    assert loaded.contract.state_dim == 22
    assert loaded.contract.action_dim == 25
    assert loaded.contract.chunk_size == 50
    assert loaded.contract.n_action_steps == 10
    images = {
        key: np.zeros((480, 640, 3), dtype=np.uint8)
        for key in loaded.contract.image_keys
    }
    prediction = loaded.infer(images, np.zeros(22, dtype=np.float32), 10)
    assert prediction.selected_actions.shape == (10, 25)
    assert prediction.full_action_chunk.shape == (50, 25)
    np.testing.assert_array_equal(
        prediction.selected_actions,
        prediction.full_action_chunk[:10],
    )
    assert np.all(np.isfinite(prediction.full_action_chunk))
