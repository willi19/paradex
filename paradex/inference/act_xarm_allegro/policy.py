"""Checkpoint loading, contract validation, and ACT inference."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import tempfile
import time

import numpy as np

from paradex.inference.act_xarm_allegro.core import (
    ALLEGRO_PHYSICAL_LOWER,
    ALLEGRO_PHYSICAL_UPPER,
    ACTION_DIM,
    ACTION_KEY,
    EXPECTED_ACTION_AXES,
    EXPECTED_STATE_AXES,
    STATE_DIM,
    STATE_KEY,
    RunnerConfig,
    SafetyConfig,
)


@dataclass(frozen=True)
class PolicyContract:
    image_keys: tuple[str, ...]
    state_dim: int
    action_dim: int
    chunk_size: int
    n_action_steps: int


@dataclass
class LoadedPolicy:
    policy: object
    preprocessor: object
    postprocessor: object
    contract: PolicyContract
    snapshot_path: str

    def infer(self, images: dict[str, np.ndarray], state: np.ndarray, action_steps: int) -> tuple[np.ndarray, float]:
        import torch
        from lerobot.policies.utils import prepare_observation_for_inference

        observation = {key: np.asarray(value).copy() for key, value in images.items()}
        observation[STATE_KEY] = np.asarray(state, dtype=np.float32).reshape(STATE_DIM)
        batch = prepare_observation_for_inference(observation, torch.device(self.policy.config.device))
        batch = self.preprocessor(batch)
        if torch.cuda.is_available() and str(self.policy.config.device).startswith("cuda"):
            torch.cuda.synchronize()
        started = time.perf_counter()
        with torch.inference_mode():
            normalized = self.policy.predict_action_chunk(batch)[:, :action_steps]
            actions = self.postprocessor(normalized)
        if torch.cuda.is_available() and str(self.policy.config.device).startswith("cuda"):
            torch.cuda.synchronize()
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        if hasattr(actions, "detach"):
            actions = actions.detach().cpu().numpy()
        result = np.asarray(actions, dtype=np.float64)
        if result.shape != (1, action_steps, ACTION_DIM):
            raise RuntimeError(
                f"Policy returned {result.shape}, expected {(1, action_steps, ACTION_DIM)}"
            )
        if not np.all(np.isfinite(result)):
            raise RuntimeError("Policy returned non-finite actions")
        return result[0], elapsed_ms


def _feature_shape(feature: object) -> tuple[int, ...]:
    shape = getattr(feature, "shape", None)
    return tuple(int(value) for value in shape)


def validate_policy_contract(config: object, expected_image_keys: tuple[str, ...], action_steps: int) -> PolicyContract:
    input_features = config.input_features
    output_features = config.output_features
    actual_image_keys = tuple(key for key in input_features if key.startswith("observation.images."))
    if set(actual_image_keys) != set(expected_image_keys):
        raise ValueError(f"Checkpoint image keys {actual_image_keys} do not match {expected_image_keys}")
    if _feature_shape(input_features[STATE_KEY]) != (STATE_DIM,):
        raise ValueError("Checkpoint observation.state is not 22D")
    for key in expected_image_keys:
        if _feature_shape(input_features[key]) != (3, 480, 640):
            raise ValueError(f"Checkpoint image feature {key} is not (3,480,640)")
    if _feature_shape(output_features[ACTION_KEY]) != (ACTION_DIM,):
        raise ValueError("Checkpoint action is not 25D")
    chunk_size = int(config.chunk_size)
    saved_steps = int(config.n_action_steps)
    if action_steps > saved_steps or saved_steps > chunk_size:
        raise ValueError(
            f"Requested {action_steps} actions, checkpoint supports {saved_steps}/{chunk_size}"
        )
    return PolicyContract(actual_image_keys, STATE_DIM, ACTION_DIM, chunk_size, saved_steps)


def load_policy(config: RunnerConfig) -> LoadedPolicy:
    """Load a checkpoint while removing only the known unsupported config field."""

    from huggingface_hub import snapshot_download
    from lerobot.configs.policies import PreTrainedConfig
    from lerobot.policies.act.modeling_act import ACTPolicy
    from lerobot.policies.factory import make_pre_post_processors

    source = Path(config.policy_path).expanduser()
    if source.is_dir():
        snapshot = str(source)
    else:
        snapshot = snapshot_download(config.policy_path, revision=config.policy_revision)
    config_path = Path(snapshot) / "config.json"
    raw_config = json.loads(config_path.read_text())
    raw_config.pop("pretrained_revision", None)
    raw_config["device"] = config.device
    with tempfile.TemporaryDirectory(prefix="act-config-") as temporary:
        Path(temporary, "config.json").write_text(json.dumps(raw_config))
        policy_config = PreTrainedConfig.from_pretrained(temporary)
    contract = validate_policy_contract(
        policy_config,
        tuple(binding.policy_key for binding in config.camera_bindings),
        config.action_steps,
    )
    policy = ACTPolicy.from_pretrained(snapshot, config=policy_config)
    preprocessor, postprocessor = make_pre_post_processors(policy_config, pretrained_path=snapshot)
    return LoadedPolicy(policy, preprocessor, postprocessor, contract, snapshot)


def load_safety_config(
    dataset_repo_id: str,
    *,
    control_hz: float,
    max_linear_speed_m_s: float,
    max_angular_speed_deg_s: float,
    max_hand_speed_rad_s: float,
    max_observation_age_ms: float,
    max_state_age_ms: float,
    max_consecutive_faults: int,
    workspace_lower: np.ndarray | None = None,
    workspace_upper: np.ndarray | None = None,
) -> SafetyConfig:
    from huggingface_hub import hf_hub_download

    validate_dataset_contract(dataset_repo_id)
    stats_path = hf_hub_download(dataset_repo_id, "meta/stats.json", repo_type="dataset")
    stats = json.loads(Path(stats_path).read_text())
    action_lower = np.asarray(stats[ACTION_KEY]["min"], dtype=np.float64)
    action_upper = np.asarray(stats[ACTION_KEY]["max"], dtype=np.float64)
    action_lower[9:] = np.maximum(action_lower[9:], ALLEGRO_PHYSICAL_LOWER)
    action_upper[9:] = np.minimum(action_upper[9:], ALLEGRO_PHYSICAL_UPPER)
    if workspace_lower is not None:
        action_lower[:3] = np.maximum(action_lower[:3], np.asarray(workspace_lower, dtype=np.float64))
    if workspace_upper is not None:
        action_upper[:3] = np.minimum(action_upper[:3], np.asarray(workspace_upper, dtype=np.float64))
    return SafetyConfig(
        state_lower=np.asarray(stats[STATE_KEY]["q01"], dtype=np.float64),
        state_upper=np.asarray(stats[STATE_KEY]["q99"], dtype=np.float64),
        action_lower=action_lower,
        action_upper=action_upper,
        allegro_start_target=np.asarray(stats[STATE_KEY]["q50"][6:], dtype=np.float64),
        control_hz=control_hz,
        max_linear_speed_m_s=max_linear_speed_m_s,
        max_angular_speed_deg_s=max_angular_speed_deg_s,
        max_hand_speed_rad_s=max_hand_speed_rad_s,
        max_observation_age_ms=max_observation_age_ms,
        max_state_age_ms=max_state_age_ms,
        max_consecutive_faults=max_consecutive_faults,
    )


def validate_dataset_contract(dataset_repo_id: str) -> dict[str, tuple[str, ...]]:
    from huggingface_hub import hf_hub_download

    info_path = hf_hub_download(dataset_repo_id, "meta/info.json", repo_type="dataset")
    info = json.loads(Path(info_path).read_text())
    state_axes = tuple(info["features"][STATE_KEY]["names"]["axes"])
    action_axes = tuple(info["features"][ACTION_KEY]["names"]["axes"])
    if state_axes != EXPECTED_STATE_AXES:
        raise ValueError(f"Dataset state order does not match robot contract: {state_axes}")
    if action_axes != EXPECTED_ACTION_AXES:
        raise ValueError(f"Dataset action order does not match decoder contract: {action_axes}")
    return {"state_axes": state_axes, "action_axes": action_axes}
