# GR00T Policy Server — runtime requirements

## What it is

`gr00t/eval/run_gr00t_server.py` loads a finetuned GR00T-N1.7 checkpoint
and serves it over **ZMQ REP/REQ** with msgpack-numpy framing.
Clients (e.g. the Isaac Lab rollout) send observation dicts and receive
action chunks back.

Wrapper script: `scripts/run_policy_server.sh`
(pre-imports the `xarm_inspire` modality config before the server reads
`NEW_EMBODIMENT` from the checkpoint).

## What you need to run it

| Layer | Requirement |
|---|---|
| Container runtime | Docker (one container per side). No conda env on host. |
| GPU | 1× ≥ 12 GB VRAM. The current setup pins the server to a single GPU via `CUDA_VISIBLE_DEVICES`. |
| Image | Built from `Isaac-GR00T/docker/Dockerfile` — `nvidia/cuda:12.8.0-devel-ubuntu22.04` base. The live image is `sungjoo_gr00t`. |
| Python | 3.10 inside the image's `/workspace/.venv` (managed by `uv`). |
| Key deps | torch 2.7.1, transformers 4.57.3, flash-attn 2.7.4, pyzmq 27.0.1, msgpack-numpy 0.4.8, peft 0.17.1, diffusers 0.35.1 (full list in `pyproject.toml`/`uv.lock`). |
| Networking | The Isaac Lab container runs `--network host`. The gr00t container is on the default bridge (172.17.0.3). Client connects to **`172.17.0.3:5555`** (or rebuild gr00t container with `--network host` and use `127.0.0.1:5555`). |
| Checkpoint | A finetuned directory containing `config.json`, the safetensors shards, `processor/`, and `experiment_cfg/`. Base GR00T-N1.7 refuses `NEW_EMBODIMENT` — must be finetuned. |
| Modality config | The Python file that registers `NEW_EMBODIMENT` (`examples/xarm_inspire/xarm_inspire_config.py`). It must be imported **before** `Gr00tPolicy(...)` runs the embodiment lookup — the wrapper script does this via `importlib`. |

## Isaac Lab is NOT required to run the server

The server is just inference. Isaac Lab is the eventual client (for closed-loop sim rollouts), but the server itself only needs the gr00t container.

For headless testing of just the server you can use:
`Isaac-GR00T/scripts/fake_sim_client.py` (synthetic observations from a teleop episode, runs inside the same gr00t container).

## Start command (current setup)

```bash
docker exec sungjoo_gr00t bash -lc \
  "CHECKPOINT=/data/checkpoints/<run_name>/<run_name>/checkpoint-<N> \
   PORT=5555 GPU=1 \
   bash /workspace/scripts/run_policy_server.sh"
```

Knobs (env vars on the wrapper):

- `CHECKPOINT` — absolute path inside the container (e.g. `/data/checkpoints/xarm_inspire_broom_run1/broom_run1/checkpoint-2000`)
- `MODALITY_CFG` — defaults to `/workspace/examples/xarm_inspire/xarm_inspire_config.py`
- `PORT` (5555), `HOST` (`0.0.0.0`), `GPU` (1), `EMBODIMENT` (`NEW_EMBODIMENT`)

## Protocol summary (for the Isaac Lab client)

- Transport: ZMQ REP (server) ↔ REQ (client)
- Encoding: msgpack-numpy
- RPCs registered by `PolicyServer`: `get_action(obs, options)`, `reset(options)`, `ping()`, `kill()`, `get_modality_config()`
- Observation dict: `state` (12-D = arm 6 + hand 6) + the 4 camera images keyed by modality (`cam0..cam3`).
- Action dict: `arm` (6-D relative deltas) + `hand` (6-D relative deltas), shape `[execution_horizon, 6]`.
