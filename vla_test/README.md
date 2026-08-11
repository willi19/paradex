# GR00T N1.7 on xArm6 + Inspire — portable bundle

This folder is a **deploy-ready snapshot** of the custom code, docs, and patches we built on top of NVIDIA's
`nvidia/Isaac-GR00T` to finetune GR00T-N1.7 on an xArm6 + Inspire right-hand embodiment and serve the resulting
policy. Drop this `sungjoo/` tree onto any host with docker + an Ada/Ampere-class GPU and you can rebuild the
inference path end-to-end.

The local source of truth is `/media/ssd1/sungjoo/Isaac-GR00T/` (a fork of `nvidia/Isaac-GR00T`); everything here
is a copy out of that fork.

## Layout

```
sungjoo/
├── dataset/                         LeRobot v2 datasets (formal format)
│   └── task/
│       ├── 1_throw_the_pepsi_to_basket_v1/    (28 eps, 25,032 frames)
│       └── 2_broom_sweep_v1/                  (30 eps, 35,688 frames)
├── checkpoints/                     finetuned GR00T-N1.7-3B (NEW_EMBODIMENT)
│   ├── xarm_inspire_broom_run2_8k/broom_run2_8k/checkpoint-{2000,4000,8000}
│   └── xarm_inspire_pepsi_run1/pepsi_run1/checkpoint-2000
├── docs/
│   ├── README.md                    you are here
│   ├── POLICY_SERVER.md             server-side runtime + ZMQ protocol detail
│   └── STATUS.html                  full phase log (phases 0–10, issues table)
└── scripts/                         every custom script + the modality config
    ├── run_policy_server.sh
    ├── run_finetune_xarm_inspire.sh
    ├── run_isaac_lab.sh
    ├── convert_teleop_to_lerobot.py
    ├── inspect_teleop_episode.py
    ├── review_lerobot_dataset.py
    ├── _smoke_dataset_loader.py
    ├── _run_open_loop_eval.py
    ├── fake_sim_client.py
    ├── _diag_arm_overlay.py
    └── xarm_inspire_config.py       (NB: in upstream tree this lives at
                                       examples/xarm_inspire/xarm_inspire_config.py)
```

## Custom-file inventory

All files in `scripts/` and the `POLICY_SERVER.md` doc are **user additions** — none of them come from the upstream
`nvidia/Isaac-GR00T` repo.

| File (this bundle) | Upstream drop-in path | Purpose | When used |
|---|---|---|---|
| `scripts/convert_teleop_to_lerobot.py` | `scripts/convert_teleop_to_lerobot.py` | xArm6+Inspire teleop (NPY/MP4) → LeRobot v2. Handles arm deg→rad, np.unwrap, per-cam frame-count drift, OS-style `(copy)` dupe-mp4s. | dataset prep |
| `scripts/inspect_teleop_episode.py` | `scripts/inspect_teleop_episode.py` | One-episode schema dump + 4-cam × 3-timestamp montage. Locked dims/fps from this. | dataset prep |
| `scripts/review_lerobot_dataset.py` | `scripts/review_lerobot_dataset.py` | Manual QA over the converted LeRobot dataset (frame-count parity, per-episode CSV). | dataset prep |
| `scripts/_smoke_dataset_loader.py` | `scripts/_smoke_dataset_loader.py` | Loads `LeRobotEpisodeLoader` + a few trajectories without touching a model — proves modality config + parquet + video keys align. | dataset prep |
| `scripts/_diag_arm_overlay.py` | `scripts/_diag_arm_overlay.py` | Plots arm `state` vs recorded `action_qpos` per joint — diagnoses unit mismatch / frame skew. | dataset prep, debug |
| `scripts/xarm_inspire_config.py` | `examples/xarm_inspire/xarm_inspire_config.py` | Modality config that registers `EmbodimentTag.NEW_EMBODIMENT` with 4 cams + 12-D state/action (arm:6 + hand:6), 16-step RELATIVE action horizon. **Must be pre-imported** before any `Gr00tPolicy(...)`. | training, serving |
| `scripts/run_finetune_xarm_inspire.sh` | `scripts/run_finetune_xarm_inspire.sh` | Wraps `gr00t.experiment.launch_finetune`. Defaults to paged 8-bit Adam + gradient checkpointing (24 GB-VRAM-safe). Env knobs: `NUM_GPUS`, `MAX_STEPS`, `GLOBAL_BATCH_SIZE`, `GRAD_ACCUM`, `RUN_NAME`, `DATASET`, `BASE_MODEL`, `EMBODIMENT`. | training |
| `scripts/run_policy_server.sh` | `scripts/run_policy_server.sh` | Pre-imports the modality config, then runs `gr00t.eval.run_gr00t_server`. Env: `CHECKPOINT`, `PORT` (5555), `HOST` (0.0.0.0), `GPU`. | serving |
| `scripts/fake_sim_client.py` | `scripts/fake_sim_client.py` | Headless ZMQ client that replays teleop observations against the live server — gives MSE/MAE vs ground-truth actions without Isaac Lab. | serving, eval |
| `scripts/_run_open_loop_eval.py` | `scripts/_run_open_loop_eval.py` | Wrapper around `gr00t.eval.open_loop_eval` that pre-imports the modality config. | eval |
| `scripts/run_isaac_lab.sh` | `scripts/run_isaac_lab.sh` | Launches `nvcr.io/nvidia/isaac-lab:2.3.2` with the right mounts + `--network host` so the in-container client can reach the policy server. | closed-loop sim |
| `docs/POLICY_SERVER.md` | `docs/POLICY_SERVER.md` | Server runtime requirements, port/network, ZMQ protocol summary, observation/action schemas. | reference |
| `docs/STATUS.html` | (lives at repo root, not inside the repo) | Phase log 0–10, decisions, issues table. Read top-down before deploying. | reference |

## Upstream patches (NOT in this bundle — apply by hand)

We modified **two** files in the upstream `gr00t/experiment/` tree. They're tiny env-var forwarders; the explanations
below are enough to recreate them on a fresh clone.

### 1. `gr00t/experiment/launch_finetune.py` — env-var optimizer + gradient-checkpointing overrides

**Why**: stock `launch_finetune.py` only honors what `FinetuneConfig` exposes, but `FinetuneConfig` has no flag for
the optimizer name or `gradient_checkpointing`. On a 24 GB RTX A5000 the model + fp32 Adam state alone is ~26 GB
and OOMs before we ever see an activation. Two-line forwarding lets us flip to `paged_adamw_8bit` (`bitsandbytes`)
and enable gradient checkpointing without editing source.

**What to change**: after `config = get_default_config().load_dict({...})` and the existing block that mutates
`config.training.*`, insert:

```python
config.training.optim = os.environ.get("GR00T_OPTIM", "adamw_torch")
if os.environ.get("GR00T_GRAD_CKPT", "0") == "1":
    config.training.gradient_checkpointing = True
```

(Already present in our `scripts/run_finetune_xarm_inspire.sh` defaults: `GR00T_OPTIM=paged_adamw_8bit`,
`GR00T_GRAD_CKPT=1`.) Also do `uv pip install bitsandbytes` inside the gr00t container; everything else is wheel.

### 2. `gr00t/experiment/experiment.py` — env-var DDP timeout

**Why**: HF Trainer's default `ddp_timeout` is 600 s. The first NCCL barrier after `__init__` doesn't fire until
each rank has finished its `meta/stats.json` regeneration (~10 min for our 30-episode dataset), so DDP runs die
at exactly 10:00 every time. Forwarding an env var lets us raise it without editing source.

**What to change**: where `TrainingArguments(...)` is constructed, ensure `ddp_timeout` is read from
`int(os.environ.get("GR00T_DDP_TIMEOUT_S", "1800"))`.

**Caveat documented in STATUS.html (Phase 8 issue 14)**: `TrainingArguments.ddp_timeout` only retimes the HF
process group; the launch-time PG that `torchrun`/accelerate creates keeps its own 600 s watchdog. The patch alone
doesn't make 2-GPU work — we dropped to single-GPU in practice. So **the patch is only useful if you're going to
also set the launch-side timeout env vars** (`TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=...`) and own that decision. For
single-GPU runs (default in `run_finetune_xarm_inspire.sh` with `NUM_GPUS=1`) you don't need this patch at all.

## Deploy the policy server in a new environment

Assumes the target host has docker + an Ampere/Ada GPU + CUDA 12.8 driver and can mount this NFS tree somewhere
(call that mount point `<NFS>`). Roughly 30 min of work.

1. **Clone Isaac-GR00T**
   ```bash
   git clone https://github.com/NVIDIA/Isaac-GR00T.git
   cd Isaac-GR00T
   ```
   STATUS.html, Phase 0 records the commit we built against — match it if you want bit-for-bit reproducibility.

2. **Apply the two upstream patches** described above to `gr00t/experiment/launch_finetune.py` and
   `gr00t/experiment/experiment.py`. (Skip the DDP one if you'll only ever run single-GPU.)

3. **Drop in the custom files**
   ```bash
   # from this bundle:
   cp <NFS>/sungjoo/scripts/{run_policy_server.sh,run_finetune_xarm_inspire.sh,run_isaac_lab.sh,convert_teleop_to_lerobot.py,inspect_teleop_episode.py,review_lerobot_dataset.py,_smoke_dataset_loader.py,_run_open_loop_eval.py,fake_sim_client.py,_diag_arm_overlay.py} \
      scripts/
   mkdir -p examples/xarm_inspire
   cp <NFS>/sungjoo/scripts/xarm_inspire_config.py examples/xarm_inspire/
   ```

4. **Build the gr00t docker image**
   ```bash
   docker build -f docker/Dockerfile -t gr00t .
   ```

5. **Launch the container with this NFS mounted**. Minimal version (mirrors the working setup):
   ```bash
   docker run -d --name gr00t --gpus all --ipc=host \
     -v <NFS>/sungjoo/checkpoints:/data/checkpoints:ro \
     -v <NFS>/sungjoo/dataset:/data/datasets:ro \
     -v $(pwd):/workspace \
     gr00t sleep infinity
   ```

6. **Start the policy server** (pick a checkpoint from `checkpoints/`):
   ```bash
   docker exec gr00t bash -lc \
     "CHECKPOINT=/data/checkpoints/xarm_inspire_pepsi_run1/pepsi_run1/checkpoint-2000 \
      PORT=5555 GPU=0 \
      bash /workspace/scripts/run_policy_server.sh"
   ```
   The wrapper pre-imports the modality config so `NEW_EMBODIMENT` is registered before `Gr00tPolicy` looks it up.

7. **Smoke-test without Isaac Lab**
   ```bash
   docker exec gr00t bash -lc \
     "python /workspace/scripts/fake_sim_client.py \
        --modality-config-path /workspace/examples/xarm_inspire/xarm_inspire_config.py \
        --dataset-path /data/datasets/task/1_throw_the_pepsi_to_basket_v1 \
        --traj-id 0 --steps 200"
   ```
   Expect MSE/MAE values well below the action variance; if the server crashes or returns NaN, the modality config
   or the embodiment tag is wrong.

8. **Closed-loop with Isaac Lab (optional)** — see `run_isaac_lab.sh` for the second-container launch + the
   client script in `isaac_lab_workspace/scripts/10_rollout_with_gr00t.py` (lives outside this bundle but is
   referenced in STATUS.html Phase 10).

## Sibling artifacts on this NFS

| What | Where | Size |
|---|---|---|
| Pepsi LeRobot dataset | `<NFS>/sungjoo/dataset/task/1_throw_the_pepsi_to_basket_v1/` | 3.2 GB |
| Broom LeRobot dataset | `<NFS>/sungjoo/dataset/task/2_broom_sweep_v1/` | 4.4 GB |
| Broom checkpoints (8k run) | `<NFS>/sungjoo/checkpoints/xarm_inspire_broom_run2_8k/broom_run2_8k/checkpoint-{2000,4000,8000}` | ~15 GB each |
| Pepsi checkpoint (2k run) | `<NFS>/sungjoo/checkpoints/xarm_inspire_pepsi_run1/pepsi_run1/checkpoint-2000` | ~15 GB |

Both checkpoint dirs are self-contained Hugging Face layouts (`config.json` + 3 safetensors shards + `processor/`
+ `experiment_cfg/`). Any of them can be passed straight to `run_policy_server.sh` via `CHECKPOINT=`.

## References

- **`docs/POLICY_SERVER.md`** — protocol-level detail: ZMQ REP/REQ + msgpack-numpy, observation/action schemas, the
  RPC list (`get_action`, `reset`, `ping`, `kill`, `get_modality_config`), and what kind of container/network setup
  the Isaac Lab client expects.
- **`docs/STATUS.html`** — phase-by-phase log, with the issues table (rows 1–14) covering every dead-end we hit
  (HF gating, parquet LFS pointers, deg/rad mismatch, frame-count drift, OOM math, NCCL timeout, …) and what fixed
  it. The single most useful doc when something breaks.
