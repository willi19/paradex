# ACT inference on xArm6 + Allegro v5

This runner is intentionally split into two processes. The ROS bridge uses the
Python 3.8 ABI of the local Humble build and never imports PyTorch. The policy
runner uses the `lerobot` environment for CUDA and talks to the bridge over
localhost ZeroMQ.

The policy contract is fixed to:

- `observation.images.cam_23029839` from `capture16/26256735`
- `observation.images.cam_25452066` from `capture18/25452066`
- `observation.state`: xArm joints 0–5, then Allegro driver joints 0–15
- `action`: absolute TCP xyz, two rotation columns, then 16 Allegro targets

## Offline validation

From `/home/temp_id/paradex`:

```bash
conda activate lerobot
export PYTHONPATH=/home/temp_id/paradex:/home/temp_id/lerobot/src
python src/inference/act_xarm_allegro.py contract --device cuda
```

Replay a prior run without cameras or ROS:

```bash
python src/inference/act_xarm_allegro.py replay \
  --replay-dir ~/shared_data/inference/act_xarm_allegro/<run-id> \
  --max-chunks 10
```

## ROS bridge

Start this first in a separate terminal. `sis` supplies the ROS overlays; the
`robot` environment supplies a compatible Python 3.8, NumPy, and pyzmq.

```bash
conda activate robot
sis
export PYTHONPATH=/home/temp_id/paradex:${PYTHONPATH}
python src/inference/act_xarm_allegro_hardware.py --namespace right
```

This form publishes feedback but refuses policy targets. For live validation,
the bridge also needs the explicit hardware-side gate:

```bash
python src/inference/act_xarm_allegro_hardware.py \
  --namespace right \
  --allow-live
```

Do not start the bridge with `--allow-live` for shadow runs.

## Shadow and live rollout

In the LeRobot terminal, shadow mode starts the two capture daemons in
hardware-triggered stream mode, starts the shared UTG at 30 Hz, checks exact
advancing frame IDs, reads robot feedback, and evaluates commands without
publishing them:

```bash
python src/inference/act_xarm_allegro.py shadow \
  --duration 30 \
  --max-chunks 30
```

The capture PCs must run the matching Paradex revision. This revision changes
`/frame/<serial>` to expose the GigE Vision hardware frame ID instead of a
process-local encode count. Deploy it to `/opt/paradex` on `capture16` and
`capture18` and restart their camera-agent services before relying on the exact
cross-PC frame check. A persistent ID offset is a hard validation failure.

The first live run is limited to one saved 10-action chunk per Space press:

```bash
python src/inference/act_xarm_allegro.py live \
  --enable-live \
  --max-chunks-per-enable 1 \
  --duration 30
```

Live controls are global:

- `R`: re-arm, but only when cameras, robot state, controller status, start
  pose, workspace, and training-support checks pass.
- `R`: re-arm and execute continuously once checks pass.
- `Esc`: clear the queue, hold the measured robot configuration, and latch an
  abort. A new `R` is required.

The operator must establish the start pose and retain access to the physical
emergency stop. There is no automatic homing.

## Safety and telemetry

Actions are rejected rather than silently clipped. Bounds intersect dataset
min/max, the Allegro v5 URDF hard limits, optional CLI workspace bounds, and
Cartesian/hand rate limits. Stale or mismatched inputs clear the ACT queue.
Repeated action rejection or any controller/camera error latches the bridge.

Every run writes `telemetry.jsonl`. Each inference boundary additionally writes
the paired JPEGs, state, the selected 10 actions, the full predicted action
chunk (50 actions for the current checkpoint), frame IDs, monotonic timestamps,
and inference timing under:

```text
~/shared_data/inference/act_xarm_allegro/<run-id>/chunk_XXXXXX/
```

Hardware execution uses temporal ensembling: each new 10-action execution
window is blended with any earlier full chunks that predict the same execution
times. Newer predictions receive larger weights; tune this with
`--temporal-ensemble-decay` (default `0.01`, `0` for an unweighted mean).

Useful overrides include `--camera POLICY_KEY=SERIAL@CAPTURE_PC` (repeat twice),
`--workspace-lower x,y,z`, `--workspace-upper x,y,z`, freshness thresholds,
rate limits, endpoints, duration, and chunk limits. Use
`python src/inference/act_xarm_allegro.py --help` for the complete interface.
