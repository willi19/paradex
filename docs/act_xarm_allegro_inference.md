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
cross-PC frame check. At each runner startup, time-coincident frames calibrate
the cameras' fixed raw-counter offset; the runner then requires exact,
advancing normalized trigger IDs. The planner waits for the next matching pair;
both raw and normalized IDs are retained in telemetry.

Live execution stays on after a single `R` press and stops on `Esc` (or the
configured duration/chunk limit):

```bash
python src/inference/act_xarm_allegro.py live \
  --enable-live \
  --duration 30
```

The live runtime uses two independent loops. A monotonic-deadline publisher
sends queued actions at `--control-hz` (30 Hz by default), while a planner
thread acquires the next synchronized camera pair, reads robot feedback, and
runs ACT before the current queue reaches four remaining actions. The predicted
chunk is placed on the queue before its JPEG/NPZ telemetry is written, so camera,
inference, and artifact I/O latency do not pause command publication at chunk
boundaries.

Before every live run, the runner issues the training dataset's median 16-joint
Allegro target through a hand-only bridge command. It does not wait for a
convergence check before inference; use `--no-allegro-preposition` only when an
externally established hand pose must be preserved. Shadow, replay, and contract
modes never preposition or publish robot commands.

Live controls are global:

- `R`: re-arm and start continuous execution. Training-support, workspace,
  action-range, rate, and observation-age values do not block publication. The
  bridge has neither an Allegro slew limit nor a command expiry watchdog by
  default.
- `Esc`: clear the queue, hold the measured robot configuration, and latch an
  abort. A new `R` is required.

The operator must establish the start pose and retain access to the physical
emergency stop. There is no automatic homing.

## Direct execution and telemetry

Live mode decodes finite ACT outputs and publishes them without dataset bounds,
workspace bounds, rate checks, freshness checks, clipping, or automatic
fault-count latching. `Esc` remains the operator-controlled abort and hold.
If the planner cannot obtain or decode an observation, it retries while the
publisher holds the last target until a new chunk is available.

Every run writes `telemetry.jsonl`. Each inference boundary additionally writes
the paired JPEGs, state, raw 10-action chunk, frame IDs, monotonic timestamps,
and inference timing under:

```text
~/shared_data/inference/act_xarm_allegro/<run-id>/chunk_XXXXXX/
```

Useful overrides include `--camera POLICY_KEY=SERIAL@CAPTURE_PC` (repeat twice),
endpoints, control rate, duration, and chunk limits. Use
`python src/inference/act_xarm_allegro.py --help` for the complete interface.
