# Miyungpa Capture

Teleoperated demonstration capture: an operator drives the XArm + hand via an XSens suit while the camera rig records synced video. Produces multi-modal manipulation demos (video + arm + hand + teleop streams).

## Scripts
| File | Purpose |
|------|---------|
| `capture.py` | Builds a `CaptureSession` with camera + arm + hand + teleop, then loops teleop-driven record sessions gated by the teleop state machine. |

## Usage
Run on the **main PC** (capture-PC daemons live; XSens receiver, arm, and hand reachable):
```bash
python src/dataset_acquisition/miyungpa/capture.py --name demo1 --arm xarm --hand allegro --device xsens
```
- `--name` is required; `--arm`, `--hand` are passed to `CaptureSession`; `--device` chooses teleop (`xsens`; `occulus` is parsed but not implemented in the session).

Control is via the teleop hand-state machine (`cs.teleop()`), not the keyboard:
- First `cs.teleop()` lets the operator settle; returns `"stop"` (begin) or `"exit"`.
- `cs.start(...)` then records while the second `cs.teleop()` runs; it returns `"stop"` (end this clip) or `"exit"` (quit).
- Loop repeats, one timestamped clip per cycle, until `"exit"`.

## Inputs & Outputs
- **Hardware:** multi-camera rig + XArm + hand (e.g. Allegro) + XSens teleop + UTGE900 sync + TimestampMonitor (arm/hand present).
- **Output:** `~/shared_data/capture/miyungpa/<name>/<timestamp>/raw/` with `videos/`, `arm/`, `hand/`, `teleop/`, `timestamps/`, and `state/state_hist.npy` + `state/state_time.npy`; camparam + C2R saved on stop.

## Related
- [`paradex/dataset_acqusition/capture.py`](../../../paradex/dataset_acqusition/capture.py) — `CaptureSession`, `.teleop()` retarget loop.
- [`paradex/io/teleop/xsens/receiver.py`](../../../paradex/io/teleop/xsens/receiver.py), [`paradex/retargetor/`](../../../paradex/retargetor).
- Post-processing: [`src/process/miyungpa/`](../../process/miyungpa).
