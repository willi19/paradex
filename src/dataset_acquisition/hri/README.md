# HRI Capture

Camera-only, keyboard-triggered recording of human-robot-interaction sessions. Produces multi-camera synced video clips you start/stop by hand, one clip per keypress cycle.

## Scripts
| File | Purpose |
|------|---------|
| `capture.py` | Camera-only `CaptureSession`; keyboard events start/stop/quit recording sessions in a loop. |

## Usage
Run on the **main PC** (capture-PC daemons must be live):
```bash
python src/dataset_acquisition/hri/capture.py --name session1
```
(`--device`, `--arm`, `--hand` args are parsed but currently unused — the session is built camera-only.)

Keyboard controls (via `listen_keyboard`):
- `c` — begin a new recording session (sets save event).
- `s` — stop the current recording.
- `q` — quit the loop and release devices.

Each `c` creates a new timestamped clip; press `s` to stop, then `c` again for the next.

## Inputs & Outputs
- **Hardware:** multi-camera rig only (camera=True; UTGE900 sync generator, no arm/hand/teleop, so no TimestampMonitor).
- **Output:** `~/shared_data/capture/hri_mingi/<name>/<timestamp>/raw/videos/...` plus saved camparam + C2R on each `stop()`.

## Related
- [`paradex/dataset_acqusition/capture.py`](../../../paradex/dataset_acqusition/capture.py) — `CaptureSession`.
- [`paradex/utils/keyboard_listener.py`](../../../paradex/utils/keyboard_listener.py) — `listen_keyboard`.
- Video upload: [`src/util/upload_video/`](../../util/upload_video).
