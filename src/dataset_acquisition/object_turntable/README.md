# Object Turntable Capture

Camera-only, keyboard-triggered capture of an object on a turntable from the multi-camera rig. Produces synced video used downstream for COLMAP reconstruction, masking, and 6D/rotation estimation.

## Scripts
| File | Purpose |
|------|---------|
| `capture.py` | Camera-only `CaptureSession`; keyboard events start/stop/quit one recording per turntable sweep. |

## Usage
Run on the **main PC** (capture-PC daemons live):
```bash
python src/dataset_acquisition/object_turntable/capture.py --name mug
```

Keyboard controls (via `listen_keyboard`):
- `c` — begin a new recording session.
- `s` — stop the current recording.
- `q` — quit and release devices.

Typical flow: place object, press `c`, rotate the turntable through a full revolution, press `s`. Repeat for additional objects/poses.

## Inputs & Outputs
- **Hardware:** multi-camera rig only (camera=True; UTGE900 sync, no arm/hand/teleop, no TimestampMonitor).
- **Output:** `~/shared_data/capture/object_turntable/<name>/<timestamp>/raw/videos/...` plus saved camparam + C2R on each `stop()`.

## Related
- [`paradex/dataset_acqusition/capture.py`](../../../paradex/dataset_acqusition/capture.py) — `CaptureSession`.
- [`paradex/utils/keyboard_listener.py`](../../../paradex/utils/keyboard_listener.py) — `listen_keyboard`.
- Post-processing: [`src/process/object_turntable/`](../../process/object_turntable) (`extract_video.py`, `generate_colmap.py`, `extract_mask_sam3.py`, `get_rotation.py`).
