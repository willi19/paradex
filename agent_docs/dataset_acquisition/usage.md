# Dataset Acquisition — How to Use

Two independent tools, used at opposite ends of a capture:

- **`CaptureSession`** — *record time*. One object that starts/stops the camera rig + arm +
  hand + teleop together and writes a fixed `raw/` directory tree.
- **`match_sync`** (`fill_framedrop` + `get_synced_data`) — *process time*. Re-align the
  per-sensor `.npy` logs to the camera frame clock after the session is saved.

> Editing either module (device wiring, the `raw/` layout, the teleop loop, the drop model,
> the matcher)? Read [`internals.md`](internals.md) instead.

Module import path is misspelled on purpose: `paradex.dataset_acqusition` (missing 'i').

---

## CaptureSession

```python
from paradex.dataset_acqusition.capture import CaptureSession

cs = CaptureSession(camera=True, arm="xarm", hand="inspire", teleop="xsens")
#   record one or more sessions:
cs.start("capture/miyungpa/<name>/<ts>")   # save_path is RELATIVE to shared_dir
# ... drive it: cs.teleop()  (teleop loop)  or  cs.move({...})  (scripted) ...
cs.stop()                                   # flush + save calibration snapshot
cs.end()                                    # release every device (camera LOCK included)
```

### Constructor — `CaptureSession(camera=False, arm=None, hand=None, teleop=None, hand_ip=False)`
| param | meaning |
|-------|---------|
| `camera` | `True` → builds `remote_camera_controller("dataset_acquisition")` + `UTGE900` sync generator. If `arm` **or** `hand` is also set, additionally builds a `TimestampMonitor` (so sensor times can be mapped to camera frames later). |
| `arm` | arm name for `get_arm(arm)` (e.g. `"xarm"`), or `None`. |
| `hand` | hand name for `get_hand(hand, ip=hand_ip)` (e.g. `"inspire"`, `"allegro"`), or `None`. |
| `teleop` | `"xsens"` → `XSensReceiver` + `Retargetor(arm, hand)` + `HandStateExtractor`. **Requires `arm` or `hand`** (else `ValueError`). `"occulus"` is *not* implemented. |
| `hand_ip` | passed to `get_hand` (Inspire needs an IP socket). |

Every field you don't ask for stays `None` and is skipped in `start/stop/end` — so
camera-only, arm-only, camera+arm+hand, etc. all work through the same three calls.

### `start(save_path, mode="video", fps=30, exposure_time=None, gain=None, stage=None)`
| param | meaning |
|-------|---------|
| `save_path` | session dir **relative to `shared_dir`** (`~/shared_data`). Convention: `capture/<pipeline>/<name>/<ts>`. |
| `mode` | camera mode — `video` (default) / `image` / `stream` / `full` (see camera `usage.md`). |
| `fps` | frame rate; also fed to the UTGE900 sync generator. |
| `exposure_time`, `gain` | `None` → per-camera `camera.json` baseline (recommended). |
| `stage` | optional sub-bucket: data lands under `raw/<stage>/…` instead of `raw/…` (the upload processor expects this layout when set). |

The camera is **always started with `syncMode=True`** here — a session records against the
hardware trigger, so the UTGE900 must actually be running (it is; `start` fires it).

Written layout under `shared_dir/<save_path>/`:
```
raw[/<stage>]/
  videos/ | images/     # camera (per-serial)
  arm/                  # arm recorder  (time.npy + data .npy)
  hand/                 # hand recorder
  teleop/               # raw XSens stream
  timestamps/           # TimestampMonitor: frame_id.npy, timestamp.npy
state/                  # teleop only: state_hist.npy, state_time.npy
```

### `stop()`
Stops all active sensors. Extra effects:
- **teleop** → dumps `state/state_hist.npy` + `state/state_time.npy`.
- **camera** → stops sync generator + timestamp monitor, then snapshots the *current*
  calibration next to the data: `save_current_camparam(...)` and `save_current_C2R(...)`.
  (So each session carries the calibration it was shot with — downstream `load_camparam(demo_path)`
  reads these back.)

After `stop()` the session is idle; you may `start()` again (e.g. a record loop).

### `end()`
Releases every device and **frees the camera lock** — always call it (or the daemon frees the
lock only after its idle timeout). One `end()` per `CaptureSession` lifetime.

### Driving between start/stop — two options
- **`teleop()`** — blocking XSens retarget loop. Returns `"stop"` or `"exit"` based on the
  operator's hand state (left-hand gesture, held ~90 ticks @ 10 ms). While a `save_path` is
  set it also logs the state stream. Typical record loop (see `src/dataset_acquisition/miyungpa/capture.py`):
  ```python
  while True:
      if cs.teleop() == "exit": break     # "prepare" phase (no save_path yet)
      cs.start("capture/miyungpa/<name>/<ts>")
      state = cs.teleop()                 # "record" phase
      cs.stop()
      if state == "exit": break
  cs.end()
  ```
- **`move(action_dict)`** — scripted control: `{"arm": <pose>, "hand": <action>}` (either key
  optional). Use this when you replay a trajectory instead of teleoperating.

### Minimal camera-only recipe
```python
cs = CaptureSession(camera=True)          # no arm/hand/teleop → no TimestampMonitor
cs.start("capture/object_turntable/<obj>/<ts>")
# ...trigger / wait...
cs.stop(); cs.end()
```

---

## match_sync — align sensor logs to camera frames (post-process)

Used **after** a session is captured, by `src/process/miyungpa/`. Camera frames and the
arm/hand recorders run on different clocks and the camera drops frames; these two functions
reconstruct a per-frame sensor value.

```python
from paradex.dataset_acqusition.match_sync import fill_framedrop, get_synced_data
import numpy as np

frame_id  = np.load(".../raw/timestamps/frame_id.npy")
pc_time   = np.load(".../raw/timestamps/timestamp.npy")
pc_time, frame_id = fill_framedrop(frame_id, pc_time)   # dense, drop-free frame timeline

sensor_t  = np.load(".../raw/arm/time.npy")
data      = np.load(".../raw/arm/position.npy")
synced    = get_synced_data(pc_time, data, sensor_t)    # one row per (reconstructed) frame
np.save(".../arm/position.npy", synced)
```

### `fill_framedrop(frame_id, pc_time) -> (pc_time_nodrop, frame_id_nodrop)`
Fits a linear `frame_id → time` model (skipping the first 10 warmup frames), then emits a
**dense** frame_id timeline `1 … last+500` with reconstructed times, so downstream indexing is
gap-free even where the camera dropped frames.

### `get_synced_data(pc_times, data, data_times) -> np.ndarray`
For each `pc_times[i]` returns the `data` row whose `data_times` is nearest (monotone
two-pointer scan; assumes both arrays are time-sorted). Output length = `len(pc_times)`.

For the full processing flow around these (download → match_sync → overlay → upload) read
[`src/process/miyungpa/CLAUDE.md`](../../src/process/miyungpa/CLAUDE.md).

---

## Gotchas
- Import path is `paradex.dataset_acqusition` (missing 'i') — intentional typo, everywhere.
- `save_path` is **relative to `shared_dir`**, not absolute; `CaptureSession` prepends it.
- The camera records with `syncMode=True` — the UTGE900 trigger must run (start fires it). A
  session started while the signal generator is down produces no frames.
- `teleop="xsens"` needs `arm` or `hand`; `"occulus"` is a valid CLI choice in some scripts but
  **not wired** in `CaptureSession`.
- `stop()` snapshots calibration only when `camera` is on; camera-off sessions save no camparam.
- Always `end()` to release the camera lock; without it, only the daemon's idle timeout frees it.
- `fill_framedrop` hard-codes a `td = 2/30` s trigger-latency offset and skips the first 10
  frames — see [`internals.md`](internals.md) before trusting it on non-30-fps data.
