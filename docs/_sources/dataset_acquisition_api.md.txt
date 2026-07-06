# Dataset Acquisition — API

Method reference for the dataset-acquisition layer: parameters (input) and return values
(output). For the architecture and how these fit together, see the {doc}`overview <dataset_acquisition>`.

Signatures are verified against the code. The subsystem lives in `paradex/dataset_acqusition/`
(module directory misspelled on purpose — missing an 'i').

Each entry is collapsed below — click to expand.

:::{dropdown} `CaptureSession` (`paradex/dataset_acqusition/capture.py`)
:open:

Multiplexes the camera rig + arm + hand + teleop through one lifecycle and writes the `raw/`
tree. Only the devices you request are built; the rest stay `None` and are skipped everywhere.

| Method | Input | Output | Description |
|--------|-------|--------|-------------|
| `CaptureSession(camera=False, arm=None, hand=None, teleop=None, hand_ip=False)` | `camera: bool`, `arm: str\|None`, `hand: str\|None`, `teleop: str\|None`, `hand_ip: bool` | instance | Build requested devices. `camera=True` → `remote_camera_controller` + `UTGE900`; plus a `TimestampMonitor` **iff** `arm` or `hand`. `teleop="xsens"` → `XSensReceiver` + `Retargetor(arm, hand)` + `HandStateExtractor`, and **requires** `arm` or `hand` (else `ValueError`). `"occulus"` is not implemented. |
| `.start(save_path, mode="video", fps=30, exposure_time=None, gain=None, stage=None)` | `save_path: str` (relative to `shared_dir`), `mode: str`, `fps: int`, `exposure_time`, `gain`, `stage: str\|None` | `None` | Make `raw[/<stage>]/`, start each recorder into its subdir, start the camera with **`syncMode=True`**, fire the sync generator **last**. `exposure_time`/`gain` `None` → per-camera `camera.json` baseline. |
| `.stop()` | — | `None` | Stop all sensors. teleop → dump `state/state_hist.npy` + `state_time.npy`. camera → stop monitor + generator, then `save_current_camparam(...)` + `save_current_C2R(...)`. Clears `save_path`/`stage` (re-arms). |
| `.end()` | — | `None` | End every device; `camera.end()` frees the daemon lock. Call once per session lifetime. |
| `.teleop()` | — | `str` | Blocking XSens retarget loop; returns `"stop"` or `"exit"` when the operator holds that hand state `>90` ticks. Logs state only while `save_path` is set. |
| `.move(action_dict)` | `action_dict: dict` (`{"arm": …, "hand": …}`, keys optional) | `None` | Scripted single step — `arm.move` / `hand.move`. No `None`-guard: passing a device the session lacks raises. |

**Attributes**: `.camera`, `.arm`, `.hand`, `.teleop_device`, `.sync_generator`,
`.timestamp_monitor`, `.retargetor`, `.state_extractor` (each `None` if not built),
`.save_path`, `.stage`, `.state_hist`, `.state_time`.

**Ordering invariant** (do not reorder): trigger consumers (`camera`, `timestamp_monitor`) are
armed **before** `sync_generator.start()` and torn down **before** `sync_generator.stop()` — the
trigger is last-on / last-off at both ends. See overview §4.
:::

:::{dropdown} `match_sync` module functions (`paradex/dataset_acqusition/match_sync.py`)

Post-hoc alignment of per-sensor logs to the camera frame clock. Used by `src/process/miyungpa/`.

| Function | Input | Output | Description |
|----------|-------|--------|-------------|
| `fill_framedrop(frame_id, pc_time)` | `frame_id: np.ndarray`, `pc_time: array-like` | `(pc_time_nodrop, frame_id_nodrop)` | Skip first 10 warmup frames, fit `time_delta = Δpc_time / Δframe_id` and an intercept `offset`, then emit a dense `frame_id_nodrop = arange(1, last+500)` with `pc_time_nodrop = (frame_id-1)*time_delta + offset - td`. Robust to drops (divides by frame-id span, not count). |
| `get_synced_data(pc_times, data, data_times)` | `pc_times: array`, `data: np.ndarray`, `data_times: array` | `np.ndarray` (len = `len(pc_times)`) | For each `pc_times[i]`, the `data` row whose `data_times` is nearest (monotone two-pointer). **Assumes both time arrays are sorted ascending.** Nearest-neighbor, not interpolation. |

**Module constant**: `td = 2 / 30` — a fixed ~66 ms trigger-to-exposure latency offset baked for
**30 fps**. Not parameterized; wrong at other frame rates.
:::

:::{dropdown} Device dependencies (documented for signature completeness)

Built by `CaptureSession`; full APIs live in their own subsystems.

| Symbol | Location | Role |
|--------|----------|------|
| `remote_camera_controller` | `paradex/io/camera_system/remote_camera_controller.py` | Main-PC camera driver (see {doc}`camera_system_api`). |
| `UTGE900` | `paradex/io/camera_system/signal_generator.py` | Hardware sync trigger; `.start(fps)` / `.stop()` / `.end()`. |
| `TimestampMonitor` | `paradex/io/camera_system/timestamp_monitor.py` | Records `frame_id.npy` + `timestamp.npy`. |
| `get_arm` / `get_hand` | `paradex/io/robot_controller/` | Arm/hand handles with `.start/.stop/.end/.move/.get_data`. |
| `XSensReceiver` | `paradex/io/teleop/xsens/receiver.py` | Teleop stream (`.get_data()` → `{'Left','Right',…}`). |
| `Retargetor`, `HandStateExtractor` | `paradex/retargetor/` | Human→robot retarget; left-hand state classifier. |
| `save_current_camparam`, `save_current_C2R` | `paradex/calibration/utils.py` | Per-dataset calibration snapshot on `stop()`. |
:::
