# Camera System — API

Method reference for the camera stack: parameters (input) and return values (output)
per class. For the architecture and how these fit together, see the
{doc}`overview <camera_system>`.

Signatures are verified against the code; internal/private methods are omitted.
Each class is collapsed below — click to expand.

:::{dropdown} Common: `start()` parameters
:open:

`start(...)` has the same signature on `Camera`, `CameraLoader`, and
`remote_camera_controller`.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `mode` | `str` | — | `image` / `video` / `stream` / `full` |
| `syncMode` | `bool` | — | use the hardware trigger (wait for pulses) |
| `save_path` | `str` | `None` | relative output dir (video/image); `None` for stream |
| `fps` | `int` | `30` | frame rate |
| `exposure_time` | `float` | `None` | microseconds; `None` → camera.json value |
| `gain` | `float` | `None` | dB; `None` → camera.json value |
:::

:::{dropdown} `remote_camera_controller` (main PC)

Drives every capture PC over ZMQ. `start()`/`stop()` set events; the background
`run()` loop does the actual send.

| Method | Input | Output | Description |
|--------|-------|--------|-------------|
| `remote_camera_controller(name, pc_list=None, auto_reload=False, stall_timeout=3.0)` | `name: str`, `pc_list: list[str]=None`, `auto_reload: bool`, `stall_timeout: float` | instance | Construct; spawns the `run()` thread. `pc_list=None` → all capture PCs. |
| `.start(...)` | see [start() params](#common-start-parameters) | `None` | Begin capture on all PCs (returns once the command response has updated health; raises the initialization error if daemon connection failed). |
| `.stop()` | — | `None` | Stop capture on all PCs (also raises the initialization error if the controller never connected). |
| `.end()` | — | `None` | Release the lock, join the loop thread. |
| `.reload_cameras()` | — | `None` | Ask every daemon to re-init its cameras. |
| `.force_takeover()` | — | `dict` per PC | Grab the lock even if another controller holds it. |
| `.is_error()` | — | `bool` | `True` if any camera reported an error. |
| `.get_status()` | — | `dict` | Live health snapshot: error flag, stalled serials, per-PC status, and the last raw daemon responses. |
:::

:::{dropdown} `CameraLoader` (capture PC)

Owns the local cameras and drives them together.

| Method | Input | Output | Description |
|--------|-------|--------|-------------|
| `CameraLoader(types=["pyspin"])` | `types: list[str]` | instance | Detect and open every local camera. |
| `.start(...)` | see [start() params](#common-start-parameters) | `None` | Start all cameras concurrently (blocks until all started). |
| `.stop()` | — | `None` | Stop all cameras. |
| `.end()` | — | `None` | Stop + release all cameras (DeInit + free SHM). |
| `.get_status_list()` | — | `list[dict]` | Per-camera status dict (see `Camera.get_status`). |
| `.get_summary()` | — | `dict` | Compact daemon payload: expected/detected counts, serials, per-camera states/frame ids/errors. |
| `.get_all_errors()` | — | `dict[str, tuple]` | `{serial: (msg, traceback)}` for cameras in error. |

**Attributes**: `.camera_names` (`list[str]` serials), `.cameralist` (`list[Camera]`).
:::

:::{dropdown} `Camera` (capture PC)

One camera + its capture thread. See the {doc}`lifecycle contract <camera_system>`.

| Method | Input | Output | Description |
|--------|-------|--------|-------------|
| `Camera(cam_type, name, frame_shape=(1536,2048,3), cfg=None)` | `cam_type: str`, `name: str` (serial), `frame_shape: tuple`, `cfg: dict=None` | instance | Open the camera, allocate SHM, spawn the capture thread. `cfg` is the per-serial `camera.json` entry passed down to `PyspinCamera`. |
| `.start(...)` | see [start() params](#common-start-parameters) | `None` | Begin capture in `mode`; blocks until acquiring. |
| `.stop(timeout=5.0)` | `timeout: float` | `None` | Stop capture; finite wait (warns and returns on timeout). |
| `.end(timeout=5.0)` | `timeout: float` | `None` | Stop + release (DeInit + free SHM); finite wait. |
| `.get_status()` | — | `dict` | `{state, frame_id, name, mode, fps, syncMode, save_path, time}`. |
| `.get_state()` | — | `str` | `CONNECTING`/`READY`/`STARTING`/`CAPTURING`/`ERROR`/`STOPPED`. |
| `.get_frame_id()` | — | `int` | Last processed frame id. |
| `.get_error()` | — | `(bool, (str, str))` | `(has_error, (message, traceback))`. |
:::

:::{dropdown} `PyspinCamera` (driver)

Thin wrapper over the PySpin SDK. `Camera` holds one as `self.camera`.

| Method | Input | Output | Description |
|--------|-------|--------|-------------|
| `.get_image()` | — | `(ndarray, dict)`, `(None, dict)`, or `(None, None)` | Grab one frame. `(None, None)` on a grab timeout (`GRAB_TIMEOUT_MS`); `(None, frame_data)` on an incomplete/zero-size frame. `dict = {pc_time, frameID}`. |
| `.start(mode, syncMode, frame_rate=None, gain=None, exposure_time=None)` | `mode: "single"\|"continuous"`, ... | `None` | Re-apply changed config + `BeginAcquisition`. |
| `.stop()` | — | `None` | `EndAcquisition` (camera stays connected). |
| `.release()` | — | `None` | `DeInit` (disconnect). |

Module-level: `get_serial_list()` → `list[str]`, `autoforce_ip()`,
`load_camera(serial, cfg=None)` → `PyspinCamera`,
`load_timestamp_monitor(serial, cfg=None)` → `PyspinTimestampMonitor`, constant
`GRAB_TIMEOUT_MS = 1000`.
:::

:::{dropdown} Implementation notes — the Event handshake

For readers going into `camera.py`: the capture thread and the caller synchronize
through an **Event handshake**, not plain flags.

```python
# run(): the capture thread
while not self.event["exit"].is_set():
    if self.event["start"].is_set():
        self.continuous_acquire() if self.mode in ["full","video","stream"] else self.single_acquire()
    time.sleep(0.001)

# start(): set the event, then wait until the thread actually begins
self.event["start"].set()
self.event["acquisition"].wait()

# continuous_acquire(): the body — grab, route to sinks, exit on stop
self.camera.start("continuous", self.syncMode, self.fps, ...)   # BeginAcquisition
self.event["acquisition"].set()
while self.event["start"].is_set() and not self.event["exit"].is_set():
    frame, frame_data = self.camera.get_image()
    if frame is None: continue          # timeout → re-check the while condition
    if save_video: video_writer.write(frame)
    if stream:     ...                  # SHM double-buffer (write_flag toggles)
self.camera.stop(); self.event["stop"].set()
```

`stop()` clears `event["start"]` and waits on `event["stop"]`; the loop above exits
on its next iteration and sets `event["stop"]`. The P4 hang lived exactly here — a
blocking `get_image()` meant the loop never iterated, so `event["stop"]` was never
set and `stop()` never returned.
:::
