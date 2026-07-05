# Capture Session — API

Method reference for the capture-session subsystem: parameters (input) and return
values (output) per class. For the architecture and how these fit together, see the
{doc}`overview <capture>`.

Signatures are verified against the code; internal/private methods are omitted. All
`save_path` arguments to `CaptureSession` are **relative** to `shared_dir`
(`~/shared_data`); the session prepends it.

---

## `CaptureSession`

`paradex/dataset_acqusition/capture.py`. The single entry point a dataset pipeline
uses. The constructor selects and connects the device set; `start`/`stop` toggle one
recording; `end` releases everything.

| Method | Input | Output | Description |
|--------|-------|--------|-------------|
| `CaptureSession(camera=False, arm=None, hand=None, teleop=None, hand_ip=False)` | `camera: bool`, `arm: str\|None`, `hand: str\|None`, `teleop: str\|None`, `hand_ip: bool` | instance | Build + connect the enabled devices. `camera=True` adds a `UTGE900` sync generator; with an `arm`/`hand` it also adds a `TimestampMonitor`. Raises `ValueError` if `teleop` is set with no `arm` or `hand`. |
| `.start(save_path, mode="video", fps=30, exposure_time=None, gain=None, stage=None)` | `save_path: str` (rel), `mode: "video"\|"image"`, `fps: int`, `exposure_time: float\|None` (µs), `gain: float\|None` (dB), `stage: str\|None` | `None` | Create `raw[/stage]/`, then `start()` every enabled device. Camera is always started with `syncMode=True`; the sync generator fires at `fps`. |
| `.stop()` | — | `None` | Stop every device. Dumps teleop `state/` arrays; on camera sessions saves `cam_param` + `C2R` at the `save_path` root. Clears `save_path`/`stage`. |
| `.end()` | — | `None` | Release every device connection (arm/hand/teleop/camera/timestamp/sync). |
| `.teleop()` | — | `"stop"` \| `"exit"` | Run the XSens retarget loop (see {doc}`overview §6 <capture>`). Raises `ValueError` if no teleop device. Records per-tick state to `state_hist`/`state_time` while a `save_path` is active. |
| `.move(action_dict)` | `action_dict: dict` with optional `"arm"`, `"hand"` keys | `None` | Forward a manual command to the arm and/or hand (no teleop). |

**Constructed attributes** (present only when the matching flag is set): `.camera`
(`remote_camera_controller`), `.sync_generator` (`UTGE900`), `.timestamp_monitor`
(`TimestampMonitor`), `.arm`, `.hand`, `.teleop_device` (`XSensReceiver`),
`.retargetor` (`Retargetor`), `.state_extractor` (`HandStateExtractor`). Unset
devices are `None`. `.save_path` / `.stage` hold the active recording (both `None`
between recordings).

---

## `UTGE900` (hardware sync generator)

`paradex/io/camera_system/signal_generator.py`. Unit-T UTG900 driven over USBTMC;
emits the square wave that hardware-triggers all cameras. Built from
`network_info["signal_generator"]["param"]`.

| Method | Input | Output | Description |
|--------|-------|--------|-------------|
| `UTGE900(addr)` | `addr: str` (device path) | instance | Open the device. Raises `FileNotFoundError` / `PermissionError` if the path is missing or not R/W. |
| `.start(fps, ch=1)` | `fps: int` (Hz), `ch: int` | `None` | Output a `square` wave at `fps` on channel `ch`. No-op if the channel is already on. |
| `.stop(ch=1)` | `ch: int` | `None` | Turn channel `ch` off. No-op if already off. |
| `.generate(ch=1, wave="square", freq=30, amp=10)` | `ch: int`, `wave: str`, `freq: int`, `amp: int` (Vpp) | `None` | Configure and enable a waveform on `ch` (used by `start`). |
| `.end()` | — | `None` | Release the front-panel lock (`System:LOCK off`). |
| `.getName()` | — | `str` | Device `*IDN?` identity string. |

---

## `TimestampMonitor` (frame-id logger)

`paradex/io/camera_system/timestamp_monitor.py`. A dedicated camera that logs only
`(frame_id, pc_time)` so robot streams can be aligned to camera frames. Added
automatically when a camera session also has an arm or hand. Built from
`network_info["timestamp"]["param"]`.

| Method | Input | Output | Description |
|--------|-------|--------|-------------|
| `TimestampMonitor(cam_type, name)` | `cam_type: str` (e.g. `"pyspin"`), `name: str` (serial) | instance | Connect the monitor camera and spawn its capture thread (blocks until connected). |
| `.start(save_path=None)` | `save_path: str\|None` | `None` | Begin logging; when `save_path` is given, accumulate `frame_id`/`timestamp` for dumping. Blocks until acquisition begins. |
| `.stop()` | — | `None` | Stop logging; on stop, writes `frame_id.npy` + `timestamp.npy` into `save_path`. |
| `.end()` | — | `None` | Stop (if running) and release the camera. |
| `.get_data()` | — | `dict` | `{frame_id, time}` — the last logged frame id and pc time. |
| `.is_signal_active(fps)` | `fps: int` | `bool` | `True` if a trigger frame arrives within `1.5 / fps` s. |
| `.wait_signal_inactive(fps)` | `fps: int` | `None` | Block until the trigger stops (no frame within `1.5 / fps` s). |
| `.is_error()` | — | `bool` | `True` if the monitor is in an error state. |
| `.error_reset()` | — | `None` | Clear the error state. |

---

## `match_sync` (post-hoc alignment helpers)

`paradex/dataset_acqusition/match_sync.py`. Module-level functions (not a class) used
in post-processing to align robot streams to logged camera frames.

| Function | Input | Output | Description |
|----------|-------|--------|-------------|
| `fill_framedrop(frame_id, pc_time)` | `frame_id: array`, `pc_time: array` | `(pc_time_nodrop, frame_id_nodrop)` | Reconstruct a drop-free frame-id axis and its pc-time by fitting a constant per-frame delta; skips the first 10 frames. |
| `get_synced_data(pc_times, data, data_times)` | `pc_times: array`, `data: array`, `data_times: array` | `np.ndarray` | Nearest-timestamp two-pointer match of `data` onto `pc_times`. |

---

## Device controllers (constructed by the session)

The session drives its arm/hand/teleop through a shared `start/stop/end` contract.
Built via `get_arm(arm)` / `get_hand(hand, ip=hand_ip)`
(`paradex/io/robot_controller/__init__.py`) and the teleop receiver. Full per-device
APIs live with their own modules; the subset the session calls:

| Method | Input | Output | Called in | Description |
|--------|-------|--------|-----------|-------------|
| `arm.start(save_path)` | `save_path: str` | `None` | `start()` | Begin logging arm state to `raw[/stage]/arm`. |
| `arm.stop()` / `arm.end()` | — | `None` | `stop()` / `end()` | Stop logging / release the arm. |
| `arm.move(action, is_servo=True, speed=None)` | `action`, `is_servo: bool`, `speed` | `None` | `teleop()` / `move()` | Command a pose (servo or blocking). |
| `arm.get_data()` | — | `dict` | `teleop()` | Read current state (`teleop()` reads `["position"]` as home). |
| `hand.start(save_path)` | `save_path: str` | `None` | `start()` | Begin logging hand state to `raw[/stage]/hand`. |
| `hand.stop()` / `hand.end()` | — | `None` | `stop()` / `end()` | Stop logging / release the hand. |
| `hand.move(action)` | `action` | `None` | `teleop()` / `move()` | Command a hand pose. |
| `teleop_device.start(save_path)` | `save_path: str` | `None` | `start()` | Begin logging teleop stream to `raw[/stage]/teleop`. |
| `teleop_device.stop()` / `.end()` | — | `None` | `stop()` / `end()` | Stop logging / release. |
| `teleop_device.get_data()` | — | `dict` | `teleop()` | Latest XSens frame (`Left`/`Right` segment poses). |

---

## Teleop helpers

Used only inside `CaptureSession.teleop()`.

| Method | Input | Output | Description |
|--------|-------|--------|-------------|
| `Retargetor(arm_name, hand_name)` | `arm_name: str`, `hand_name: str` | instance | Human→robot retargeter for the session's arm/hand. |
| `.start(home_pose)` | `home_pose: ndarray` (4×4) | `None` | Anchor retargeting to the arm's current pose. |
| `.get_action(data)` | `data: dict` (XSens frame) | `(wrist_pose, hand_action)` | Retargeted arm pose + hand command. |
| `.stop()` | — | `None` | Pause retargeting (state 1/2). |
| `HandStateExtractor()` | — | instance | Discrete-state classifier for the left hand. |
| `.get_state(pose_data)` | `pose_data` (left-hand pose) | `int` | `0` move / `1` pause / `2` stop / `3` exit. |
