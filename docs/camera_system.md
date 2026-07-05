# Camera System

Engineering reference for Paradex's camera subsystem: architecture, components,
state flow, configuration, and error handling. Read this before the code to get the
overall structure.

- Redesign proposal: [design/camera-recording-redesign.md](https://github.com/willi19/paradex/blob/main/design/camera-recording-redesign.md)
- Per-symbol API: {doc}`API Reference <autoapi/index>`

---

## 1. Overview

Cameras are spread across **6 capture PCs** (one GigE camera nearly saturates a NIC,
so they can't all hang off one host). The **main PC** never touches a camera
directly — it sends commands to a daemon on each capture PC. Every camera fires on
one shared **hardware trigger (UTGE900)** for synchronized capture.

The stack has five layers; commands flow top to bottom.

```{mermaid}
flowchart LR
    subgraph Main["Main PC"]
      ORCH["capture / inference"]
      RCC["remote_camera_controller"]
      ORCH --> RCC
    end
    subgraph Cap["Capture PC (×6)"]
      D["server_daemon"]
      CL["CameraLoader"]
      CAM["Camera × k → PyspinCamera"]
      D --> CL --> CAM
    end
    GEN["UTGE900"]
    RCC -- "ZMQ: register/start/stop/heartbeat" --> D
    GEN -. "hardware trigger" .-> CAM
```

| Layer | Component | Location | Responsibility |
|-------|-----------|----------|----------------|
| Control | `remote_camera_controller` | Main PC | Fan out commands, keep heartbeat |
| Service | `camera_server_daemon` | Capture PC | Receive/dispatch commands, controller lock |
| Group | `CameraLoader` | Capture PC | Drive N cameras together, resolve parameters |
| Device | `Camera` | Capture PC | Capture-thread state machine, sink routing |
| Driver | `PyspinCamera` | Capture PC | PySpin SDK calls |

---

## 2. Core Concepts

| Term | Meaning |
|------|---------|
| **Mode** | `image` / `video` / `stream` / `full`. Selects the capture method *and* where frames go. |
| **Sink** | A frame's destination. `stream`→SHM, `video`→`.avi`, `image`→one still, `full`→SHM+`.avi`. |
| **Acquisition** | One capture thread per camera grabbing frames continuously. A separate axis from the sink. |
| **Hardware sync** | Each trigger pulse increments every camera's `frame_id` at once. Same instant = same id. |
| **Controller lock** | One controller per daemon. Acquired via `register`; auto-released after 15 s without a heartbeat. |

---

## 3. Components

### 3.1 `remote_camera_controller` (main PC)

- **Responsibility**: fan commands out to the capture PCs in parallel and keep the connection alive with heartbeats.
- **Key methods**: `initialize()` (ping check, sockets, lock), `start(mode, sync, ...)`, `stop()`, `end()`, `run()` (background loop).
- **Interface**: ZMQ REQ → each daemon's `command_port (5482)`; liveness via `ping_port (5480)`.
- **Note**: `start()`/`stop()` only **set events** — the actual command send happens in the `run()` loop.

```python
def run(self):
    self.initialize()
    while not self.exit_event.is_set():
        cmd = {'action': 'heartbeat'}
        if self.start_event.is_set(): cmd = {'action': 'start', 'mode': ..., ...}
        if self.stop_event.is_set():  cmd = {'action': 'stop'}
        response = self.send_command(cmd)     # one thread per PC, sent concurrently
        time.sleep(0.1)
```

### 3.2 `camera_server_daemon` (capture PC)

- **Responsibility**: receive commands → dispatch via `execute_command` → call `CameraLoader`. Manage the single-controller lock.
- **Ports**: `ping 5480` (REP), `monitor 5481` (PUB status), `command 5482` (REP).
- **Commands**: `register` / `start` / `stop` / `heartbeat` / `reload` / `end`.
- **Lock + timeout**: the command socket uses `RCVTIMEO 15s`. After 15 s with no command it releases the lock and stops the cameras (in case the controller died).

```python
self.command_socket.setsockopt(zmq.RCVTIMEO, 15000)   # 15 s
while True:
    try:
        cmd = self.command_socket.recv_json()
        resp = self.execute_command(cmd)              # register/start/stop/heartbeat/...
        self.command_socket.send_json(resp)
    except zmq.Again:                                 # 15 s idle → release lock + stop cameras
        ...
```

### 3.3 `CameraLoader` (capture PC)

- **Responsibility**: start/stop N cameras concurrently (via threads). Resolve gain/exposure **per camera**.
- **Resolution rule**: `explicit arg > camera.json[serial] > default`. `None` means "use camera.json", not "keep whatever was last set" (§7).

```python
for camera, path in zip(self.cameralist, save_paths):
    cfg = self.cam_config.get(camera.name, {})
    e = exposure_time if exposure_time is not None else cfg.get("exposure", DEFAULT_EXPOSURE)
    g = gain          if gain          is not None else cfg.get("gain", DEFAULT_GAIN)
    Thread(target=camera.start, args=(mode, syncMode, path, fps, e, g)).start()
```

`stop`/`end` reuse the same fan-out helper (`_broadcast`).

### 3.4 `Camera` (capture PC) — state machine

- **Threading model**: on construction it spawns one capture thread (`run`). That thread and the outside caller synchronize through a **set of Events** (a handshake, not just flags).
- **States** (from `get_state()`):

```{mermaid}
stateDiagram-v2
    [*] --> CONNECTING
    CONNECTING --> READY: connect_camera()
    READY --> STARTING: start()
    STARTING --> CAPTURING: acquisition set
    CAPTURING --> READY: stop()
    CAPTURING --> ERROR: exception
    ERROR --> READY: error_reset()
    READY --> STOPPED: end()
    STOPPED --> [*]
```

- **Core loops**: `continuous_acquire` (stream/video/full), `single_acquire` (image).

```python
# run(): capture while 'start' is set, exit when 'exit' is set
while not self.event["exit"].is_set():
    if self.event["start"].is_set():
        self.continuous_acquire() if self.mode in ["full","video","stream"] else self.single_acquire()
    time.sleep(0.001)

# start(): set the event, then wait until the thread actually begins (handshake)
self.event["start"].set()
self.event["acquisition"].wait()

# continuous_acquire(): the body
self.camera.start("continuous", self.syncMode, self.fps, ...)   # BeginAcquisition
self.event["acquisition"].set()
while self.event["start"].is_set() and not self.event["exit"].is_set():
    frame, frame_data = self.camera.get_image()
    if frame is None: continue          # timeout → re-check the while condition
    if save_video: video_writer.write(frame)
    if stream:     ...                  # SHM double-buffer (write_flag toggles)
self.camera.stop(); self.event["stop"].set()
```

### 3.5 `PyspinCamera` (capture PC) — driver

- **Responsibility**: direct PySpin SDK calls. `get_image()` (grab), `start()` (configure + `BeginAcquisition`), `_configure*` (gain/exposure/trigger/framerate).

```python
def get_image(self):
    try:
        pImageRaw = self.cam.GetNextImage(GRAB_TIMEOUT_MS)   # finite timeout (§8)
    except ps.SpinnakerException:
        return None, None                                    # no frame
    ...
    return frame, frame_data

def start(self, mode, syncMode, frame_rate=None, gain=None, exposure_time=None):
    if syncMode:                       self._configureTrigger()   # only re-apply what changed
    if gain     != self.gain:          self._configureGain()
    if exposure != self.exposure_time: self._configureExposure()
    self.cam.BeginAcquisition()
```

---

## 4. Command Flow

```{mermaid}
sequenceDiagram
    participant M as Main PC (controller)
    participant D as Capture PC (daemon)
    M->>D: register (acquire lock)
    M->>D: start(mode, sync, save_path, fps, exposure, gain)
    D->>D: CameraLoader.start → per-camera acquisition
    loop every ~0.1 s
      M->>D: heartbeat
      D-->>M: ok / camera errors
    end
    M->>D: stop
    M->>D: end (release lock)
```

---

## 5. Data Path: Acquisition → Sinks

One capture thread per camera grabs frames and routes them to sinks by mode.
"Producing a frame" and "where it goes" are separate axes, but today `mode` couples
them (the redesign target).

```{mermaid}
flowchart TD
    T["continuous_acquire()"] --> G["get_image()"]
    G -->|frame| SHM["SHM double-buffer (stream/full)"]
    G -->|frame| VID["VideoWriter .avi (video/full)"]
    G -.->|none → (None,None)| T
    SHM --> R["MultiCameraReader → consumers"]
```

---

## 6. Hardware Sync

Each trigger pulse increments every camera's `frame_id` at once, so at any instant
the cameras should hold the same id. `sync_check.py` verifies this via the
`frame_id` spread (max−min) across cameras: 0–1 is in sync; a persistent/growing
spread means frame drops or a camera not receiving the trigger.

```{mermaid}
flowchart LR
    GEN["trigger #N"] --> C1["cam A #N"]
    GEN --> C2["cam B #N"]
    GEN --> C3["cam C #N"]
    C1 & C2 & C3 --> CHK["spread = max-min<br/>0-1 = OK"]
```

---

## 7. Configuration — Gain / Exposure

Per-camera baselines live in `system/current/camera.json`. Resolution is always:

```{mermaid}
flowchart LR
    A["start(exposure=None, gain=None)"] --> B{"explicit arg?"}
    B -->|yes| U["use it"]
    B -->|no| C["camera.json[serial]"]
    C -->|missing| D["default 2500us / 3dB"]
```

`None` means "use the camera.json value", so a one-off override (e.g. an exposure
sweep) never leaks into the next capture. Tune live with
`src/util/camera_tuning/live_tuner.py`.

---

## 8. Error Handling & Recovery

**Frame-loss hang (P4).** On a LAN drop / stopped trigger, the old `get_image()`
blocked forever. The capture thread stalled in the §3.4 loop, never re-checked the
`while` condition, so `event["stop"]` was never set and `stop()` never returned —
wedging the whole daemon (cameras couldn't restart even after `pkill`).

```{mermaid}
flowchart LR
    subgraph Before["Before"]
      B1["GetNextImage() blocks forever"] --> B2["stop() never returns → daemon wedged"]
    end
    subgraph After["After (P4)"]
      A1["GetNextImage(1000ms) → (None,None)"] --> A2["loop re-checks → stop()/end() return"]
    end
```

**Recovery**: if wedged at the hardware level, run `python src/camera/reset_cameras.py`
from the main PC (`pkill -9` + relaunch the daemons). **Validation**: §9.

---

## 9. Validation

| Script | Verifies | Hardware |
|--------|----------|----------|
| `src/validate/camera_system/hang_recovery.py` | stop/end don't hang on frame loss (watchdog flags a hang as FAIL) | required |
| `src/validate/camera_system/hang_recovery_mock.py` | `get_image` uses a finite timeout and returns `(None,None)` | not required |
| `src/validate/camera_system/sync_check.py` | `frame_id` alignment across cameras | required |

---

## 10. File Reference

| Target | File |
|--------|------|
| Main-PC control | `paradex/io/camera_system/remote_camera_controller.py` |
| Capture-PC daemon | `paradex/io/camera_system/camera_server_daemon.py` |
| Group control / parameter resolution | `paradex/io/camera_system/camera_loader.py` |
| Capture thread / sinks | `paradex/io/camera_system/camera.py` (`continuous_acquire`) |
| PySpin driver | `paradex/io/camera_system/pyspin.py` |
| Live tuner | `src/util/camera_tuning/live_tuner.py` |
| Recover wedged cameras | `src/camera/reset_cameras.py` |
