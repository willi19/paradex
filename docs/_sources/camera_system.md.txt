# Camera System

Overview of Paradex's camera subsystem — the architecture and how the pieces fit.
Read this to build the mental model; for method signatures, parameters, and return
values see the {doc}`API reference <camera_system_api>`.

- Redesign proposal: [design/camera-recording-redesign.md](https://github.com/willi19/paradex/blob/main/design/camera-recording-redesign.md)
- Generated per-symbol API: {doc}`API Reference <autoapi/index>`

---

## 1. Architecture

Cameras are spread across **6 capture PCs** (one GigE camera nearly saturates a NIC,
so they can't all hang off one host). The **main PC** never touches a camera
directly — it sends commands to a daemon on each capture PC. Every camera fires on
one shared **hardware trigger (UTGE900)** for synchronized capture.

The stack has five layers; commands flow top to bottom.

```{mermaid}
flowchart TB
    subgraph Main["Main PC"]
      ORCH["capture / inference"] --> RCC["remote_camera_controller"]
    end
    subgraph Cap["Capture PC (×6)"]
      D["server_daemon"] --> CL["CameraLoader"] --> CAM["Camera × k<br/>→ PyspinCamera"]
    end
    GEN["UTGE900<br/>trigger"]
    RCC -- "ZMQ<br/>register / start /<br/>stop / heartbeat" --> D
    GEN -. "hardware<br/>trigger" .-> CAM
```

| Layer | Component | Location | Responsibility |
|-------|-----------|----------|----------------|
| Control | `remote_camera_controller` | Main PC | Fan out commands, keep heartbeat |
| Service | `camera_server_daemon` | Capture PC | Receive/dispatch commands, controller lock |
| Group | `CameraLoader` | Capture PC | Drive N cameras together, resolve parameters |
| Device | `Camera` | Capture PC | Capture-thread state machine, sink routing |
| Driver | `PyspinCamera` | Capture PC | PySpin SDK calls |

Method-level details for each component are in the {doc}`API reference <camera_system_api>`.

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

## 3. Command Flow

The controller holds a single-controller lock on each daemon, starts a mode, then
keeps sending heartbeats. If it goes silent for 15 s the daemon releases the lock and
stops the cameras.

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

## 4. Data Path: Acquisition → Sinks

One capture thread per camera grabs frames and routes them to sinks by mode.
"Producing a frame" and "where it goes" are separate axes, but today `mode` couples
them (the [redesign](https://github.com/willi19/paradex/blob/main/design/camera-recording-redesign.md)
splits them so recording can toggle without restarting acquisition).

```{mermaid}
flowchart TD
    T["continuous_acquire()"] --> G["get_image()"]
    G -->|frame| SHM["SHM double-buffer<br/>stream / full"]
    G -->|frame| VID["VideoWriter .avi<br/>video / full"]
    G -.->|"no frame"| T
    SHM --> R["MultiCameraReader<br/>consumers"]
```

---

## 5. Camera State Machine

A `Camera` runs one capture thread and coordinates with the caller through Events.
`get_state()` reports where it is:

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

Per-camera baselines live in `system/current/camera.json`. Resolution is always
`explicit arg > camera.json[serial] > default`, so a one-off override (e.g. an
exposure sweep) never leaks into the next capture. Tune live with
`src/util/camera_tuning/live_tuner.py`.

```{mermaid}
flowchart LR
    A["start(exposure=None, gain=None)"] --> B{"explicit arg?"}
    B -->|yes| U["use it"]
    B -->|no| C["camera.json[serial]"]
    C -->|missing| D["default 2500us / 3dB"]
```

---

## 8. Error Handling & Recovery

**Frame-loss hang (P4).** On a LAN drop / stopped trigger, the old `get_image()`
blocked forever; the capture thread stalled and never re-checked its stop condition,
so `stop()` never returned and the whole daemon wedged (cameras couldn't restart even
after `pkill`). The fix gives `get_image()` a finite timeout so the loop stays
responsive.

```{mermaid}
flowchart TB
    subgraph Before["Before"]
      B1["GetNextImage()<br/>blocks forever"] --> B2["stop() never returns<br/>→ daemon wedged"]
    end
    subgraph After["After (P4)"]
      A1["GetNextImage(1000ms)<br/>→ (None,None)"] --> A2["loop re-checks<br/>→ stop()/end() return"]
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

Method-by-method API (parameters / returns): {doc}`Camera System — API <camera_system_api>`.
