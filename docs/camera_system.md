# Camera System

A visual tour of how Paradex's **camera subsystem** is implemented — read this
**before** diving into the code so the files make sense. (Camera is just one
subsystem of the wider rig; robot / capture / pipeline guides can sit alongside
this one under "Guide".) For the full redesign proposal see
[`design/camera-recording-redesign.md`](https://github.com/willi19/paradex/blob/main/design/camera-recording-redesign.md);
for the per-symbol API see the {doc}`API Reference <autoapi/index>`.

## Big picture — a distributed rig

Cameras are spread across ~6 **capture PCs** (bandwidth: one GigE camera nearly
fills a NIC). The **main PC** never touches a camera directly — it sends commands to
a daemon on each capture PC. All cameras fire on one shared **hardware trigger** so
frames are synchronized regardless of which PC they hang off.

```{mermaid}
flowchart LR
    subgraph Main["Main PC"]
      ORCH["capture / inference<br/>scripts"]
      RCC["remote_camera_controller"]
      ORCH --> RCC
    end
    subgraph Cap["Capture PC (×6)"]
      D["server_daemon"]
      CL["CameraLoader"]
      CAM["Camera × k (PySpin)"]
      D --> CL --> CAM
    end
    GEN["UTGE900<br/>signal generator"]
    RCC -- "ZMQ: register / start / stop / heartbeat" --> D
    GEN -. "hardware trigger (electrical)" .-> CAM
```

## Talking to a capture PC

The controller holds a **single-controller lock** on each daemon. It starts a mode,
then sends heartbeats; if it goes silent for 15 s the daemon auto-releases and stops
the cameras (a safety net — but see the hang section below).

```{mermaid}
sequenceDiagram
    participant M as Main PC (controller)
    participant D as Capture PC (daemon)
    M->>D: register (take lock)
    M->>D: start(mode, sync, save_path, fps, exposure, gain)
    D->>D: CameraLoader.start → per-camera acquisition
    loop every ~0.1 s
      M->>D: heartbeat
      D-->>M: ok  /  camera errors
    end
    M->>D: stop
    M->>D: end (release lock)
```

## Inside one camera: acquisition vs. sinks

This is the key mental model. Each camera runs **one capture thread** that grabs
frames in a loop; where those frames go (the *sinks*) depends on the mode:

- **stream** → a double-buffered **shared-memory** block that live consumers read
  (e.g. 6D-pose init) via `MultiCameraReader`.
- **video** / **full** → an `.avi` on disk (`full` = video **and** SHM at once).
- **image** → one still frame.

```{mermaid}
flowchart TD
    T["capture thread<br/>continuous_acquire()"] --> G["get_image()<br/>GetNextImage(GRAB_TIMEOUT_MS)"]
    G -->|frame| SHM["SHM double-buffer<br/>(stream / full)"]
    G -->|frame| VID["cv2.VideoWriter .avi<br/>(video / full)"]
    G -.->|no frame → (None, None)| T
    SHM --> R["MultiCameraReader → consumers"]
    VID --> DISK["capture disk"]
```

> The `mode` argument couples *what the camera does* with *where frames go*. The
> [redesign](https://github.com/willi19/paradex/blob/main/design/camera-recording-redesign.md)
> proposes splitting these so recording can toggle without restarting acquisition.

## Hardware sync = matching frame ids

Every trigger pulse increments the frame id on **all** cameras at once, so at any
instant they should report the same id. `sync_check.py` verifies this by measuring
the spread of frame ids across cameras — a persistent spread means a camera is
dropping or isn't triggered.

```{mermaid}
flowchart LR
    GEN["trigger pulse #N"] --> C1["cam A → frame #N"]
    GEN --> C2["cam B → frame #N"]
    GEN --> C3["cam C → frame #N"]
    C1 & C2 & C3 --> CHK["spread = max-min id<br/>0-1 = in sync"]
```

## Why cameras used to hang (and the fix)

When frames stopped arriving (LAN drop, trigger off) the old `get_image()` blocked
**forever**, so the whole daemon wedged and the next run couldn't start cameras —
even after `pkill`.

```{mermaid}
flowchart TD
    subgraph Before["Before (the bug)"]
      A["LAN drop / trigger off"] --> B["GetNextImage() blocks forever"]
      B --> C["loop never re-checks stop"]
      C --> E["Camera.stop() waits forever → daemon hangs"]
    end
    subgraph After["After (P4 fix)"]
      A2["LAN drop / trigger off"] --> B2["GetNextImage(1000ms) → (None, None)"]
      B2 --> C2["loop re-checks start/exit"]
      C2 --> E2["stop()/end() return (finite waits)"]
    end
```

Recovery if a camera is still wedged at the hardware level: run
`python src/camera/reset_cameras.py` from the main PC (force-kills + relaunches the
daemons). Validate the fix with `src/validate/camera_system/hang_recovery.py`.

## Where gain / exposure come from

`system/current/camera.json` holds a per-camera baseline. Resolution is
deterministic — an explicit argument wins, otherwise the camera.json value, so a
one-off override (e.g. an exposure sweep) can't leak into the next capture. Tune it
live with `src/util/camera_tuning/live_tuner.py`.

```{mermaid}
flowchart LR
    A["start(exposure=None, gain=None)"] --> B{"explicit arg?"}
    B -->|yes| U["use arg"]
    B -->|no| C["camera.json[serial]"]
    C -->|missing| D["default 2500us / 3dB"]
```

## Where to look

| I want to… | File |
|-----------|------|
| Drive cameras from the main PC | `paradex/io/camera_system/remote_camera_controller.py` |
| Understand the capture-PC daemon | `paradex/io/camera_system/camera_server_daemon.py` |
| See the acquisition loop & sinks | `paradex/io/camera_system/camera.py` (`continuous_acquire`) |
| Low-level PySpin (grab, config, trigger) | `paradex/io/camera_system/pyspin.py` |
| Multi-camera image resolution | `paradex/io/camera_system/camera_loader.py` |
| Validate the hang fix / sync | `src/validate/camera_system/{hang_recovery,sync_check}.py` |
| Tune gain/exposure live | `src/util/camera_tuning/live_tuner.py` |
| Recover hung cameras | `src/camera/reset_cameras.py` |
