# Camera System

This is the canonical camera-system guide. Read this first when you need to know
where camera code runs, which API owns a lifecycle step, or how to recover a rig.
For method signatures, see {doc}`Camera System API <camera_system_api>`. For the
future acquisition/recording redesign, see
`design/camera-recording-redesign.md`; that file is a proposal, not the current
runtime contract.

---

## 1. Quick Mental Model

The main PC never opens FLIR cameras. It sends ZMQ commands to long-running
daemons on the capture PCs. The daemons own the hardware.

```text
Main PC app / pipeline
  -> remote_camera_controller
    -> ZMQ register/start/stop/heartbeat/end
      -> capture PC server_daemon
        -> CameraLoader
          -> Camera
            -> PyspinCamera
              -> FLIR hardware
```

Normal distributed capture therefore has two deployment surfaces:

| Surface | Runs where | Owns |
|---------|------------|------|
| `remote_camera_controller` | main PC | command fan-out, heartbeat, stall detection |
| `server_daemon.py` | every capture PC | camera lock, `CameraLoader`, hardware lifecycle |

Local/single-PC validation scripts can instantiate `CameraLoader` or `Camera`
directly, but production capture goes through the daemon.

---

## 2. Which File Runs Where

Use this table before opening code.

| Situation | Entry point | Location | What it touches |
|-----------|-------------|----------|-----------------|
| Capture images/video from the rig | `src/capture/camera/*_remote.py` or `CaptureSession` users | main PC | `remote_camera_controller` only |
| Keep capture PCs ready | `src/camera/server_daemon.py` | capture PC | `CameraLoader -> Camera -> PyspinCamera` |
| Preview camera health | `src/camera/monitor_daemon.py` | capture PC | status only |
| Reset wedged capture PCs | `src/camera/reset_cameras.py` | main PC | SSH kill/relaunch of daemons |
| Single-PC camera validation | `src/validate/camera_system/*` | local/capture PC | direct `Camera` / `CameraLoader` |
| Live tuning | `src/util/camera_tuning/live_tuner.py` | local/capture PC | direct `PyspinCamera` |

Core implementation files:

| Layer | File | Responsibility |
|-------|------|----------------|
| Main-PC control | `paradex/io/camera_system/remote_camera_controller.py` | command thread, heartbeat, stall/error state |
| Capture-PC service | `paradex/io/camera_system/camera_server_daemon.py` | ZMQ server, controller lock, dead-man timeout |
| Camera group | `paradex/io/camera_system/camera_loader.py` | enumerate cameras, per-serial config, fan-out lifecycle |
| Camera thread | `paradex/io/camera_system/camera.py` | state machine, start/stop/end, image/video/SHM sinks |
| PySpin wrapper | `paradex/io/camera_system/pyspin.py` | SDK node config, grab timeout, hardware stop/release |

---

## 3. Lifecycle Contract

The lifecycle is split deliberately. Do not use heartbeat as a substitute for
command success.

| Step | Owner | Contract |
|------|-------|----------|
| `register` | daemon | acquire the single-controller lock; stop prior running cameras on takeover |
| `start` | daemon + `CameraLoader` | arm every detected camera or return `status="error"` with per-camera detail |
| `heartbeat` | controller loop + daemon | report ongoing `running`, camera counts, states, frame ids, and errors |
| `stop` | daemon + `Camera` | bounded stop; do not wedge the daemon if a camera is slow |
| `end` | controller + daemon | stop running cameras if needed, release the controller lock |
| idle timeout | daemon | if no command/heartbeat arrives within `PARADEX_CAMERA_IDLE_TIMEOUT_S` (default 5s), stop cameras and release lock |

Current command flow:

```{mermaid}
sequenceDiagram
    participant App as Main PC app
    participant RCC as remote_camera_controller
    participant D as capture-PC daemon
    participant CL as CameraLoader
    participant C as Camera/PyspinCamera

    App->>RCC: start(mode, sync, save_path, fps, exposure, gain)
    RCC->>D: register
    RCC->>D: start(...)
    D->>CL: start(...)
    CL->>C: per-camera start(...)
    D-->>RCC: ok/error + counts + states + frame_ids
    loop every ~0.1s
      RCC->>D: heartbeat
      D-->>RCC: running + errors + frame_ids
    end
    App->>RCC: stop()
    RCC->>D: stop
    App->>RCC: end()
    RCC->>D: end
```

Important details:

- `remote_camera_controller.start()` returns after the command response updates
  health. If daemon initialization failed, it raises that initialization error.
- `camera_server_daemon.start` returns `ok` only after checking
  `CameraLoader.get_all_errors()`.
- `remote_camera_controller` only marks `capturing=True` when every daemon reports
  `status="ok"` for the `start` command.
- Heartbeat stalls are detected from per-camera `frame_ids`, not from wall-clock
  guessing alone.

---

## 4. Data Path

Acquisition and outputs are **decoupled**. There are two acquisition modes, and on
top of continuous acquisition the outputs are independent **sinks** toggled at runtime.

| Mode | Acquisition | Outputs |
|------|-------------|---------|
| `image` | one-shot `single_acquire()` | one `.png` |
| `acquire` | continuous `acquire()` loop | none until a sink is enabled |

| Sink (on `acquire`) | Toggle | Output |
|---------------------|--------|--------|
| video | `set_record(path, on)` | `.avi` |
| stream | `set_stream(on)` | shared-memory double buffer |
| snapshot | `snapshot(path, count)` | next N frames as images |

```{mermaid}
flowchart TD
    START["Camera.start(image)"] --> ONE["single_acquire<br/>one .png"]
    ARM["arm() -> Camera.start(acquire)"] --> CONT["acquire loop"]
    CONT --> GRAB["PyspinCamera.get_image(timeout)"]
    GRAB --> ROUTE{"sinks enabled?"}
    ROUTE -->|set_stream| SHM["SHM double buffer"]
    ROUTE -->|set_record| AVI["VideoWriter .avi"]
    ROUTE -->|snapshot| PNG["images"]
    GRAB -. "timeout -> (None, None)" .-> CONT
```

`video` / `stream` / `full` are no longer modes — that is just `acquire` with the
corresponding sink(s) on. `set_param(gain/exposure)` also applies live in this loop.

---

## 5. Config Contract

Per-camera baseline config lives in `system/current/camera.json` and is read on
each capture PC. The main PC sends only runtime overrides in the ZMQ `start`
command.

Parameter resolution:

```text
explicit start arg > camera.json[serial] > default
```

Supported per-serial keys:

| Key | Applies to | Default |
|-----|------------|---------|
| `gain` | PySpin `Gain` | `3.0` |
| `exposure` | PySpin `ExposureTime` | `2500.0` |
| `pixel_format` | PySpin `PixelFormat` | `BayerRG8` |
| `packet_size` | PySpin `GevSCPSPacketSize` | `9000` |
| `buffer_count` | PySpin `StreamBufferCountManual` | `10` |
| `buffer_mode` | PySpin `StreamBufferHandlingMode` | `OldestFirst` |

Example:

```json
{
  "25305466": {
    "gain": 15.0,
    "exposure": 2500.0,
    "packet_size": 9000,
    "buffer_count": 10
  }
}
```

`CameraLoader` passes this per-serial config to `Camera`, which passes it to
`load_camera(..., cfg=...)`, which constructs `PyspinCamera(..., cfg=...)`.

---

## 6. Health And Failure Modes

Use this section when a run fails.

| Symptom | Likely layer | What to check |
|---------|--------------|---------------|
| main app says capture PC unreachable | daemon/ping | is `src/camera/server_daemon.py` running on that PC? |
| `start` returns `error` | daemon/camera | `msg`, `errors`, per-camera `states` in controller `get_status()` |
| frame ids stop changing | acquisition/trigger/LAN | controller `stalled`, heartbeat `frame_ids`, trigger generator |
| `stop` used to hang | `get_image()` / PySpin | finite `GRAB_TIMEOUT_MS` should let loops re-check stop/exit |
| next run cannot start cameras | capture-PC daemon/hardware | run `src/camera/reset_cameras.py` from the main PC |
| camera count mismatch | config/enumeration | `expected_camera_count` vs `detected_camera_count` in heartbeat |

Start vs heartbeat:

- `start` proves the initial arm succeeded.
- `heartbeat` proves the running capture remains healthy.
- daemon idle timeout handles a crashed controller.

---

## 7. Validation And Recovery

No-hardware validation:

```bash
python src/validate/camera_system/hang_recovery_mock.py
```

Hardware validation:

```bash
python src/validate/camera_system/hang_recovery.py
python src/validate/camera_system/sync_check.py
```

Recovery from the main PC:

```bash
python src/camera/reset_cameras.py
python src/camera/reset_cameras.py --pc_list capture1 capture2
python src/camera/reset_cameras.py --no_restart
```

After camera-daemon code changes, every capture PC needs the updated code and a
daemon restart. Updating only the main PC does not update `Camera`, `CameraLoader`,
or `PyspinCamera`.

---

## 8. Where Future Design Lives

Current runtime docs are this page plus {doc}`Camera System API <camera_system_api>`.
Future design work belongs in `design/camera-recording-redesign.md`.

Keep these separate:

- Current API: `start(mode, ...)`, `heartbeat`, `stop`, `end`.
- Future proposal: `go_live`, `start_recording`, `stop_recording`, `grab_still`.

Do not document proposal APIs as available runtime behavior until the callers and
daemon protocol have been migrated.
