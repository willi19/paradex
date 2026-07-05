# Capture Session

Overview of Paradex's capture-session subsystem — how one call orchestrates the
cameras, robot arm, robot hand, teleop input, hardware sync, and timestamp logging
into a single labelled dataset. Read this for the mental model; for method
signatures, parameters, and return values see the {doc}`API reference <capture_api>`.

- Generated per-symbol API: {doc}`API Reference <autoapi/index>`
- Camera internals used underneath: {doc}`Camera System <camera_system>`

---

## 1. What a Capture Session Is

`CaptureSession` is the one abstraction a dataset pipeline talks to. It wraps the
individual device controllers so that a single `start()` / `stop()` pair records
**every enabled sensor** into a consistent on-disk layout, and a single `end()`
releases them all.

A session is assembled from up to five subsystems, each optional and selected at
construction time:

```{mermaid}
flowchart TB
    CS["CaptureSession"]
    subgraph Devices["Devices (opt-in per session)"]
      CAM["camera<br/>remote_camera_controller"]
      ARM["arm<br/>get_arm(...)"]
      HAND["hand<br/>get_hand(...)"]
      TEL["teleop<br/>XSensReceiver"]
    end
    subgraph Sync["Sync + logging (auto with camera)"]
      GEN["sync_generator<br/>UTGE900"]
      TS["timestamp_monitor<br/>TimestampMonitor"]
    end
    CS --> CAM & ARM & HAND & TEL
    CAM --> GEN
    CAM -.->|"if arm or hand"| TS
```

The constructor is `CaptureSession(camera=False, arm=None, hand=None, teleop=None,
hand_ip=False)`. What gets built is derived from the flags:

| Flag set | Built |
|----------|-------|
| `camera=True` | `remote_camera_controller` + `UTGE900` sync generator |
| `camera=True` **and** (`arm` or `hand`) | additionally a `TimestampMonitor` |
| `arm="xarm"` | `get_arm(arm)` → `XArmController` |
| `hand="allegro"` / `"inspire"` | `get_hand(hand, ip=hand_ip)` |
| `teleop="xsens"` | `XSensReceiver` + `Retargetor` + `HandStateExtractor` |

A teleop device requires at least one of `arm` or `hand`; otherwise the constructor
raises `ValueError`. Only `xsens` is wired (the `occulus` branch is commented out).

---

## 2. Core Concepts

| Term | Meaning |
|------|---------|
| **Session** | One `CaptureSession` instance; a device set held open across many recordings. |
| **Recording** | One `start()` … `stop()` span, producing one `<save_path>` on disk. |
| **`save_path`** | Directory **relative** to `shared_dir` (`~/shared_data`). The session prepends `shared_dir`. |
| **`stage`** | Optional sub-label; when set, per-session data lands under `raw/<stage>/` instead of `raw/`. Lets one `save_path` hold a sweep of trials. |
| **Sync capture** | The camera is always started with `syncMode=True`; the `UTGE900` fires the shared hardware trigger at `fps`. |
| **Timestamp monitor** | A camera that only logs `frame_id`/`pc_time`, used to align robot data to camera frames when an arm/hand is present. |

---

## 3. Lifecycle

A session is opened once, driven through many recordings, then closed. Devices are
constructed in `__init__` (connections stay live for the session's whole lifetime);
`start`/`stop` toggle recording; `end` tears the connections down.

```{mermaid}
sequenceDiagram
    participant App as Pipeline script
    participant CS as CaptureSession
    participant Dev as Devices (cam/arm/hand/teleop)
    App->>CS: CaptureSession(camera, arm, hand, teleop)
    CS->>Dev: connect (stay live)
    loop per recording
      App->>CS: start(save_path, mode, fps, exposure, gain, stage)
      CS->>Dev: start(...) on every enabled device
      App->>CS: (drive motion / teleop / replay)
      App->>CS: stop()
      CS->>Dev: stop(); save camparam + C2R
    end
    App->>CS: end()
    CS->>Dev: end (release all)
```

`start()` fans out to whichever devices exist, in order: arm → hand → teleop →
camera → timestamp monitor → sync generator. `stop()` reverses the responsibility:
it stops the devices, dumps the teleop state arrays, and snapshots the calibration.
`end()` releases every device connection.

---

## 4. Dataset Layout Produced

`start(save_path, ..., stage=None)` writes under `shared_dir/<save_path>`. Each
device controller owns one subdirectory of `raw/` (or `raw/<stage>/` when `stage`
is given). On `stop()` the current calibration is snapshotted at the `save_path`
**root** (not inside `raw/`).

```{mermaid}
flowchart TB
    ROOT["shared_dir/&lt;save_path&gt;/"]
    ROOT --> CAMPARAM["cam_param<br/>(save_current_camparam)"]
    ROOT --> C2R["C2R<br/>(save_current_C2R)"]
    ROOT --> RAW["raw/  (or raw/&lt;stage&gt;/)"]
    RAW --> VID["videos/ or images/<br/>(camera)"]
    RAW --> ARMD["arm/"]
    RAW --> HANDD["hand/"]
    RAW --> TELD["teleop/"]
    RAW --> TSD["timestamps/<br/>frame_id.npy, timestamp.npy"]
    RAW --> STD["state/<br/>state_hist.npy, state_time.npy<br/>(teleop only)"]
```

| Path | Written by | Present when |
|------|-----------|--------------|
| `raw[/stage]/videos` or `/images` | camera controller | `camera=True` |
| `raw[/stage]/arm` | `arm.start(...)` | `arm` set |
| `raw[/stage]/hand` | `hand.start(...)` | `hand` set |
| `raw[/stage]/teleop` | `teleop_device.start(...)` | `teleop` set |
| `raw[/stage]/timestamps` | `timestamp_monitor.start(...)` | `camera` + (`arm` or `hand`) |
| `raw[/stage]/state` | `stop()` dumps `state_hist`/`state_time` | `teleop` set |
| `cam_param`, `C2R` (root) | `save_current_camparam` / `save_current_C2R` on `stop()` | `camera=True` |

`mode` (default `"video"`, else `"image"`) chooses whether the camera writes
`videos/` or `images/`.

---

## 5. Hardware Sync & Timestamp Alignment

With a camera, every recording is hardware-synced: the `UTGE900` signal generator
emits a square wave at `fps` on the shared trigger line, so all cameras expose
together. Robot controllers, however, sample on their own clocks. When an arm or
hand is present the session adds a `TimestampMonitor` — a dedicated camera that logs
only `(frame_id, pc_time)` — so robot streams can be resampled onto camera frames in
post-processing.

```{mermaid}
flowchart TB
    GEN["UTGE900<br/>square wave @ fps"] -->|"hardware trigger"| CAM["cameras<br/>(synced frames)"]
    GEN --> TSMON["TimestampMonitor<br/>logs (frame_id, pc_time)"]
    ARM["arm stream<br/>(own clock)"] --> MATCH["match_sync<br/>fill_framedrop / get_synced_data"]
    TSMON --> MATCH
    MATCH --> ALIGNED["robot data<br/>resampled to frames"]
```

The alignment helpers live in `paradex/dataset_acqusition/match_sync.py`:

- `fill_framedrop(frame_id, pc_time)` — reconstructs a drop-free frame-id / pc-time
  axis from the logged (gappy) samples by fitting a constant per-frame time delta
  (skips the first 10 frames for startup jitter).
- `get_synced_data(pc_times, data, data_times)` — nearest-timestamp two-pointer
  match of an arbitrary `data` stream onto the frame time axis.

---

## 6. Teleop Loop

When built with `teleop="xsens"`, `teleop()` runs the retargeting control loop and
is the session's driver for XSens-based pipelines. It reads the XSens stream, maps
the left-hand pose to a discrete **state** (`HandStateExtractor.get_state`), and uses
the state to command the robot and to signal session boundaries — it is called twice
per recording: once to "prepare" (before `start`), once to "record" (while `start`
is active).

```{mermaid}
flowchart TB
    S["teleop_device.get_data()"] --> ST{"state = get_state(Left)"}
    ST -->|"0 move"| ACT["retargetor.get_action<br/>→ arm.move / hand.move"]
    ST -->|"1/2 stop"| RS["retargetor.stop()"]
    ST -->|"2 held &gt;90"| RET_STOP["return &quot;stop&quot;"]
    ST -->|"3 held &gt;90"| RET_EXIT["return &quot;exit&quot;"]
```

`get_state` values: `0` = move (retarget + drive robot), `1` = pause retargeting,
`2` = stop (held ≈90 ticks at 10 ms → returns `"stop"`), `3` = exit (held ≈90 ticks
→ returns `"exit"`). While a `save_path` is active the per-tick state and wall time
are appended to `state_hist`/`state_time`, which `stop()` writes to `state/`.
`move(action_dict)` is a manual alternative that forwards `"arm"`/`"hand"` commands
without teleop.

---

## 7. Pipelines

Every dataset pipeline in `src/dataset_acquisition/` (and `src/capture/robot/`) is a
thin script over one `CaptureSession`, differing only in device set, trigger, and
`save_path`.

| Pipeline | Devices | Driven by | Notes |
|----------|---------|-----------|-------|
| `graphics/motion_blur` | camera + arm | trajectory replay | Sweeps (exposure, gain) × speed; one synced video per `stage`. |
| `hri` | camera only | keyboard (`c`/`s`/`q`) | Camera-only; no timestamp monitor (sync still active). |
| `object_turntable` | camera only | keyboard (`c`/`s`/`q`) | Same control structure as `hri`; feeds COLMAP post-processing. |
| `miyungpa` | camera + arm + hand + teleop | `teleop()` gestures | Full teleop demo; adds timestamp monitor + `state/` arrays. |
| `capture/robot/teleop_real` | arm + hand + teleop (no camera) | `teleop()` gestures | `camera=False` → robot streams only, no sync generator. |

Two `graphics` siblings (`image_capture.py`, `image_traj.py`) deliberately **bypass**
`CaptureSession` and call `remote_camera_controller` directly with `syncMode=False`
(free-run exposure grids); they are not capture sessions.

```{mermaid}
flowchart TB
    subgraph CamOnly["camera only"]
      HRI["hri"] & OT["object_turntable"]
    end
    subgraph CamArm["camera + arm"]
      MB["graphics/motion_blur"]
    end
    subgraph Full["camera + arm + hand + teleop"]
      MP["miyungpa"]
    end
    subgraph NoCam["arm + hand + teleop (no camera)"]
      TR["capture/robot/teleop_real"]
    end
    CS["CaptureSession"] --> CamOnly & CamArm & Full & NoCam
```

---

## 8. Control / Data Flow

Putting it together for the fullest case (camera + arm + hand + teleop): commands
fan out on `start`, data flows to disk during the recording, calibration is
snapshotted on `stop`.

```{mermaid}
flowchart TB
    APP["pipeline script"] -->|"start / stop / end"| CS["CaptureSession"]
    CS -->|"start(save_path, sync=True, fps, exp, gain)"| CAM["remote_camera_controller"]
    CS -->|"start(fps)"| GEN["UTGE900"]
    CS -->|"start(.../timestamps)"| TS["TimestampMonitor"]
    CS -->|"start(.../arm)"| ARM["arm controller"]
    CS -->|"start(.../hand)"| HAND["hand controller"]
    CS -->|"start(.../teleop)"| TEL["XSensReceiver"]
    CAM --> DISK["shared_dir/&lt;save_path&gt;/raw"]
    ARM & HAND & TEL & TS --> DISK
    CS -.->|"stop(): snapshot"| CAL["cam_param + C2R"]
```

---

## 9. File Reference

| Target | File |
|--------|------|
| Session orchestrator | `paradex/dataset_acqusition/capture.py` (`CaptureSession`) |
| Post-hoc frame alignment | `paradex/dataset_acqusition/match_sync.py` |
| Hardware trigger | `paradex/io/camera_system/signal_generator.py` (`UTGE900`) |
| Timestamp logging camera | `paradex/io/camera_system/timestamp_monitor.py` (`TimestampMonitor`) |
| Main-PC camera control | `paradex/io/camera_system/remote_camera_controller.py` |
| Arm / hand factories | `paradex/io/robot_controller/__init__.py` (`get_arm`, `get_hand`) |
| Teleop input | `paradex/io/teleop/xsens/receiver.py` (`XSensReceiver`) |
| Retarget + hand state | `paradex/retargetor/unimanual.py`, `paradex/retargetor/state.py` |
| Pipeline scripts | `src/dataset_acquisition/*/capture.py`, `src/capture/robot/teleop_real.py` |

Note: the module directory is spelled `dataset_acqusition` (missing the second
`i`) — this is intentional, do not "fix" it.

Method-by-method API (parameters / returns): {doc}`Capture Session — API <capture_api>`.
