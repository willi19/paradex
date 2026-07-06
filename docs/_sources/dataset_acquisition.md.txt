# Dataset Acquisition

Overview of Paradex's dataset-acquisition subsystem — how one *start/stop* records the
multi-camera rig, arm, hand, and teleop together into a fixed `raw/` layout, and how the
per-sensor logs are re-aligned to the camera clock afterwards. Read this for the mental model;
for method signatures see the {doc}`API reference <dataset_acquisition_api>`.

- Core library: `paradex/dataset_acqusition/` (`capture.py`, `match_sync.py`)
- Capture scripts built on it: `src/dataset_acquisition/` (+ `src/capture/robot/teleop_real.py`)
- The sync consumer: `src/process/miyungpa/`
- Generated per-symbol API: {doc}`API Reference <autoapi/index>`

:::{note}
The module directory is spelled **`dataset_acqusition`** (missing an 'i'). This is intentional
and load-bearing — every `from paradex.dataset_acqusition …` import depends on it. Do not "fix" it.
:::

---

## 1. What the subsystem does

A capture session streams several sensors at once — cameras (over the network, via the remote
camera controller), a robot arm, a robot hand, and optionally an XSens teleop suit. This layer
is the thin **orchestrator** that fans a single lifecycle across all of them and lands the data
in one predictable directory tree. It is *not* a device driver: the actual IO lives in
`paradex/io/…` (camera, robot, teleop) and calibration save/load in `paradex/calibration/`.

Two tools, used at opposite ends of a capture:

| Tool | When | Role |
|------|------|------|
| `CaptureSession` (`capture.py`) | **record time** | One object; `start()/stop()/end()` drive every enabled sensor together and write `raw/`. |
| `match_sync` (`match_sync.py`) | **process time** | `fill_framedrop` + `get_synced_data` re-align the arm/hand `.npy` logs to the camera frame clock. |

```{mermaid}
flowchart LR
    subgraph REC["record — CaptureSession"]
      CAM["remote camera controller"]
      ARM["arm recorder"]
      HAND["hand recorder"]
      TEL["teleop (XSens)"]
      SG["UTGE900 sync generator"]
      TS["TimestampMonitor"]
    end
    REC --> RAW["shared_data/&lt;path&gt;/raw/<br/>{videos, arm, hand, teleop, timestamps}"]
    RAW --> MS["match_sync<br/>(fill_framedrop → get_synced_data)"]
    MS --> SYNC["&lt;path&gt;/{arm,hand}/*.npy<br/>(one row per camera frame)"]
```

---

## 2. Core concepts

| Term | Meaning |
|------|---------|
| **Session** | One `CaptureSession` recording under `shared_data/<save_path>/`. `save_path` is **relative** to `shared_dir`; convention `capture/<pipeline>/<name>/<ts>`. |
| **Stage** | Optional sub-bucket: with `stage` set, data lands under `raw/<stage>/…` instead of `raw/…` (the upload processor expects this when present). |
| **Sync generator** | The UTGE900 hardware trigger. In a session the camera records in **sync mode**, so the trigger must be running for any frame to arrive. |
| **Trigger consumer** | Anything that blocks/listens on the trigger — the **camera** and the **TimestampMonitor**. Both must be armed before the trigger starts and torn down before it stops (§4). |
| **TimestampMonitor** | Records per-frame `frame_id.npy` + `timestamp.npy`. Built **only** when `camera` and (`arm` or `hand`) are all present — there's nothing to cross-sync otherwise. |
| **Calibration snapshot** | On `stop()`, a camera session copies the *current* camparam + C2R next to the data, so each dataset carries the calibration it was shot with. |

---

## 3. CaptureSession lifecycle

`CaptureSession(camera, arm, hand, teleop, hand_ip)` builds only the devices you ask for; every
unused one stays `None` and is skipped in `start/stop/end`. So camera-only, arm-only, and
camera+arm+hand+teleop all flow through the same three calls.

```{mermaid}
flowchart TB
    C["CaptureSession(camera, arm, hand, teleop)"] --> ST["start(save_path, mode, fps, …)"]
    ST --> DRV{"drive"}
    DRV -->|teleoperation| TL["teleop()  → 'stop' / 'exit'"]
    DRV -->|scripted| MV["move({'arm':…, 'hand':…})"]
    TL --> SP["stop()"]
    MV --> SP
    SP -->|record again| ST
    SP --> EN["end()  (releases camera lock)"]
```

- **`start(save_path, mode="video", fps=30, exposure_time=None, gain=None, stage=None)`** — makes
  `raw[/<stage>]/`, starts each enabled recorder into its subdir, then starts the camera in
  **`syncMode=True`** and fires the sync generator last.
- **`stop()`** — stops everything; teleop dumps `state/`, and a camera session snapshots
  calibration (`save_current_camparam` + `save_current_C2R`). Re-arms for another `start()`.
- **`end()`** — releases every device; `camera.end()` frees the daemon lock. Always call it.
- **`teleop()`** — blocking XSens retarget loop (§5). **`move(action_dict)`** — scripted single step.

### The `raw/` layout `start()` writes
```
shared_data/<save_path>/
  raw[/<stage>]/
    videos/ | images/     # camera, per-serial
    arm/                  # arm recorder   (time.npy + data .npy)
    hand/                 # hand recorder
    teleop/               # raw XSens stream
    timestamps/           # frame_id.npy, timestamp.npy  (TimestampMonitor)
  state/                  # teleop only: state_hist.npy, state_time.npy
  cam_param/ , C2R …      # calibration snapshot (camera sessions, on stop)
```

---

## 4. The sync-generator ordering invariant

This is the one rule you cannot reorder. The UTGE900 trigger must be the **last thing turned on
and the last thing turned off**, because both the camera *and* the TimestampMonitor are trigger
consumers that block on its pulses.

```{mermaid}
sequenceDiagram
    participant S as sensors (arm/hand/teleop)
    participant C as camera
    participant T as TimestampMonitor
    participant G as sync generator (trigger)
    Note over S,G: start()
    S->>S: start
    C->>C: start (syncMode=True, waits for trigger)
    T->>T: start (listens for trigger)
    G->>G: start  ← LAST
    Note over S,G: stop()  (mirror)
    C->>C: stop  ← FIRST
    T->>T: stop
    G->>G: stop  ← LAST
```

- **Start** — arm/hand/teleop → `camera.start()` → `timestamp_monitor.start()` →
  `sync_generator.start(fps)` **last**. A sync-mode camera armed *after* the trigger already
  fired misses the first pulses.
- **Stop** — `camera.stop()` → `timestamp_monitor.stop()` → `sync_generator.stop()` **last**.
  Kill the trigger first and a still-acquiring camera/monitor blocks forever waiting for a pulse
  that never comes.

So the trigger **brackets** the two consumers on both ends. A new trigger-driven sensor must be
armed inside that bracket at both start and stop.

---

## 5. Teleop state machine

`teleop()` runs a blocking loop (`time.sleep(0.01)`); the operator's **left** hand drives a state
machine (`HandStateExtractor.get_state(data['Left'])`). It is the control flow for teleop capture
pipelines (e.g. miyungpa) — *not* keyboard.

| State | Meaning | Action |
|-------|---------|--------|
| 0 | active | retarget → `hand.move(hand_action)` + `arm.move(wrist_pose)` |
| 1 | pause | `retargetor.stop()` |
| 2 | stop-hold | `retargetor.stop()`, `stop_counter++` |
| 3 | exit-hold | `exit_counter++` |

A counter held `> 90` consecutive ticks (~0.9 s) returns `"stop"` / `"exit"` (with a `chime`
audio cue). State is logged to `state_hist`/`state_time` **only while a `save_path` is set** —
i.e. during the record phase, not the prepare phase. Typical record loop:

```python
while True:
    if cs.teleop() == "exit": break     # prepare phase (no save_path)
    cs.start("capture/miyungpa/<name>/<ts>")
    state = cs.teleop()                 # record phase
    cs.stop()
    if state == "exit": break
cs.end()
```

---

## 6. match_sync — post-hoc frame alignment

Camera frames and the arm/hand recorders run on different clocks, and the camera drops frames.
After a session is saved, `match_sync` reconstructs a per-frame sensor value. Used by
`src/process/miyungpa/`.

```{mermaid}
flowchart LR
    FID["raw/timestamps/frame_id.npy"] --> FF["fill_framedrop"]
    PCT["raw/timestamps/timestamp.npy"] --> FF
    FF --> DENSE["dense, drop-free<br/>frame timeline"]
    DENSE --> GS["get_synced_data<br/>(2-pointer nearest)"]
    ST["raw/arm|hand/time.npy + data"] --> GS
    GS --> OUT["&lt;demo&gt;/arm|hand/*.npy<br/>(one row per frame)"]
```

- **`fill_framedrop(frame_id, pc_time)`** — fits a linear `frame_id → time` model (skipping the
  first 10 warmup frames), emits a dense `1 … last+500` frame timeline with reconstructed times
  so downstream indexing is gap-free. Bakes a fixed `td = 2/30 s` trigger-latency offset (30 fps
  assumption).
- **`get_synced_data(pc_times, data, data_times)`** — for each frame time, the nearest sensor row
  (monotone two-pointer; assumes both arrays are time-sorted). Output length = `len(pc_times)`.

---

## 7. Component reference

| Component | Location | Responsibility |
|-----------|----------|----------------|
| `CaptureSession` | `paradex/dataset_acqusition/capture.py` | Multiplex camera+arm+hand+teleop through one start/stop/end; write `raw/`; snapshot calibration. |
| `CaptureSession.teleop` | same | XSens retarget loop + left-hand state machine. |
| `fill_framedrop` | `paradex/dataset_acqusition/match_sync.py` | Linear frame→time fit; dense drop-free timeline. |
| `get_synced_data` | same | Two-pointer nearest-time sensor matcher. |
| Camera driver | `paradex/io/camera_system/` | `remote_camera_controller`, `UTGE900`, `TimestampMonitor`. |
| Robot drivers | `paradex/io/robot_controller/` | `get_arm` / `get_hand`. |
| Teleop | `paradex/io/teleop/xsens/` + `paradex/retargetor/` | `XSensReceiver`, `Retargetor`, `HandStateExtractor`. |
| Calibration snapshot | `paradex/calibration/utils.py` | `save_current_camparam`, `save_current_C2R`. |

---

## 8. Downstream consumers

| Consumer | Uses |
|----------|------|
| `src/dataset_acquisition/{miyungpa,hri,object_turntable,graphics/motion_blur}/` | `CaptureSession` to record datasets. |
| `src/capture/robot/teleop_real.py`, `src/object6d/validate/capture_test.py` | `CaptureSession`. |
| `src/process/miyungpa/{process,process_client,visualizer}.py` | `match_sync` to align arm/hand to video. |

Method-by-method API (parameters / returns): {doc}`Dataset Acquisition — API <dataset_acquisition_api>`.
