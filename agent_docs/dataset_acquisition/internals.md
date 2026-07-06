# Dataset Acquisition — Internals (for editors)

You're changing `paradex/dataset_acqusition/` itself. Two files, ~230 lines total, no state
machine of their own beyond the teleop loop — the complexity is all in *what they orchestrate*.

> Just calling these? Read [`usage.md`](usage.md) instead.

Module dir name is `dataset_acqusition` (missing 'i') — intentional; renaming it breaks every
`from paradex.dataset_acqusition...` importer (grep before touching).

---

## capture.py — `CaptureSession`

A device multiplexer. Each optional subsystem (`camera`, `arm`, `hand`, `teleop_device`) is
either a live handle or `None`, and every lifecycle method is `if self.x is not None: self.x.<op>()`.
Adding a new device = add it to `__init__`, then to `start/stop/end` following that pattern.

### Construction wiring (`__init__`)
- Guard: `teleop is not None` **requires** `arm or hand` → `ValueError`.
- `camera=True`:
  - `self.camera = remote_camera_controller(name="dataset_acquisition")` (the rcc; connects +
    locks the daemons — see camera `internals.md`).
  - `self.sync_generator = UTGE900(**network_info["signal_generator"]["param"])`.
  - `self.timestamp_monitor = TimestampMonitor(...)` **only if** `arm or hand` (no monitor for
    camera-only sessions — nothing to cross-sync).
- `arm` → `get_arm(arm)`; `hand` → `get_hand(hand, ip=hand_ip)`.
- `teleop=="xsens"` → `XSensReceiver` + `Retargetor(arm_name=arm, hand_name=hand)` +
  `HandStateExtractor`. `"occulus"` branch is commented out (imports would pull oculus deps).
- `self.save_path` / `self.stage` start `None`; they gate the record-phase behavior.

### `start()` — the raw/ layout is defined here
`raw_rel = "raw/<stage>" if stage else "raw"`. Every recorder is handed
`os.path.join(shared_dir, save_path, raw_rel, <name>)`:
| device | subdir | notes |
|--------|--------|-------|
| arm | `arm/` | `self.arm.start(dir)` |
| hand | `hand/` | `self.hand.start(dir)` |
| teleop | `teleop/` | also resets `self.state_hist=[]`, `self.state_time=[]` |
| camera | (whole `raw_rel`) | `self.camera.start(mode, True, raw_rel_path, fps, exposure_time, gain)` — **`syncMode` is hard-coded `True`** |
| timestamp_monitor | `timestamps/` | only if it exists |

**Ordering invariant (do not reorder).** The UTGE900 trigger must be the **last thing on and
the last thing off**, because both the camera *and* the `timestamp_monitor` are **trigger
consumers** that block/listen for its pulses:
- `start()`: arm/hand/teleop → `camera.start()` → `timestamp_monitor.start()` → **`sync_generator.start(fps)` last**.
  A sync-mode camera armed *after* the trigger already fired misses the first pulses; both
  consumers must be listening before a single pulse goes out.
- `stop()` (mirror): `camera.stop()` → `timestamp_monitor.stop()` → **`sync_generator.stop()` last**.
  Kill the trigger *first* and a still-acquiring camera / monitor blocks forever waiting for a
  pulse that never comes — so the trigger has to outlive both.

So the trigger **brackets** the two consumers on both ends. If you add another trigger-driven
sensor, arm it inside this bracket (after `camera`, before `sync_generator`) at both start and stop.

### `stop()`
Mirror-image teardown. Notable side effects (not just "stop"):
- teleop → `np.save(state/state_hist.npy)`, `state/state_time.npy` (recomputes `raw_rel` from
  `self.stage`, so it must run before those fields are cleared).
- camera → `camera.stop()`, `timestamp_monitor.stop()`, `sync_generator.stop()` (in that order —
  see the ordering invariant above), then `save_current_camparam(shared_dir/save_path)` +
  `save_current_C2R(...)` — this is why each dataset carries its own calibration. If you move the
  calibration snapshot, `process` code that does `load_camparam(demo_path)` breaks.
- Clears `self.save_path = self.stage = None` at the end (re-arms for another `start`).

### `end()`
`arm/hand/teleop/camera` each `.end()`; camera also ends `sync_generator` + `timestamp_monitor`.
`camera.end()` is what releases the daemon lock — don't drop it.

### `teleop()` — the record/prepare loop
Blocking loop @ `time.sleep(0.01)`:
- `home_pose = arm.get_data()["position"]` (or `np.eye(4)` if no arm) → `retargetor.start(home_pose)`.
- Each tick: `data = teleop_device.get_data()`; skip while `data["Right"] is None`.
- `state = state_extractor.get_state(data['Left'])` — the **left** hand drives the state machine:
  | state | meaning | action |
  |-------|---------|--------|
  | 0 | active | retarget `data` → `hand.move(hand_action)` + `arm.move(wrist_pose)` |
  | 1 | pause | `retargetor.stop()` |
  | 2 | stop-hold | `retargetor.stop()`, `stop_counter++` |
  | 3 | exit-hold | `exit_counter++` |
  - counters reset when their state isn't seen; `> 90` consecutive (~0.9 s) → returns `"stop"` /
    `"exit"` (with a `chime` cue).
- State logging happens **only while `self.save_path` is set** (i.e. record phase, not prepare).
- `chime.theme('pokemon')` at import; audio cues are load-bearing UX, not debug.

Gotcha: the `if state == 2` / `else` and `if state == 3` / `else` blocks mean `stop_counter`
resets on *any* non-2 state and `exit_counter` on *any* non-3 — but because state 2's `else`
also catches state 3 (and vice-versa), only one counter climbs at a time. Read carefully before
"simplifying" the branch structure.

### `move(action_dict)`
Scripted alternative to `teleop()`: `{"arm": ...}` → `arm.move`, `{"hand": ...}` → `hand.move`.
No `None`-guard here — passing `"arm"` on a `hand`-only session `AttributeError`s. Callers own that.

---

## match_sync.py — post-hoc frame alignment

Module-level `td = 2 / 30` — a fixed ~66 ms trigger-to-exposure latency offset baked for 30 fps.
**Not parameterized**; if you capture at another fps this constant is wrong.

### `fill_framedrop(frame_id, pc_time)`
1. Drops the first `real_start = 10` frames (startup jitter).
2. `time_delta = (pc_time[-1]-pc_time[0]) / (frame_id[-1]-frame_id[0])` — seconds per frame
   *index*, robust to dropped frames because it divides by the frame-id span, not the count.
3. `offset = mean(pc_time - (frame_id-1)*time_delta)` — the linear fit's intercept.
4. Emits a **dense** `frame_id_nodrop = arange(1, last+500)` and
   `pc_time_nodrop = (frame_id_nodrop-1)*time_delta + offset - td`.

The `+500` is slack so the reconstructed timeline outruns the real capture; downstream code
indexes into it and stops at the real data length. Editing the model: keep it a pure linear
fit — `overlay`/sync consumers assume `pc_time_nodrop[k]` is the time of frame `k+1`.

### `get_synced_data(pc_times, data, data_times)`
Monotone two-pointer nearest-neighbor: advances `j` while `data_times[j+1]` is at least as close
to `pc_times[i]` as `data_times[j]`, then appends `data[j]`. **Assumes both arrays are sorted
ascending.** O(n+m). Returns `np.array(synced_data)` of length `len(pc_times)`. If you need
interpolation instead of nearest, this is the function to change (but check every caller in
`src/process/miyungpa/` — they expect exact sensor rows, e.g. quaternions you can't lerp naively).

---

## Consumers (change these two files → verify against them)
- `CaptureSession`: `src/dataset_acquisition/{miyungpa,hri,object_turntable,graphics/motion_blur}/`,
  `src/capture/robot/teleop_real.py`, `src/object6d/validate/capture_test.py`.
- `match_sync`: `src/process/miyungpa/{process,process_client,visualizer}.py`.

There is **no test suite** — validate by running a real capture (`src/validate/` for device
smoke tests) or by re-running a miyungpa process on an existing session.
