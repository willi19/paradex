# paradex/video — Internals (for agents editing this module)

**You are here to change the pipeline** (path resolution, the resume/skip guard, the drop-frame
heuristic, the nvenc subprocess, the delete-on-success move) or to understand why a video ended
`failed`. If you only want to *call* something, read [`usage.md`](usage.md).

Two files. `raw_video_processor.py` is the real work; `util.py` is four independent CPU-ffmpeg
one-shots with no shared state. The through-line: **this module is destructive — on success it
`os.remove`s both its output and the raw source — and the encode path is GPU-only with no fallback.**

---

## 1. The per-video worker — `undistort_raw_video(video_path, progress_dict, video_id)`

This is what each pool worker runs. Flow:

1. **Resolve paths** (`serial = filename stem`). Two layouts, keyed off the `raw/` path component:
   - new: `.../{session}/raw/{stage}/videos/{serial}.avi`
   - old: `.../{session}/raw/videos/{serial}.avi` (`stage = ""`)
   - `root_name` = the session dir with every `capture_path_list` prefix stripped (leading `/` removed).
   - Local output: `.../{session}/videos/{stage}/{serial}.avi`; NAS: `{shared_dir}/{root_name}/videos/{stage}/{serial}.avi`.
2. **Skip / resume guard:** if the output already exists with the **same frame count**, skip re-encode
   → just `rsync_copy` + delete raw. If frame counts **mismatch**, treat as an incomplete file →
   delete it and reprocess. This frame-count check is the **only safety net** before the raw source is
   deleted.
3. **Load calibration:** `load_camparam(shared_dir/root_name)` → `precompute_undistort_map_torch(intrinsics[serial])`
   builds a GPU `grid_tensor`. Failure here → status `failed`, early return (raw kept).
4. **Frame loop** (see §2 for drop-frame detection), piping raw `bgr24` bytes into an ffmpeg
   `h264_nvenc` subprocess (`-rc vbr -cq 19 -b:v 0 -preset p4 -tune hq`, GOP 30). ffmpeg stderr →
   `{out_path}.ffmpeg.log`.
5. **Finish:** close stdin, `wait()`. On broken pipe or nonzero rc → status `failed`, **delete the
   partial output**, keep the log, return. On success → `rsync_copy` to NAS, then **`os.remove` both
   the local output and the raw source**.

Progress is written through `update_progress(progress_dict, video_id, updates)`, which does a
read-copy-write on the `Manager.dict` (`temp = dict(...); temp.update(...); dict[id] = temp`) —
**necessary** because mutating a nested dict in a `Manager.dict` in place does not propagate.

---

## 2. Drop-frame detection — a pixel heuristic

```python
patch = frame[:30, :30]
is_dropped = (patch[::2, ::2] > 240).all() and (patch[1::2, 1::2] < 15).all()
```

The camera writer stamps a checkerboard marker into the **top-left 30×30 patch** on dropped frames
(even pixels bright > 240, odd pixels dark < 15). A hit → the frame is replaced with a **black frame**
(`np.zeros_like`) so frame **indices stay aligned**; otherwise it's undistorted. If the writer's marker
convention changes, this misfires silently — the producer side is in
[`agent_docs/camera_system/`](../camera_system/). This is a heuristic, not a checksum.

---

## 3. The orchestrator — `RawVideoProcessor`

- `__init__` calls `get_raw_videopath_list()` (globs both layouts across `capture_path_list`) and sets
  `num_workers = min(4, cpu_count())`, plus a `Manager` `dict` (`progress_dict`) and `list` (`log`).
- `process()` opens a `Pool`, `apply_async`s each video with `callback`/`error_callback` appending to
  `self.log`, then `pool.close()` and one `print_progress()`. **`process()` returns immediately** —
  dispatch only; use `wait_and_monitor()` or poll `finished()`.
- `finished()` = `all(r.ready() for r in self.process_list)`. `stop()` = `pool.terminate()` + `join()`.

**Oversubscription risk:** each of the ≤ 4 workers also spawns an ffmpeg **nvenc** process and touches
the **GPU**. Raising `num_workers` past 4 can oversubscribe the single encoder — the cap is deliberate.

---

## 4. `util.py` — independent CPU converters

Four functions, each `subprocess.run(ffmpeg..., check=True)` wrapped in
`try/except CalledProcessError`. On failure they **print and return** (no raise, no cleanup of a
partial output) — so a caller that assumes success can proceed on a missing/partial file. All use CPU
`libx264` (portable, unlike the pipeline's nvenc). `change_to_h264` uses `crf 17` (visually lossless);
the rest use `crf 23`. `images_to_video` writes an OpenCV `mp4v` temp first, re-encodes to libx264,
then `os.remove`s the temp. Comments are Korean — leave them.

---

## 5. Gotchas for editors

- **`h264_nvenc` requires an NVIDIA GPU + nvenc-enabled ffmpeg — no CPU fallback.** On a box without
  nvenc, every pipeline video ends `failed` with the reason in `{out_path}.ffmpeg.log`. If you need
  portability, `util.py`'s `libx264` path is the model; don't assume the pipeline runs on CI/laptops.
- **The worker deletes the raw source on success** (`os.remove(video_path)`). It's a **move, not a
  copy** — a bug in the NAS path derivation or a silent rsync failure loses data. `rsync_copy` returns
  `False` (never raises) on failure, but **the worker does not check its return value** before deleting
  the source. Tread carefully if you touch step 5.
- **Layout parsing keys entirely off a `raw/` path component** and the presence/absence of a `{stage}`
  dir between `raw/` and `videos/`. Non-conforming paths resolve to wrong output locations **silently**
  — no validation.
- **`load_camparam` reads `shared_dir/root_name`** (the **NAS** copy of the calibration). The session's
  calibration must already be on the NAS when this runs, or the worker fails at step 3.
- **`load_params` status is only set inside the `if os.path.exists(out_path)` block.** When the output
  doesn't yet exist, the run jumps from `starting` straight into the calibration load without emitting
  `loading_params` — harmless for correctness, but don't rely on that status transition always firing.
- The frame-timing averages (`avg_read_ms`/`avg_undistort_ms`/`avg_write_ms`) divide by a **fixed 30**,
  so the final partial 30-frame window reports skewed numbers. Advisory only.
