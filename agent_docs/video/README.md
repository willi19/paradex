# agent_docs/video — agent orientation

Docs for **AI agents working on `paradex/video/`** — the **post-capture** video layer.
Cameras write raw distorted `.avi` to the capture PCs' local disks; this module turns those
into calibrated, compressed, uploaded videos. It runs **after** a capture session, not during
one (that's `camera_system/`). Calling the pipeline or a converter? See [`usage.md`](usage.md).
Editing the pipeline / drop-frame heuristic / delete-on-success path? See [`internals.md`](internals.md).

Mental model: **raw local `.avi` → undistort + drop-frame fix → GPU H.264 re-encode → rsync to
NAS → delete local.** `raw_video_processor.py` is the pipeline; `util.py` is a bag of standalone
ffmpeg one-shot converters used elsewhere.

## File map
| File | What it is |
|------|-----------|
| `raw_video_processor.py` | The batch pipeline: find raw videos → undistort + re-encode → upload. `RawVideoProcessor` (pool orchestrator) + `undistort_raw_video` (per-video worker) + `get_raw_videopath_list`. |
| `util.py` | Standalone ffmpeg helpers — `change_to_h264`, `images_to_video`, `convert_avi_to_mp4`, `convert_avi_to_compressed_avi`. Not part of the pipeline; called by `src/process/miyungpa/` etc. |

All paths relative to [`paradex/video/`](../../paradex/video/).

---

## Who calls this
| Caller | Uses |
|--------|------|
| [`src/util/upload_video/client.py`](../../src/util/upload_video/client.py) | `RawVideoProcessor()` — the real distributed upload app (dashboard on main PC). |
| [`src/validate/upload_raw_video/upload_local.py`](../../src/validate/upload_raw_video/upload_local.py) | `RawVideoProcessor()` — local smoke test. |
| `src/process/miyungpa/{process,process_client,visualizer}.py` | `convert_avi_to_mp4` from `util.py`. |

So the **pipeline** (`RawVideoProcessor`) is only driven from the upload apps; the `util.py`
converters are independent utilities anyone can import.

---

## The pipeline (`raw_video_processor.py`)

### `RawVideoProcessor` — the orchestrator
Runs on each capture PC. On construction it scans for raw videos and sets up a
`multiprocessing.Pool` (≤ 4 workers) with a `Manager`-shared `progress_dict` for live status.

```python
from paradex.video.raw_video_processor import RawVideoProcessor
rvp = RawVideoProcessor()      # scans capture_path_list for raw .avi
rvp.process()                  # fan out across the pool (non-blocking dispatch)
rvp.wait_and_monitor()         # block + print per-video progress until done
# or poll: rvp.get_progress() / rvp.get_status_summary() / rvp.finished()
rvp.stop()                     # pool.terminate()
```

`get_progress()` returns `{video_id: {status, progress, current_frame, total_frames, fps, eta,
avg_read_ms, avg_undistort_ms, avg_write_ms, message}}`. `status` moves through
`pending → starting → loading_params → processing → uploading → completed | failed`.

### `undistort_raw_video(video_path, progress_dict, video_id)` — the per-video worker
1. **Resolve paths** from the raw layout. Two supported layouts (serial = filename stem):
   - new: `.../{session}/raw/{stage}/videos/{serial}.avi`
   - old: `.../{session}/raw/videos/{serial}.avi` (stage = `""`)
   - Output goes to `.../{session}/videos/{stage}/{serial}.avi` locally, then `{shared_dir}/{root_name}/videos/{stage}/{serial}.avi` on the NAS.
2. **Skip / resume**: if the output exists with the **same frame count**, skip re-encode → just rsync + delete. If frame count mismatches, it's an incomplete file → delete and reprocess.
3. **Load calibration**: `load_camparam(shared_dir/root_name)` → `precompute_undistort_map_torch(intrinsics[serial])` builds a GPU `grid_tensor`.
4. **Frame loop** — for each frame:
   - **Drop-frame detection**: the camera writes a checkerboard marker into the top-left 30×30 patch for dropped frames (even pixels > 240, odd pixels < 15). Detected → replaced with a **black frame** (keeps frame indices aligned). Otherwise `apply_undistort_torch(frame, grid_tensor)`.
   - Pipe the raw `bgr24` bytes into an **`ffmpeg` `h264_nvenc`** subprocess (NVIDIA GPU encode, `-cq 19 -preset p4 -tune hq`, GOP 30). ffmpeg stderr → `{out_path}.ffmpeg.log`.
5. **Finish**: close stdin, `wait()` ffmpeg. On broken pipe or nonzero rc → status `failed`, delete partial output, keep the log. On success → `rsync_copy` to NAS, then **delete both the local output and the raw source**.

### `get_raw_videopath_list()`
Globs every root in `capture_path_list` for both layouts (`**/raw/*/videos/*.avi` and
`**/raw/videos/*.avi`). This is what `RawVideoProcessor` enumerates at construction.

---

## `util.py` — standalone ffmpeg converters
Independent of the pipeline; each shells out to `ffmpeg` with `subprocess.run(check=True)`.
| Function | Does |
|----------|------|
| `change_to_h264(inp, out)` | Re-encode any input → libx264 `crf 17` (high quality), yuv420p. Does **not** delete input. |
| `images_to_video(image_files, out_mp4, fps)` | Sorted image list → mp4 (OpenCV `mp4v` temp → libx264 `crf 23` re-encode → deletes temp). |
| `convert_avi_to_mp4(inp, out)` | AVI → mp4, libx264 `crf 23` `preset fast`, keeps audio (aac 192k). |
| `convert_avi_to_compressed_avi(inp, out)` | AVI → compressed AVI, libx264 `crf 23`, strips audio (`-an`). |

---

## Gotchas for editors
- **`h264_nvenc` requires an NVIDIA GPU + nvenc-enabled ffmpeg.** The pipeline hard-codes it — no CPU (`libx264`) fallback. On a box without nvenc, every video ends `failed` with the reason in `{out_path}.ffmpeg.log`. `util.py` uses CPU `libx264` instead and is portable.
- **The worker deletes the raw source on success** (`os.remove(video_path)`). It's a move, not a copy — a bug in the NAS path or rsync can lose data. The frame-count skip guard is the only safety net.
- **Drop-frame detection is a heuristic** on the top-left 30×30 pixels (a marker the camera writer stamps). If the raw writer's marker convention changes, this misfires — see `agent_docs/camera_system/` for the writer side.
- Layout parsing keys entirely off a `raw/` path component and the presence/absence of a `{stage}` dir between `raw/` and `videos/`. Non-conforming paths resolve to wrong output locations silently.
- `load_camparam` reads `shared_dir/root_name` (the **NAS** copy of the calibration), where `root_name` is the capture dir with `capture_path_list` prefixes stripped — so the calibration must already be on the NAS for the session when this runs.
- `num_workers = min(4, cpu_count())`; each worker also spawns an ffmpeg process and touches the GPU, so raising the cap can oversubscribe the encoder.
