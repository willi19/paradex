# Video Processing — API

Method reference for the video-processing stack: parameters (input) and return
values (output). For the architecture and how these fit together, see the
{doc}`overview <process>`.

Signatures are verified against the code. The subsystem lives in
`paradex/video/`; the distributed driver is now the `paradex.process` framework, wired by
`src/util/upload_video/` (`main.py` + `worker.py`).

Each entry is collapsed below — click to expand.

:::{dropdown} `RawVideoProcessor` (`paradex/video/raw_video_processor.py`)
:open:

Discovers every raw video on the host and processes them through a
`multiprocessing.Pool` of `undistort_raw_video` workers, aggregating live progress
through a `Manager().dict()`.

| Method | Input | Output | Description |
|--------|-------|--------|-------------|
| `RawVideoProcessor()` | — | instance | Glob raw videos (`get_raw_videopath_list()`), set `num_workers = min(4, cpu_count())`, allocate the shared `progress_dict` + `log`. Does **not** start work. |
| `.process()` | — | `None` | Open the pool and `apply_async` one `undistort_raw_video` per video (seeds each entry as `pending`), then `pool.close()` and print one progress snapshot. Non-blocking. |
| `.finished()` | — | `bool` | `True` once every async result `.ready()`. |
| `.get_progress()` | — | `dict` | Snapshot copy of the whole `progress_dict` (`{video_id: info}`). |
| `.get_overall_progress()` | — | `float` | Mean of every entry's `progress` (0–100); `0.0` if empty. |
| `.get_status_summary()` | — | `dict` | Count per status (`pending`/`starting`/`loading_params`/`processing`/`uploading`/`completed`/`failed`). |
| `.print_progress()` | — | `None` | Print overall %, status summary, and a per-video line to stdout. |
| `.wait_and_monitor(update_interval=2.0)` | `update_interval: float` | `None` | Block until `finished()`, printing progress every `update_interval` seconds; then print the final `log`. |
| `.stop()` | — | `None` | `pool.terminate()` + `join()` (abort in-flight workers). |
| `.async_callback(result)` | `result: str` | `None` | Pool success callback; appends the worker's return string to `log`. |
| `.error_callback(e)` | `e: Exception` | `None` | Pool error callback; appends `"ERROR in process: {e}"` to `log`. |

**Attributes**: `.videopath_list` (`list[str]`), `.num_workers` (`int`),
`.progress_dict` (`Manager.dict`), `.log` (`Manager.list`), `.pool`,
`.process_list` (`list[AsyncResult]`).
:::

:::{dropdown} Module functions (`paradex/video/raw_video_processor.py`)

| Function | Input | Output | Description |
|----------|-------|--------|-------------|
| `undistort_raw_video(video_path, progress_dict, video_id)` | `video_path: str`, `progress_dict: Manager.dict`, `video_id: str` | `str` | The per-video worker (§3 of the overview). Undistort + drop-correct + encode (ffmpeg NVENC) + `rsync` to NAS + delete locals. Returns a status string (`"... : success"` / `"... : already processed"` / a failure message). Never raises to the pool on ffmpeg/param failure — it records `status='failed'` and returns the reason. |
| `get_raw_videopath_list()` | — | `list[str]` | Glob every `capture_path_list` root for `**/raw/*/videos/*.avi` (new, staged) and `**/raw/videos/*.avi` (old, stage-less). |
| `update_progress(progress_dict, video_id, updates)` | `progress_dict: Manager.dict`, `video_id: str`, `updates: dict` | `None` | Safe read-modify-write of one `Manager.dict` entry (copy → update → reassign, since nested-dict mutation doesn't propagate through a `Manager`). |

**`progress_dict[video_id]` shape** (keys written by the worker):

| Key | Type | Meaning |
|-----|------|---------|
| `status` | `str` | `pending` → `starting` → `loading_params` → `processing` → `uploading` → `completed` / `failed`. |
| `video_path` | `str` | Absolute raw input path. |
| `current_frame` / `total_frames` | `int` | Frames written so far / total in the raw clip. |
| `progress` | `float` | `current_frame / total_frames * 100`. |
| `fps` / `eta` | `float` | Live processing rate and seconds remaining. |
| `avg_read_ms` / `avg_undistort_ms` / `avg_write_ms` | `float` | Per-stage timing over the last 30 frames. |
| `message` | `str` | Human-readable status line. |
:::

:::{dropdown} `util.py` converters (`paradex/video/util.py`)

Standalone CPU (`libx264`) encode/convert helpers. **Not** on the
`RawVideoProcessor` path; used by other scripts that assemble or transcode video.

| Function | Input | Output | Description |
|----------|-------|--------|-------------|
| `change_to_h264(temp_video_path, output_video_path)` | `temp_video_path: str`, `output_video_path: str` | `None` | Re-encode to H.264 `libx264 -preset slow -crf 17 -pix_fmt yuv420p`. Prints on failure (does not raise). |
| `images_to_video(image_files, output_video_path, frame_rate)` | `image_files: list[str]`, `output_video_path: str`, `frame_rate: int` | `None` | Write sorted images to an `mp4v` temp via `cv2.VideoWriter`, then re-encode to `libx264 -crf 23` and delete the temp. No-op if `image_files` is empty. |
| `convert_avi_to_mp4(input_path, output_path)` | `input_path: str`, `output_path: str` | `None` | `libx264 -preset fast -crf 23` + `aac` audio. |
| `convert_avi_to_compressed_avi(input_path, output_path)` | `input_path: str`, `output_path: str` | `None` | `libx264 -preset fast -crf 23 -an` (audio dropped). |
:::

:::{dropdown} upload_video driver — now `paradex.process` (`src/util/upload_video/`)

The old `VideoProgressPublisher` (`client.py`) and `VideoProgressMonitor`
(`process.py`, Flask/SocketIO) classes were **removed**. `undistort_raw_video` is now
driven by the `paradex.process` batch framework — one `Job` per raw video.

| Symbol | Location | Description |
|--------|----------|-------------|
| `discover()` / `process(job, ctx)` | `src/util/upload_video/worker.py` | Capture-PC worker: `discover()` returns one `Job` per local raw `.avi`; `process` calls `undistort_raw_video` **unchanged**, forwarding its `progress_dict` updates into `ctx.status(frame=, total=)` via a `_CtxProgress` adapter. Launched with `serve_jobs` (no `shard` — data is local per PC). |
| `run_distributed(worker_cmd)` | `src/util/upload_video/main.py` | Main-PC orchestrator: SSH-launch the worker on every capture PC and print the shared `paradex.process` console dashboard (per-PC counts, per-video frame/ETA, rig ETA). |

The framework API itself (`Job`, `Ctx`, `serve_jobs`, `run_distributed`, `shard`) is
documented under `agent_docs/process/`. The `DataPublisher`/`DataCollector` transport
(below) is unchanged. The standalone `RawVideoProcessor` (above) is **retained** and
still used directly by `src/validate/upload_raw_video/upload_local.py` for a single-PC
run.
:::

:::{dropdown} Transport dependency — `data_sender` (`paradex/io/capture_pc/data_sender.py`)

Referenced above; documented here for signature completeness.

| Method | Input | Output | Description |
|--------|-------|--------|-------------|
| `DataPublisher(port=1234, name=None)` | `port: int`, `name: str=None` | instance | Capture-PC ZMQ PUB endpoint. |
| `DataPublisher.send_data(metadata, data)` | `metadata: list[dict]`, `data: list[bytes]` | `None` | Publish one message; video progress passes `data=[]` (metadata only). |
| `DataPublisher.close()` | — | `None` | Close the socket. |
| `DataCollector(...)` | see module | instance | Main-PC SUB endpoint aggregating all publishers. |
| `DataCollector.start()` | — | `None` | Begin the receive loop. |
| `DataCollector.get_data(pc_name=None)` | `pc_name: str=None` | `dict` | Latest merged `{video_id: info}` (optionally one PC). |
:::
