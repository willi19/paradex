# Video Processing

Overview of Paradex's video-processing subsystem — how raw captured `.avi` videos
become undistorted, frame-drop-corrected, NAS-hosted clips. Read this for the mental
model; for method signatures, parameters, and return values see the
{doc}`API reference <process_api>`.

- Core library: `paradex/video/` (`raw_video_processor.py`, `util.py`)
- Distributed driver: the `paradex.process` batch framework, wired by
  `src/util/upload_video/` (`main.py` + `worker.py`)
- Generated per-symbol API: {doc}`API Reference <autoapi/index>`

---

## 1. What the subsystem does

Capture writes **raw** videos straight off the cameras — lens-distorted, and with
occasional **dropped frames** (a frame the camera never delivered, encoded as a
sentinel checkerboard tile). Before that footage is useful for reconstruction or
overlay, each raw clip must be:

1. **Undistorted** — remap every frame through the camera's intrinsics (GPU
   `grid_sample`), so straight lines are straight.
2. **Frame-drop corrected** — detect the sentinel pattern and substitute a black
   frame, keeping every clip the same length and frame-aligned across cameras.
3. **Re-encoded** — pipe raw `bgr24` frames into `ffmpeg` (`h264_nvenc`) to a
   compact H.264 `.avi`.
4. **Uploaded** — `rsync` the result to shared NAS storage, then delete the local
   raw + temp files.

The unit of work is **one video = one worker**. A pool of workers chews through all
raw videos found on a capture PC; the whole thing is orchestrated per-PC and
monitored live from the main PC.

```{mermaid}
flowchart TB
    RAW["raw/{stage}/videos/{serial}.avi<br/>(distorted, dropped frames)"]
    RAW --> UND["undistort<br/>(GPU grid_sample)"]
    UND --> DROP["frame-drop correction<br/>(sentinel → black frame)"]
    DROP --> ENC["encode<br/>(ffmpeg h264_nvenc)"]
    ENC --> UP["rsync to NAS<br/>then delete local"]
    UP --> OUT["shared_data/.../videos/{stage}/{serial}.avi"]
```

---

## 2. Core Concepts

| Term | Meaning |
|------|---------|
| **Raw video** | Off-camera `.avi` under `.../raw/[{stage}/]videos/{serial}.avi`. Distorted; may contain dropped frames. |
| **Worker** | `undistort_raw_video(video_path, progress_dict, video_id)` — the per-video function run in a pool process. |
| **Dropped frame** | A frame the camera never delivered. Marked by a sentinel checkerboard in the top-left `30×30` patch (even pixels `>240`, odd pixels `<15`); replaced with an all-black frame. |
| **Stage** | Optional sub-phase in the new capture layout (`raw/{stage}/videos/…`). Empty string for the old stage-less layout. |
| **Progress dict** | A `multiprocessing.Manager().dict()` keyed by `video_id`; every worker writes its own status/progress into it so the parent can report without IPC plumbing. |
| **video_id** | The video path with `home_path` stripped — the stable key used in the progress dict and over the wire. |

---

## 3. Per-video worker pipeline

`undistort_raw_video` is the whole pipeline for a single clip. It is **idempotent
and self-cleaning**: if a finished output already exists with a matching frame count
it skips straight to upload; a partially-written output is deleted and reprocessed.

```{mermaid}
flowchart TB
    S["derive serial + stage<br/>from path"] --> CHK{"output exists &<br/>frame count matches?"}
    CHK -->|yes| SKIP["upload existing<br/>→ delete local → done"]
    CHK -->|"no / partial"| P["load intrinsics<br/>precompute undistort grid"]
    P --> FF["spawn ffmpeg (h264_nvenc)<br/>stdin = raw bgr24 pipe"]
    FF --> LOOP["for each frame"]
    LOOP --> D{"dropped?<br/>(sentinel patch)"}
    D -->|yes| BLACK["write black frame"]
    D -->|no| U["GPU undistort"]
    BLACK --> W["write frame → ffmpeg stdin"]
    U --> W
    W --> LOOP
    LOOP --> FIN["close pipe, wait ffmpeg"]
    FIN --> UP["rsync_copy → NAS<br/>delete out + raw"]
```

Each worker reports through `progress_dict[video_id]` with a `status` that walks
`pending → starting → (loading_params) → processing → uploading → completed`
(or `failed`), plus live `progress`, `current_frame`, `fps`, and `eta` refreshed
every 30 frames.

**Encoding note.** The active processor pipes raw frames into `ffmpeg` with NVENC
(`h264_nvenc`, `-cq 19`). The helper module `paradex/video/util.py` holds standalone
CPU `libx264` converters (`change_to_h264`, `convert_avi_to_mp4`,
`convert_avi_to_compressed_avi`, `images_to_video`) used by other scripts; they are
not on the `RawVideoProcessor` path.

---

## 4. Distributed processing model

Raw videos live on the **6 capture PCs** (that's where the cameras wrote them), so
processing runs *on each capture PC* and the **main PC** only orchestrates and
displays. The two sides talk over ZMQ (port `1234`) via
`paradex.io.capture_pc.data_sender`.

```{mermaid}
flowchart TB
    subgraph Main["Main PC"]
      RD["run_distributed<br/>(paradex.process)"] --> COL["DataCollector<br/>(ZMQ 1234)"]
      RD --> DASH["console dashboard<br/>(per-PC counts + rig ETA)"]
    end
    subgraph Cap["Capture PC (×6)"]
      W["worker.py<br/>serve_jobs(discover, process)"] --> URV["undistort_raw_video<br/>(per-video, worker Pool)"]
      W --> DP["DataPublisher<br/>(ZMQ 1234)"]
    end
    RD -. "run_script (SSH)<br/>launches worker.py" .-> W
    DP -- "progress items<br/>(status / % / frame / fps / eta)" --> COL
```

The upload_video driver is now the `paradex.process` batch framework (the old
Flask/SocketIO `VideoProgressMonitor` in `process.py` and `VideoProgressPublisher`
in `client.py` were removed).

- **Main PC** (`src/util/upload_video/main.py`): calls
  `run_distributed("python src/util/upload_video/worker.py")` — SSH-launches the
  worker on every capture PC, aggregates their ZMQ status via a `DataCollector`, and
  prints the shared `paradex.process` console dashboard (per-PC counts, per-video
  frame progress, and a rig-wide ETA) until all PCs finish.
- **Capture PC** (`src/util/upload_video/worker.py`): a `paradex.process` worker.
  `discover()` returns one `Job` per local raw `.avi`; `process(job, ctx)` calls
  `undistort_raw_video` **unchanged**, forwarding its per-frame `progress_dict`
  updates into `ctx.status(frame=…, total=…)` through a small `_CtxProgress` adapter
  so the framework derives elapsed/fps/ETA. Data is local per PC, so no `shard` is
  needed. `serve_jobs` publishes status over the standard `paradex.process` ZMQ
  channel.

For a single-PC run (no dashboard), `src/validate/upload_raw_video/upload_local.py`
still uses the retained `RawVideoProcessor` directly: `RawVideoProcessor().process()`
then `.wait_and_monitor()`.

---

## 5. Data paths

```{mermaid}
flowchart TB
    IN["capture PC (local)<br/>captures{i}/.../raw/{stage}/videos/{serial}.avi"]
    TMP["capture PC (temp)<br/>.../videos/{stage}/{serial}.avi"]
    NAS["shared NAS<br/>shared_data/{root}/videos/{stage}/{serial}.avi"]
    IN -->|"undistort + encode"| TMP
    TMP -->|"rsync_copy"| NAS
    TMP -.->|"deleted after upload"| X["(removed)"]
    IN -.->|"deleted after upload"| X
```

| Stage | Location | Notes |
|-------|----------|-------|
| Input | `captures{1,2}/…/raw/[{stage}/]videos/{serial}.avi` | Found by `get_raw_videopath_list()` globbing `capture_path_list`. Both new (`raw/{stage}/videos`) and old (`raw/videos`) layouts are matched. |
| Local temp | `{root_dir}/videos/{stage}/{serial}.avi` | H.264 output; removed after a successful upload. |
| NAS | `{shared_dir}/{root_name}/videos/{stage}/{serial}.avi` | `root_name` = capture root with the `capture_path_list` prefix stripped. Written via `rsync_copy`. |

The raw input is deleted only after the upload succeeds, so an interrupted run can be
safely re-run (idempotent skip / partial-file reprocess in §3).

---

## 6. Downstream consumers

`paradex/video` produces clean per-camera clips; the post-processing pipelines in
`src/process/` consume them (they are *not* part of this subsystem — see
{doc}`src/process <../src/process/CLAUDE>` for their own docs):

| Consumer | Uses the clips for |
|----------|--------------------|
| `src/process/miyungpa/` | Sync arm/hand streams to video, render robot-overlay / merged videos, compute object contact. |
| `src/process/object_turntable/` | Turntable object scan → charuco → rotation → SAM masks → COLMAP reconstruction. |

---

## 7. Component Reference

| Component | Location | Responsibility |
|-----------|----------|----------------|
| `RawVideoProcessor` | `paradex/video/raw_video_processor.py` | Discover raw videos, fan out a worker pool, aggregate/print progress. |
| `undistort_raw_video` | `paradex/video/raw_video_processor.py` | Per-video worker: undistort + drop-correct + encode + upload. |
| `get_raw_videopath_list` | `paradex/video/raw_video_processor.py` | Glob `capture_path_list` for raw `.avi` (both layouts). |
| `update_progress` | `paradex/video/raw_video_processor.py` | Safe read-modify-write of a `Manager.dict` entry. |
| `util.py` converters | `paradex/video/util.py` | Standalone CPU `libx264` encode/convert helpers (off the main path). |
| `worker.py` (`discover`/`process`) | `src/util/upload_video/worker.py` | Capture-PC `paradex.process` worker: one Job per raw video; reuses `undistort_raw_video` via a `_CtxProgress` adapter, no `shard`. |
| `main.py` (`run_distributed`) | `src/util/upload_video/main.py` | Main-PC orchestrator: SSH-launch workers, aggregate ZMQ status, print the shared console dashboard. |

---

## 8. File Reference

| Target | File |
|--------|------|
| Processor + worker + discovery | `paradex/video/raw_video_processor.py` |
| CPU encode/convert helpers | `paradex/video/util.py` |
| Undistort maps (CPU + GPU) | `paradex/image/undistort.py` (`precompute_undistort_map_torch`, `apply_undistort_torch`) |
| Camera intrinsics loader | `paradex/calibration/utils.py` (`load_camparam`) |
| NAS upload | `paradex/utils/upload_file.py` (`rsync_copy`) |
| ZMQ transport | `paradex/io/capture_pc/data_sender.py` (`DataPublisher` / `DataCollector`) |
| Distributed framework | `paradex/process/` (`run_distributed`, `serve_jobs`) |
| Main-PC orchestrator | `src/util/upload_video/main.py` |
| Capture-PC worker | `src/util/upload_video/worker.py` |
| Single-PC entry (retained `RawVideoProcessor`) | `src/validate/upload_raw_video/upload_local.py` |

Method-by-method API (parameters / returns): {doc}`Video Processing — API <process_api>`.
