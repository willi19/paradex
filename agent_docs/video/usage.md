# paradex/video — How to Call Each Helper

Recipes for the post-capture video layer. Two files: `raw_video_processor.py` (the distributed
undistort+encode+upload **pipeline**) and `util.py` (standalone ffmpeg **converters**). Import each
symbol from its submodule — the package `__init__.py` is empty.

> Editing the pipeline, the drop-frame heuristic, or the delete-on-success path? Read [`internals.md`](internals.md).

---

## 1. The pipeline — [`raw_video_processor.py`](../../paradex/video/raw_video_processor.py)

Runs **on each capture PC** after a session: finds raw `.avi`, undistorts + fixes dropped frames,
GPU H.264 re-encodes, rsyncs to the NAS, deletes local copies. Driven from the upload apps
(`src/util/upload_video/`, `src/validate/upload_raw_video/`), not during capture.

```python
from paradex.video.raw_video_processor import RawVideoProcessor

rvp = RawVideoProcessor()      # scans capture_path_list for raw .avi at construction
rvp.process()                  # fan out across the pool (dispatch + one status print)
rvp.wait_and_monitor()         # block, re-print progress every 2s until done
rvp.stop()                     # pool.terminate() + join
```

Non-blocking polling instead of `wait_and_monitor()`:

| Call | Returns |
|------|---------|
| `get_progress()` | `{video_id: {status, progress, current_frame, total_frames, fps, eta, avg_read_ms, avg_undistort_ms, avg_write_ms, message}}` |
| `get_overall_progress()` | `float` mean progress % across all videos |
| `get_status_summary()` | `{pending, starting, loading_params, processing, uploading, completed, failed}` counts |
| `finished()` | `bool` — all async results `ready()` |
| `print_progress()` | prints the formatted dashboard once |

`status` moves through `pending → starting → loading_params → processing → uploading → completed | failed`.

### `get_raw_videopath_list()` — standalone enumerator
```python
from paradex.video.raw_video_processor import get_raw_videopath_list
paths = get_raw_videopath_list()   # globs both raw layouts across capture_path_list
```
Matches both `**/raw/*/videos/*.avi` (new, with a `{stage}` dir) and `**/raw/videos/*.avi` (old,
stage-less). This is what `RawVideoProcessor` enumerates at construction.

---

## 2. Standalone ffmpeg converters — [`util.py`](../../paradex/video/util.py)

Independent of the pipeline; each shells out to `ffmpeg` via `subprocess.run(check=True)` and uses
**CPU `libx264`** (portable — no GPU needed). Called from `src/process/miyungpa/` and similar.

```python
from paradex.video.util import (
    change_to_h264, images_to_video, convert_avi_to_mp4, convert_avi_to_compressed_avi,
)
```

| Function | Does | Notes |
|----------|------|-------|
| `change_to_h264(inp, out)` | any input → libx264 `crf 17`, `preset slow`, yuv420p | high quality; does **not** delete input |
| `images_to_video(image_files, out_mp4, fps)` | sorted image list → mp4 | OpenCV `mp4v` temp → libx264 `crf 23` re-encode → deletes temp |
| `convert_avi_to_mp4(inp, out)` | AVI → mp4, libx264 `crf 23` `preset fast` | keeps audio (aac 192k) |
| `convert_avi_to_compressed_avi(inp, out)` | AVI → compressed AVI, libx264 `crf 23` | strips audio (`-an`) |

```python
convert_avi_to_mp4("raw.avi", "out.mp4")     # the one miyungpa uses
images_to_video(sorted(glob("frames/*.png")), "clip.mp4", 30)
```

> On ffmpeg failure these **print and return** (they catch `CalledProcessError`) rather than raise —
> check the output file exists if you depend on it. Details in [`internals.md`](internals.md).
