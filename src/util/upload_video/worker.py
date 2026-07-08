"""Raw-video undistort + upload, on the paradex.process framework.

One :class:`Job` per raw `.avi` found locally on this capture PC. The transform is
the proven :func:`undistort_raw_video` (NVENC ffmpeg pipe + torch undistort, then
rsync to NAS and delete the local source) — reused unchanged; this module only
adapts its per-frame progress into ``ctx.status`` so the framework dashboard shows
"340/1200 frames @28fps • ETA 1m35s" per video, aggregated per-PC with a rig ETA.

Data is local to each PC (raw videos live under ``capture_path_list``), so jobs are
already partitioned by machine — no ``shard`` needed.

Run:
    python src/util/upload_video/worker.py            # distributed (this PC)
    python src/util/upload_video/worker.py --local    # this machine only
"""

import sys

from paradex.process import Job, run_jobs, serve_jobs
from paradex.utils.path import home_path
from paradex.video.raw_video_processor import get_raw_videopath_list, undistort_raw_video


class _CtxProgress:
    """dict-like that forwards ``undistort_raw_video``'s progress into ``ctx.status``.

    ``undistort_raw_video`` writes ``progress_dict[video_id] = {current_frame,
    total_frames, status, message, ...}``; we translate those into frame-level
    status so the framework derives elapsed/fps/ETA itself.
    """

    def __init__(self, ctx):
        self.ctx = ctx
        self._d = {}
        self.failed = False

    def get(self, key, default=None):
        return self._d.get(key, default)

    def __getitem__(self, key):
        return self._d[key]

    def __setitem__(self, key, value):
        self._d[key] = value
        if value.get("status") == "failed":
            self.failed = True
        cf, tf = value.get("current_frame"), value.get("total_frames")
        msg = value.get("message", "")
        if tf:
            self.ctx.status(msg, frame=cf, total=tf)
        elif msg:
            self.ctx.status(msg)


def discover():
    """One Job per raw `.avi` on this PC (video id = path relative to home)."""
    jobs = []
    for vp in get_raw_videopath_list():
        vid = vp.replace(home_path, "").lstrip("/")
        jobs.append(Job(id=vid, meta={"video_path": vp}))
    return jobs


def process(job, ctx):
    shim = _CtxProgress(ctx)
    res = undistort_raw_video(job.meta["video_path"], shim, job.id)
    # undistort_raw_video self-manages NAS upload + cleanup and returns a summary
    # string; surface real failures to the framework as exceptions.
    if shim.failed or (isinstance(res, str) and "success" not in res and "already" not in res):
        raise RuntimeError(res)


if __name__ == "__main__":
    jobs = discover()
    if "--local" in sys.argv:
        run_jobs(jobs, process, num_workers=4)
    else:
        serve_jobs(jobs, process, num_workers=4)   # local-per-PC data → no shard
