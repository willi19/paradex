"""object_turntable reconstruction pipeline, on the paradex.process framework.

One :class:`Job` per scan (`capture/object_turntable/<obj>/<index>`), running the
five stages in order with per-stage progress:

    extract_video -> extract_charuco -> get_rotation -> extract_mask_sam3 -> generate_colmap

Why single-machine / in-process (``num_workers=1``):
  * data lives in the local ``paradex_download`` working copy (not sharded across PCs);
  * each stage already fans out internally over frames/cameras with its own
    ``mp.Pool`` — an in-process job lets those child pools spawn (a daemon pool
    worker could not) and gives real tracebacks for the GPU stages.

What the framework adds over the old hand-run scripts: skip-if-done, per-object
exception capture (one bad scan doesn't sink the batch), and a live status/ETA
dashboard across all scans.

Stage modules are imported lazily inside :func:`process` so ``discover()`` (and a
dry run) work without torch / sam3 / pycolmap present.

Run:
    python src/process/object_turntable/worker.py                 # all scans
    python src/process/object_turntable/worker.py big_green_spray  # only these objects
"""

import os
import sys
import glob

from paradex.process import Job, run_jobs
from paradex.utils.path import home_path

# stages live next to this file; make them importable however this is launched
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

TURNTABLE_REL = "capture/object_turntable"
STAGES = ["extract_video", "extract_charuco", "get_rotation",
          "extract_mask_sam3", "generate_colmap"]


def _root():
    return os.path.join(home_path, "paradex_download", TURNTABLE_REL)


def _colmap_done(job):
    return os.path.isdir(os.path.join(job.meta["demo_full"], "colmap"))


def discover(obj_filter=None):
    """One Job per `<obj>/<index>` scan in the local turntable working copy."""
    jobs = []
    root = _root()
    if not os.path.isdir(root):
        return jobs
    for obj_name in sorted(os.listdir(root)):
        if obj_filter and obj_name not in obj_filter:
            continue
        obj_dir = os.path.join(root, obj_name)
        if not os.path.isdir(obj_dir):
            continue
        for index in sorted(os.listdir(obj_dir)):
            demo_full = os.path.join(obj_dir, index)
            if not os.path.isdir(demo_full):
                continue
            jobs.append(Job(
                id=f"{obj_name}/{index}",
                meta={"obj_name": obj_name, "index": index, "demo_full": demo_full},
                done=_colmap_done,
            ))
    return jobs


def _stage(ctx, i, sub, label):
    """Report overall pipeline progress: stage ``i`` (of len(STAGES)) + sub-fraction."""
    ctx.status(label, progress=(i + sub) / len(STAGES))


def process(job, ctx):
    obj_name = job.meta["obj_name"]
    index = job.meta["index"]
    demo_full = job.meta["demo_full"]

    # lazy stage imports (pull torch / sam3 / pycolmap only when actually running)
    import extract_video, extract_charuco, get_rotation
    import extract_mask_sam3, generate_colmap

    # 1) extract frames from each camera video
    videos = sorted(glob.glob(os.path.join(demo_full, "videos", "*")))
    for k, vp in enumerate(videos):
        _stage(ctx, 0, k / max(1, len(videos)), f"extract_video {k + 1}/{len(videos)}")
        extract_video.process_video(vp)

    # 2) triangulate charuco per frame
    _stage(ctx, 1, 0.0, "extract_charuco")
    extract_charuco.process_task((obj_name, index))

    # 3) per-frame turntable rotation
    _stage(ctx, 2, 0.0, "get_rotation")
    get_rotation.process(demo_full)

    # 4) SAM3 object masks (GPU)
    _stage(ctx, 3, 0.0, "extract_mask_sam3")
    extract_mask_sam3.load_mask(demo_full)

    # 5) COLMAP reconstruction with known poses
    _stage(ctx, 4, 0.0, "generate_colmap")
    generate_colmap.colmap_demo(demo_full)

    ctx.status("done", progress=1.0)


if __name__ == "__main__":
    obj_filter = [a for a in sys.argv[1:] if not a.startswith("-")] or None
    jobs = discover(obj_filter)
    # in-process (num_workers=1): stages spawn their own pools + GPU stages need it.
    run_jobs(jobs, process, num_workers=1)
