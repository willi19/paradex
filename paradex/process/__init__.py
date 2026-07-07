"""
paradex.process — a small, declarative batch-processing framework.

Write a `discover()` that lists `Job` specs and a `process(job, ctx)` that does the
actual transform. The framework handles the boring-but-essential parts for you:

    skip-if-done  ->  download inputs from NAS (cache)  ->  run your transform
                  ->  upload outputs to NAS  ->  report status  ->  catch & record errors

Minimal example::

    from paradex.process import Job, run_jobs

    def discover():
        return [
            Job(id=f"{sess}/{serial}",
                inputs={"video": f"{sess}/raw/videos/{serial}.avi"},
                outputs={"video": f"{sess}/videos/{serial}.avi"},
                meta={"serial": serial})
            for sess, serial in find_work()
        ]

    def process(job, ctx):
        src = ctx.input("video")     # local cached path (already downloaded)
        dst = ctx.output("video")    # local path to write; uploaded automatically
        n = frame_count(src)
        for i, frame in enumerate(read(src)):
            ctx.status(f"{i}/{n}", progress=i / n)
            write(dst, transform(frame))

    run_jobs(discover(), process, num_workers=4)

All `inputs`/`outputs` paths are **relative to the NAS root** (`shared_dir`,
i.e. ``~/shared_data``). See `Processor` for live monitoring / a pluggable
`on_update` hook (e.g. to feed the ZMQ web dashboard).
"""

from paradex.process.processor import Job, Ctx, Processor, run_jobs, Skip

# Distributed helpers are optional (need zmq + capture-PC config). Import lazily
# so single-machine use never fails if those aren't available.
try:
    from paradex.process.distributed import serve_jobs, run_distributed, shard
    __all__ = ["Job", "Ctx", "Processor", "run_jobs", "Skip",
               "serve_jobs", "run_distributed", "shard"]
except Exception:  # pragma: no cover - zmq / pc config missing
    __all__ = ["Job", "Ctx", "Processor", "run_jobs", "Skip"]
