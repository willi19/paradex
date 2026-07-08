"""Core implementation of the paradex.process batch framework.

See :mod:`paradex.process` for the high-level overview and usage example.
"""

import os
import time
import traceback
from dataclasses import dataclass, field
from multiprocessing import Pool, Manager, cpu_count
from typing import Callable, Dict, List, Optional, Any

from paradex.utils.path import shared_dir, download_dir
from paradex.utils.upload_file import rsync_copy

# Cache root: inputs are downloaded here, outputs are built here before upload.
CACHE_ROOT = os.path.join(download_dir, "process_cache")

# Status lifecycle (a job moves through a subset of these):
#   pending -> [skipped]
#           -> caching -> processing -> uploading -> completed
#           -> failed   (at any point, with traceback recorded)
STATUSES = ("pending", "caching", "processing", "uploading",
            "completed", "skipped", "failed")
DONE_STATUSES = ("completed", "skipped", "failed")


class Skip(Exception):
    """Raise inside ``process(job, ctx)`` to stop and mark the job ``skipped``."""


@dataclass
class Job:
    """One unit of work.

    Attributes:
        id:      Unique id. Used as the status key and the per-job cache subdir,
                 so keep it filesystem-safe (slashes are fine, they nest dirs).
        inputs:  ``{name: nas_relpath}`` — files pulled from NAS (``shared_dir``)
                 into the local cache before ``process`` runs. Reach them with
                 ``ctx.input(name)``.
        outputs: ``{name: nas_relpath}`` — files your ``process`` writes locally
                 (path from ``ctx.output(name)``) and which are uploaded to NAS
                 after it returns successfully.
        meta:    Arbitrary extra data for your ``process`` fn (``job.meta[...]``).
        done:    Optional ``(job) -> bool`` completion predicate. This is an
                 intentional idempotency extension point for jobs whose real
                 completion cannot be proven by output-file existence alone
                 (for example multi-file manifests or external quality gates).
                 When supplied it is authoritative; otherwise the conservative
                 default skips only when *all* declared ``outputs`` already
                 exist on NAS. For ``num_workers > 1`` the callable must be
                 picklable, just like ``process_fn``.
    """
    id: str
    inputs: Dict[str, str] = field(default_factory=dict)
    outputs: Dict[str, str] = field(default_factory=dict)
    meta: Dict[str, Any] = field(default_factory=dict)
    done: Optional[Callable[["Job"], bool]] = None


class Ctx:
    """Handle passed to ``process(job, ctx)``.

    Gives the transform its cached input paths, its output paths to write, and a
    ``status()`` reporter — nothing else. Caching, uploading, skipping and error
    capture are handled by the framework around it.
    """

    def __init__(self, job: Job, progress_dict, cache_dir: str):
        self.job = job
        self._progress = progress_dict
        self._cache_dir = cache_dir
        # Resolved local paths, filled by the framework during the caching step.
        self._inputs: Dict[str, str] = {}

    def input(self, name: str) -> str:
        """Local path of a cached input (already downloaded from NAS)."""
        return self._inputs[name]

    def output(self, name: str) -> str:
        """Local path where you should write output ``name`` (parent dir ensured).

        Whatever you write here is uploaded to ``job.outputs[name]`` on NAS after
        ``process`` returns.
        """
        rel = self.job.outputs[name]
        path = os.path.join(self._cache_dir, "out", os.path.basename(rel))
        os.makedirs(os.path.dirname(path), exist_ok=True)
        return path

    def status(self, message: str = "", progress: Optional[float] = None,
               frame: Optional[int] = None, total: Optional[int] = None):
        """Report progress for the live dashboard / ETA.

        Args:
            message:  free-text ("overlay", "colmap: matching", ...).
            progress: fraction 0..1 (stored as 0..100 percent). Optional if you
                      pass ``frame``/``total`` — it is derived as ``frame/total``.
            frame:    current item index (e.g. video frame). With ``total`` this
                      drives the "340/1200 frames" readout and a frame-rate-based
                      ETA (more accurate than a bare percentage).
            total:    total number of items (frames). Report it once; it sticks.

        Timing (``elapsed``/``fps``/``eta``) is computed for you from the job's
        processing start — you only report where you are, not how long it took.
        """
        upd: Dict[str, Any] = {}
        if message:
            upd["message"] = message
        if frame is not None:
            upd["frame"] = int(frame)
        if total is not None:
            upd["total"] = int(total)
        if progress is None and frame is not None and total:
            progress = frame / total
        if progress is not None:
            upd["progress"] = max(0.0, min(1.0, progress)) * 100.0
        _update(self._progress, self.job.id, upd)


# --------------------------------------------------------------------------- #
# internals
# --------------------------------------------------------------------------- #
def _update(progress_dict, job_id, updates):
    """Atomically merge ``updates`` into ``progress_dict[job_id]`` (Manager.dict-safe),
    stamping derived timing (``elapsed``/``fps``/``eta``) so every consumer — console
    table, ZMQ dashboard, web UI — reads the same precomputed numbers.

    ``eta`` is seconds-remaining, preferring a frame-rate estimate (``frame``/``total``)
    and falling back to a percentage-based one; ``None`` while it can't be estimated.
    """
    cur = dict(progress_dict.get(job_id, {}))
    cur.update(updates)

    started = cur.get("started_at")
    if started is not None:
        now = time.time()
        elapsed = max(0.0, now - started)
        cur["elapsed"] = elapsed
        prog = cur.get("progress")
        frame, total = cur.get("frame"), cur.get("total")
        if cur.get("status") in DONE_STATUSES:
            cur["eta"] = 0.0
        elif frame and total and elapsed > 0:
            rate = frame / elapsed                       # frames / sec
            cur["fps"] = rate
            cur["eta"] = (total - frame) / rate if rate > 0 else None
        elif prog and 0 < prog < 100 and elapsed > 0:
            cur["eta"] = elapsed * (100.0 - prog) / prog
        else:
            cur["eta"] = None

    progress_dict[job_id] = cur


def _is_done(job: Job) -> bool:
    """Return whether ``job`` can be skipped before caching/processing.

    The default rule is deliberately conservative and data-grounded: only
    declared NAS outputs prove completion. ``job.done`` is the explicit escape
    hatch for workflows whose completion record is not one output path; errors
    from that predicate are allowed to propagate so a bad predicate fails the job
    instead of silently bypassing work.
    """
    if job.done is not None:
        return bool(job.done(job))
    if not job.outputs:
        return False  # no declared outputs -> can't infer doneness, always run
    return all(os.path.exists(os.path.join(shared_dir, rel))
               for rel in job.outputs.values())


def _download(nas_rel: str, local: str):
    """Pull one file NAS->local, skipping if an identical copy is already cached."""
    nas = os.path.join(shared_dir, nas_rel)
    if not os.path.exists(nas):
        raise FileNotFoundError(f"NAS input missing: {nas}")
    if os.path.exists(local) and os.path.getsize(local) == os.path.getsize(nas):
        return  # already cached
    os.makedirs(os.path.dirname(local), exist_ok=True)
    if not rsync_copy(nas, local):
        raise RuntimeError(f"rsync download failed: {nas} -> {local}")


def _run_job(job: Job, process_fn, progress_dict, keep_cache: bool):
    """Worker body: skip -> cache -> process -> upload -> report. Never raises."""
    cache_dir = os.path.join(CACHE_ROOT, job.id)
    try:
        # 1) skip if already done on NAS
        if _is_done(job):
            _update(progress_dict, job.id,
                    {"status": "skipped", "progress": 100.0, "message": "already done"})
            return f"{job.id}: skipped"

        # 2) cache inputs from NAS
        _update(progress_dict, job.id, {"status": "caching", "message": "downloading inputs"})
        ctx = Ctx(job, progress_dict, cache_dir)
        for name, rel in job.inputs.items():
            local = os.path.join(cache_dir, "in", os.path.basename(rel))
            _download(rel, local)
            ctx._inputs[name] = local

        # 3) run the user's transform (stamp start so ETA/elapsed can be derived)
        _update(progress_dict, job.id,
                {"status": "processing", "message": "processing", "started_at": time.time()})
        process_fn(job, ctx)

        # 4) upload declared outputs to NAS
        _update(progress_dict, job.id, {"status": "uploading", "message": "uploading outputs"})
        for name, rel in job.outputs.items():
            local = ctx.output(name)
            if not os.path.exists(local):
                raise FileNotFoundError(
                    f"output '{name}' was declared but not written: {local}")
            nas = os.path.join(shared_dir, rel)
            if not rsync_copy(local, nas):
                raise RuntimeError(f"rsync upload failed: {local} -> {nas}")

        _update(progress_dict, job.id,
                {"status": "completed", "progress": 100.0, "message": "success"})
        return f"{job.id}: success"

    except Skip as e:
        _update(progress_dict, job.id,
                {"status": "skipped", "message": str(e) or "skipped"})
        return f"{job.id}: skipped ({e})"

    except Exception as e:  # any failure: record, keep the batch going
        tb = traceback.format_exc()
        _update(progress_dict, job.id,
                {"status": "failed", "message": f"{type(e).__name__}: {e}", "traceback": tb})
        return f"{job.id}: FAILED: {e}"

    finally:
        if not keep_cache:
            _rmtree_quiet(cache_dir)


def _rmtree_quiet(path):
    import shutil
    try:
        shutil.rmtree(path, ignore_errors=True)
    except Exception:
        pass


# --------------------------------------------------------------------------- #
# display helpers (shared by the console table and the distributed dashboard)
# --------------------------------------------------------------------------- #
def fmt_dur(seconds: Optional[float]) -> str:
    """Human-friendly duration: ``95`` -> ``"1m35s"``, ``3725`` -> ``"1h02m"``."""
    if seconds is None or seconds != seconds:  # None or NaN
        return "?"
    seconds = int(max(0, round(seconds)))
    if seconds < 60:
        return f"{seconds}s"
    if seconds < 3600:
        return f"{seconds // 60}m{seconds % 60:02d}s"
    return f"{seconds // 3600}h{(seconds % 3600) // 60:02d}m"


def fmt_job(info: dict) -> str:
    """One-line progress for a running job: percent, frame-of-N, elapsed, ETA, msg."""
    prog = info.get("progress", 0.0)
    parts = [f"{prog:.0f}%"]
    frame, total = info.get("frame"), info.get("total")
    if frame is not None and total:
        parts.append(f"{frame}/{total}")
        if info.get("fps"):
            parts[-1] += f" @{info['fps']:.0f}fps"
    elapsed = _live_elapsed(info)
    if elapsed is not None:
        parts.append(fmt_dur(elapsed))
    if info.get("eta") is not None:
        parts.append(f"ETA {fmt_dur(info['eta'])}")
    msg = info.get("message", "")
    tail = f" - {msg}" if msg else ""
    return " • ".join(parts) + tail


def _live_elapsed(info: dict) -> Optional[float]:
    """Elapsed for a job, recomputed live from ``started_at`` when still running."""
    started = info.get("started_at")
    if started is None:
        return info.get("elapsed")
    if info.get("status") in DONE_STATUSES:
        return info.get("elapsed")
    return max(0.0, time.time() - started)


def batch_eta(progress, num_workers: int) -> Optional[float]:
    """Estimate seconds until a whole batch finishes on ``num_workers`` parallel slots.

    Uses the mean wall-time of already-completed jobs as the per-job cost, so the
    estimate sharpens as the run proceeds. Falls back to the max in-flight job ETA
    early on (before any job has completed). ``None`` when nothing can be estimated.
    """
    vals = list(progress.values()) if hasattr(progress, "values") else progress
    durations = [v["elapsed"] for v in vals
                 if v.get("status") == "completed" and v.get("elapsed")]
    remaining = [v for v in vals if v.get("status") not in DONE_STATUSES]
    if not remaining:
        return 0.0
    if durations:
        avg = sum(durations) / len(durations)
        # work still queued/running, spread over the parallel slots
        done_frac = sum((v.get("progress", 0.0) / 100.0) for v in remaining)
        work_left = max(0.0, len(remaining) - done_frac)
        return work_left * avg / max(1, num_workers)
    etas = [v["eta"] for v in remaining if v.get("eta") is not None]
    return max(etas) if etas else None


# --------------------------------------------------------------------------- #
# public API
# --------------------------------------------------------------------------- #
class Processor:
    """Runs a batch of :class:`Job` through ``process_fn`` with caching, skip,
    upload, console reporting and per-job error capture.

    Args:
        jobs:        list of :class:`Job`.
        process_fn:  ``(job, ctx) -> None`` — must be a top-level (picklable)
                     function when ``num_workers > 1``.
        num_workers: process-pool size. ``1`` runs **in-process** (no pool), which
                     is the easiest way to debug — real tracebacks, no pickling.
        keep_cache:  keep per-job cache/output files after success (default drop).
        on_update:   optional ``(progress: dict) -> None`` hook called each refresh
                     with ``{job_id: {status, progress, message, ...}}`` — feed it
                     to the ZMQ dashboard, a log, etc.
        quiet:       suppress the built-in console table (e.g. when using
                     ``on_update`` for display).
    """

    def __init__(self, jobs: List[Job], process_fn: Callable, num_workers: int = 4,
                 keep_cache: bool = False, on_update: Optional[Callable] = None,
                 quiet: bool = False):
        self.jobs = jobs
        self.process_fn = process_fn
        self.num_workers = max(1, min(num_workers, cpu_count()))
        self.keep_cache = keep_cache
        self.on_update = on_update
        self.quiet = quiet

        self._manager = Manager()
        self.progress = self._manager.dict()
        self.log = self._manager.list()
        for job in jobs:
            self.progress[job.id] = {"status": "pending", "progress": 0.0,
                                     "message": "waiting", "meta": job.meta}
        self._pool = None
        self._results = []

    # -- lifecycle ---------------------------------------------------------- #
    def run(self, update_interval: float = 1.0):
        """Process every job, blocking until done, refreshing status each interval."""
        if not self.jobs:
            print("[paradex.process] no jobs to run.")
            return self.get_progress()

        if self.num_workers == 1:
            # In-process: easiest to debug; refresh after each job.
            for job in self.jobs:
                res = _run_job(job, self.process_fn, self.progress, self.keep_cache)
                self.log.append(res)
                self._refresh()
        else:
            self._pool = Pool(processes=self.num_workers)
            self._results = [
                self._pool.apply_async(
                    _run_job,
                    args=(job, self.process_fn, self.progress, self.keep_cache),
                    callback=self.log.append,
                    error_callback=lambda e: self.log.append(f"POOL ERROR: {e}"),
                )
                for job in self.jobs
            ]
            self._pool.close()
            while not self._finished():
                self._refresh()
                time.sleep(update_interval)

        self._refresh()
        self._print_summary()
        return self.get_progress()

    def stop(self):
        if self._pool:
            self._pool.terminate()
            self._pool.join()

    # -- queries ------------------------------------------------------------ #
    def get_progress(self) -> dict:
        return dict(self.progress)

    def summary(self) -> Dict[str, int]:
        out = {s: 0 for s in STATUSES}
        for v in self.progress.values():
            out[v.get("status", "pending")] = out.get(v.get("status", "pending"), 0) + 1
        return out

    def failures(self) -> Dict[str, dict]:
        return {jid: v for jid, v in self.progress.items() if v.get("status") == "failed"}

    # -- internals ---------------------------------------------------------- #
    def _finished(self) -> bool:
        return all(r.ready() for r in self._results)

    def _refresh(self):
        if self.on_update:
            try:
                self.on_update(self.get_progress())
            except Exception as e:
                print(f"[paradex.process] on_update hook error: {e}")
        if not self.quiet:
            self._print_table()

    def _print_table(self):
        s = self.summary()
        done = s["completed"] + s["skipped"] + s["failed"]
        eta = batch_eta(self.progress, self.num_workers)
        eta_str = f" | batch ETA {fmt_dur(eta)}" if eta is not None else ""
        print(f"\n[paradex.process] {done}/{len(self.jobs)} done | "
              f"done={s['completed']} skip={s['skipped']} fail={s['failed']} "
              f"run={s['processing']} cache={s['caching']} up={s['uploading']} "
              f"wait={s['pending']}{eta_str}")
        for jid, info in sorted(self.progress.items()):
            st = info.get("status", "?")
            if st in ("processing", "caching", "uploading"):
                print(f"    {jid}: {st} {fmt_job(info)}")
            elif st == "failed":
                print(f"    {jid}: FAILED - {info.get('message', '')}")

    def _print_summary(self):
        s = self.summary()
        print(f"\n[paradex.process] finished: "
              f"{s['completed']} ok, {s['skipped']} skipped, {s['failed']} failed.")
        for jid, info in self.failures().items():
            print(f"  ✗ {jid}: {info.get('message', '')}")


def run_jobs(jobs: List[Job], process_fn: Callable, num_workers: int = 4,
             keep_cache: bool = False, on_update: Optional[Callable] = None,
             quiet: bool = False) -> dict:
    """One-call helper: build a :class:`Processor`, run it, return the final progress.

    See :class:`Processor` for argument meanings.
    """
    return Processor(jobs, process_fn, num_workers=num_workers, keep_cache=keep_cache,
                     on_update=on_update, quiet=quiet).run()
