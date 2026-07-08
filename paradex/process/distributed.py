"""Distributed execution for paradex.process.

Same `Job` / `process(job, ctx)` you already wrote, now spread across the capture
PCs. Each PC discovers *its own* local jobs and processes them; the main PC
launches the workers over SSH and aggregates live status over ZMQ.

    capture PC:   serve_jobs(discover_local(), process)   # publishes status
    main PC:      run_distributed("python src/process/my_worker.py")

This is a thin layer over :class:`paradex.process.Processor`: the worker just
wires the processor's ``on_update`` hook to a :class:`DataPublisher`, and the main
PC reuses :func:`run_script` + :class:`DataCollector`. The core framework is
unchanged and stays usable single-machine.

Status wire format — each job is published as one item keyed by ``job.id`` (must
be unique across PCs; session/serial ids already are). Each PC also publishes a
summary sentinel named ``_pc::{pc}`` carrying its counts and a ``finished`` flag,
which the main PC uses to know when every PC is done.
"""

import time
from typing import Callable, List, Optional

from paradex.utils.path import pc_name
from paradex.utils.system import get_pc_list
from paradex.io.capture_pc.ssh import run_script
from paradex.io.capture_pc.data_sender import DataPublisher, DataCollector
from paradex.process.processor import (
    Processor, Job, DONE_STATUSES, fmt_dur, fmt_job, batch_eta,
)

DEFAULT_PORT = 1234
SENTINEL_PREFIX = "_pc::"


# --------------------------------------------------------------------------- #
# work partitioning
# --------------------------------------------------------------------------- #
def shard(jobs: List[Job], pc_list: Optional[List[str]] = None,
          pc: Optional[str] = None) -> List[Job]:
    """Return only the jobs this PC owns, by stable hash of ``job.id``.

    Use when every PC discovers the **same** job list (e.g. all reading shared
    NAS). If instead each PC discovers its own *local* files, you don't need this
    — the data is already partitioned by machine.
    """
    pc_list = sorted(pc_list or get_pc_list())
    pc = pc or pc_name
    if pc not in pc_list:
        return jobs  # not a known capture PC; run everything locally
    idx = pc_list.index(pc)
    n = len(pc_list)
    # zlib.crc32 is stable across processes/runs (unlike hash()).
    import zlib
    return [j for j in jobs if zlib.crc32(j.id.encode()) % n == idx]


# --------------------------------------------------------------------------- #
# worker side (runs on each capture PC)
# --------------------------------------------------------------------------- #
def _progress_to_items(progress: dict, pc: str, num_workers: int = 1) -> List[dict]:
    """Flatten a Processor progress dict into DataPublisher items.

    Carries the derived timing fields (``elapsed``/``fps``/``eta``/``frame``/``total``/
    ``started_at``) so the main PC's dashboard and any web UI show frame-level progress
    and ETA without recomputing anything.
    """
    items = []
    counts = {}
    for jid, info in progress.items():
        st = info.get("status", "pending")
        counts[st] = counts.get(st, 0) + 1
        items.append({
            "name": jid,
            "pc": pc,
            "status": st,
            "progress": info.get("progress", 0.0),
            "message": info.get("message", ""),
            "frame": info.get("frame"),
            "total": info.get("total"),
            "fps": info.get("fps"),
            "elapsed": info.get("elapsed"),
            "eta": info.get("eta"),
            "started_at": info.get("started_at"),
        })
    finished = all(info.get("status") in DONE_STATUSES for info in progress.values())
    items.append({
        "name": f"{SENTINEL_PREFIX}{pc}",
        "pc": pc,
        "status": "summary",
        "counts": counts,
        "total": len(progress),
        "num_workers": num_workers,
        "eta": batch_eta(progress, num_workers),
        "finished": finished and len(progress) > 0,
    })
    return items


def serve_jobs(jobs: List[Job], process_fn: Callable, num_workers: int = 4,
               port: int = DEFAULT_PORT, keep_cache: bool = False,
               name: Optional[str] = None) -> dict:
    """Run ``jobs`` on this PC and publish live status over ZMQ for the main PC.

    Drop-in replacement for :func:`paradex.process.run_jobs` on a capture PC.
    Blocks until all local jobs finish, then sends a final ``finished`` summary.
    """
    pc = name or pc_name
    publisher = DataPublisher(port=port, name=pc)

    def on_update(progress: dict):
        publisher.send_data(_progress_to_items(progress, pc, num_workers), [])

    try:
        proc = Processor(jobs, process_fn, num_workers=num_workers,
                         keep_cache=keep_cache, on_update=on_update, quiet=True)
        progress = proc.run()
        # Final flush so the main PC always sees the terminal state + finished flag.
        on_update(progress)
        time.sleep(0.3)
        return progress
    finally:
        publisher.close()


# --------------------------------------------------------------------------- #
# main side (orchestrator)
# --------------------------------------------------------------------------- #
def run_distributed(worker_cmd: str, pc_list: Optional[List[str]] = None,
                    port: int = DEFAULT_PORT, log: bool = True,
                    poll_interval: float = 1.0, timeout: Optional[float] = None) -> dict:
    """Launch ``worker_cmd`` on every capture PC and aggregate status until done.

    Args:
        worker_cmd:  shell command run on each PC (must call :func:`serve_jobs`),
                     e.g. ``"python src/process/my_worker.py"``.
        pc_list:     PCs to use (default: all from ``get_pc_list()``).
        port:        ZMQ port (must match the workers' ``serve_jobs`` port).
        log:         write remote stdout to ``test.log`` on each PC (see run_script).
        poll_interval: dashboard refresh / poll seconds.
        timeout:     give up after this many seconds (None = wait indefinitely).

    Returns the final aggregated ``{job_id_or_sentinel: item}`` mapping.
    """
    pc_list = pc_list or get_pc_list()
    collector = DataCollector(pc_list=pc_list, port=port)
    collector.start()
    # Subscribe before workers start publishing (ZMQ PUB drops messages with no peer).
    time.sleep(0.3)

    run_script(worker_cmd, pc_list=pc_list, log=log)

    start = time.monotonic()
    try:
        while True:
            data = collector.get_data()
            _print_dashboard(data, pc_list)
            if _all_finished(data, pc_list):
                print("\n[paradex.process] all PCs finished.")
                break
            if timeout is not None and (time.monotonic() - start) > timeout:
                print("\n[paradex.process] timeout reached; stopping monitor.")
                break
            time.sleep(poll_interval)
        return collector.get_data()
    finally:
        collector.end()


def _all_finished(data: dict, pc_list) -> bool:
    sentinels = {k: v for k, v in data.items() if k.startswith(SENTINEL_PREFIX)}
    if len(sentinels) < len(pc_list):
        return False  # haven't heard from every PC yet
    return all(v.get("finished") for v in sentinels.values())


def _print_dashboard(data: dict, pc_list):
    sentinels = {k[len(SENTINEL_PREFIX):]: v
                 for k, v in data.items() if k.startswith(SENTINEL_PREFIX)}
    jobs = {k: v for k, v in data.items() if not k.startswith(SENTINEL_PREFIX)}

    tot = {"completed": 0, "skipped": 0, "failed": 0}
    for v in jobs.values():
        st = v.get("status")
        if st in tot:
            tot[st] += 1
    # Whole-rig ETA = the slowest still-working PC (they run in parallel).
    pc_etas = [s.get("eta") for s in sentinels.values()
               if not s.get("finished") and s.get("eta") is not None]
    rig_eta = f" | rig ETA {fmt_dur(max(pc_etas))}" if pc_etas else ""
    print(f"\n[paradex.process] {len(jobs)} jobs | "
          f"done={tot['completed']} skip={tot['skipped']} fail={tot['failed']}{rig_eta}")
    for pc in pc_list:
        s = sentinels.get(pc)
        if s is None:
            print(f"    {pc}: (no report yet)")
        else:
            c = s.get("counts", {})
            flag = "DONE" if s.get("finished") else "running"
            eta = s.get("eta")
            eta_str = "" if s.get("finished") or eta is None else f" ETA {fmt_dur(eta)}"
            print(f"    {pc}: {flag} {dict(c)} of {s.get('total', 0)}{eta_str}")
    # Live per-job lines for whatever is currently processing (frame-of-N + ETA).
    running = [(jid, v) for jid, v in jobs.items() if v.get("status") == "processing"]
    for jid, v in sorted(running)[:12]:
        print(f"    [{v.get('pc')}] {jid}: {fmt_job(v)}")
    for jid, v in sorted(jobs.items()):
        if v.get("status") == "failed":
            print(f"    ✗ [{v.get('pc')}] {jid}: {v.get('message', '')}")
