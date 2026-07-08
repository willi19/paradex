"""Main-PC orchestrator for miyungpa processing.

SSH-launches `worker.py` on every capture PC (each takes its shard) and shows a
live aggregated dashboard — per-PC counts, per-demo frame progress, and a rig-wide
ETA — until all PCs finish. Replaces the old hand-rolled ZMQ REQ/REP distributor
(process_main.py / process_client.py).
"""

from paradex.process import run_distributed

if __name__ == "__main__":
    run_distributed("python src/process/miyungpa/worker.py")
