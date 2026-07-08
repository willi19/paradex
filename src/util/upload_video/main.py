"""Main-PC orchestrator for raw-video undistort + upload.

SSH-launches `worker.py` on every capture PC (each processes its own local raw
videos) and shows a live dashboard — per-PC counts, per-video frame progress
(frames @fps), and a rig-wide ETA — until all PCs finish.

Replaces the old Flask/SocketIO web monitor (process.py) + DataPublisher client
(client.py) with the shared paradex.process dashboard.
"""

from paradex.process import run_distributed

if __name__ == "__main__":
    run_distributed("python src/util/upload_video/worker.py")
