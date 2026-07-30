"""Main-PC orchestrator for raw-video undistort + upload.

SSH-launches `worker.py` on every capture PC (each processes its own local raw
videos) and shows live progress — per-PC counts, per-video frame progress
(frames @fps), and a rig-wide ETA — until all PCs finish.

Two views over the same aggregated status:
    python src/util/upload_video/main.py            # console dashboard
    python src/util/upload_video/main.py --web      # + browser page on :8080

Run:
    python src/util/upload_video/main.py [--web] [--web-port 8080] [--quiet]
"""

import argparse

from paradex.process import run_distributed

WORKER = "python src/util/upload_video/worker.py"

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--web", action="store_true",
                    help="serve the browser dashboard (http://<main-pc>:PORT)")
    ap.add_argument("--web-port", type=int, default=8080)
    ap.add_argument("--quiet", action="store_true",
                    help="suppress the console dashboard (implies --web)")
    ap.add_argument("--linger", type=float, default=300.0,
                    help="seconds to keep the web page up after all PCs finish")
    args = ap.parse_args()

    web = args.web or args.quiet
    run_distributed(
        WORKER,
        web_port=args.web_port if web else None,
        console=not args.quiet,
        linger=args.linger if web else 0.0,
    )
