# agent_docs/dataset_acquisition — agent orientation

Docs for **AI agents working on the dataset-acquisition library** (`paradex/dataset_acqusition/`
— the module name is misspelled, missing an 'i'; that is intentional, don't "fix" it). Routed
here from the repo-root `AGENTS.md`. Read **one** file for your task; don't scan everything.

This subsystem is the thin recording layer: `CaptureSession` fans one *start/stop* across the
camera rig + arm + hand + teleop and writes a fixed `raw/` layout; `match_sync` re-aligns the
per-sensor logs to the camera clock afterwards. It is *orchestration*, not device drivers —
the actual IO lives in `paradex/io/…` (camera, robot, teleop) and calibration save/load in
`paradex/calibration/`.

| Your task | Read |
|-----------|------|
| **Record** a session (drive `CaptureSession`) from a new/edited capture script | [`usage.md`](usage.md) §CaptureSession |
| **Post-process**: align arm/hand logs to the camera frames (`fill_framedrop` / `get_synced_data`) | [`usage.md`](usage.md) §match_sync |
| Change how `CaptureSession` wires devices / writes the `raw/` tree / runs the teleop loop | [`internals.md`](internals.md) |
| Change the frame-drop model or the 2-pointer matcher | [`internals.md`](internals.md) §match_sync |
| Understand a specific capture *script* (miyungpa / hri / turntable / motion_blur / graphics) | its `src/dataset_acquisition/<name>/CLAUDE.md` |

Rule of thumb: **calling** these two modules → `usage.md`; **editing** them → `internals.md`.

## Where the code lives
| Piece | File |
|-------|------|
| `CaptureSession` (record orchestrator) | [`paradex/dataset_acqusition/capture.py`](../../paradex/dataset_acqusition/capture.py) |
| `fill_framedrop`, `get_synced_data` (post-hoc sync) | [`paradex/dataset_acqusition/match_sync.py`](../../paradex/dataset_acqusition/match_sync.py) |
| Capture scripts built on `CaptureSession` | [`src/dataset_acquisition/`](../../src/dataset_acquisition/) (+ `src/capture/robot/teleop_real.py`) |
| The sync consumer | [`src/process/miyungpa/`](../../src/process/miyungpa/) |

Related subsystems: camera (`agent_docs/camera_system/`), calibration params (`agent_docs/calibration/`).
