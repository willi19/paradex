# src/util — Operational Utilities

Assorted standalone utilities for the Paradex pipeline: calibration-board printing, object marker registration, robot URDF/visualization helpers, and a distributed video-processing monitor. Each subdirectory is independent.

## Subdirectories
| Directory | Purpose | Entry point |
|-----------|---------|-------------|
| [`marker/`](marker) | Generate printable ChArUco calibration boards (A4 PDF). | `generate_charuco.py` |
| [`register_object/`](register_object) | Compute object marker-offset tables (marker positions in the object frame) for 6D pose recovery. | `box.py`, `pringles.py` |
| [`robot/`](robot) | Build combined arm+hand URDFs, visualize link geometry/collision spheres, replay live arm pose in Viser. | `merge_urdf.py`, `replay.py`, `visualize.py` |
| [`upload_video/`](upload_video) | Distributed raw-video processing with a live Flask/SocketIO web dashboard. | `process.py` (main PC) + `client.py` (capture PC) |

## Quick commands
```bash
# Calibration board
python src/util/marker/generate_charuco.py

# Object registration (per object)
python src/util/register_object/box.py
python src/util/register_object/pringles.py

# Robot
python src/util/robot/merge_urdf.py --arm xarm --hand allegro
python src/util/robot/replay.py --arm xarm --hand allegro
python src/util/robot/visualize.py            # edit link/robot in source

# Video processing monitor (main PC; auto-launches clients via SSH)
python src/util/upload_video/process.py        # dashboard at http://localhost:8081
```

## Notes
- `robot/get_bounding_sphere.py` and `robot/replay_sim.py` are empty placeholders.
- See each subdirectory's `README.md` (humans) and `CLAUDE.md` (agents) for details.

## Related core modules
[`paradex/image/`](../../paradex/image), [`paradex/robot/`](../../paradex/robot), [`paradex/video/`](../../paradex/video), [`paradex/io/capture_pc/`](../../paradex/io/capture_pc), [`paradex/visualization/`](../../paradex/visualization).
