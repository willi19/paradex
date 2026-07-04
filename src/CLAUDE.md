# CLAUDE.md — src/

Application scripts. Each leaf dir has its own `CLAUDE.md` — read that before editing code in it.

## Orientation
- `src/` combines `paradex/` modules; it should not define general-purpose APIs (those live in `paradex/`).
- Distributed pattern: a Capture-PC daemon/client (e.g. `src/camera/server_daemon.py`, `*_remote` *client* side) waits for commands; a Main-PC orchestrator (`*_remote.py`, `*_main.py`) sends them via `paradex.io.capture_pc` (SSH/TCP). Confirm which side a script is before changing IO.
- Camera serials are string keys everywhere; multi-cam images are `Dict[serial_str, np.ndarray]`.

## Group index
| Group | Read its CLAUDE.md for |
|-------|------------------------|
| `calibration/` | intrinsic→extrinsic→handeye order, COLMAP solve, Tsai-Lenz C2R |
| `camera/` | capture-PC daemons (ZMQ command server, monitor dashboard) |
| `capture/` | image/video/stream main↔capture split, robot teleop/teaching |
| `dataset_acquisition/` | `CaptureSession`-based dataset pipelines + output layouts |
| `inference/` | 6D pose → grasp → IK → execute backbone; several scratch files |
| `object6d/` | LoFTR/template 6D pose, C2R validation (needs external `_object_6d_tracking`) |
| `process/` | post-capture stage orders; some files have known bugs (flagged in their docs) |
| `util/` | charuco gen, object marker-offset registration, URDF merge, video upload |
| `validate/` | per-subsystem smoke tests mirroring `paradex.io` modules |

## When working here
- Adding a new capture pipeline → mirror `dataset_acquisition/*/capture.py` and reuse `CaptureSession`.
- Touching camera IO → the source of truth is `paradex/io/camera_system`; `src/validate/camera_system` mirrors/tests it.
- Several `asdf.py`, `*_template.py`, empty/0-byte, and known-buggy files exist — each group's CLAUDE.md flags them; don't treat them as canonical.
