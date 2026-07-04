# `src/` — Applications

Application scripts that combine `paradex/` library modules into concrete workflows.
Each application group has its own `README.md` (for humans) and `CLAUDE.md` (orientation for Claude). Start there for run commands and gotchas.

> **Distributed system note:** Many scripts come in pairs — a **Capture-PC** daemon/client that waits for commands, and a **Main-PC** orchestrator (often suffixed `_remote` or `_main`) that drives it over SSH/TCP. Each app's README states which side runs where.

## Application groups

| Group | What it does |
|-------|--------------|
| [calibration/](calibration/README.md) | Camera intrinsic/extrinsic + hand-eye (camera→robot) calibration. Run order: intrinsic → extrinsic → handeye. |
| [camera/](camera/README.md) | Capture-PC daemons (`server_daemon.py` camera command server, `monitor_daemon.py` status dashboard). |
| [capture/](capture/README.md) | Image/video/stream capture (camera) and robot teleop/teaching data collection. |
| [dataset_acquisition/](dataset_acquisition/README.md) | Per-dataset capture pipelines (graphics, motion_blur, hri, miyungpa, object_turntable). |
| [inference/](inference/README.md) | 6D pose + grasp inference/evaluation pipelines (bodex, grasp_eval, grasp_w_gui, pringles_test). |
| [object6d/](object6d/README.md) | Multi-view 6D object pose estimation (LoFTR/template matching) + camera-to-robot validation. |
| [process/](process/README.md) | Post-capture processing (miyungpa: sync+overlay+contact; object_turntable: COLMAP reconstruction). |
| [util/](util/README.md) | Operational utilities (charuco generation, object registration, URDF tools, video upload). |
| [validate/](validate/README.md) | System validation harnesses for cameras, sync, robots, teleop, and calibration quality. |

## Typical end-to-end flow

```
calibrate (intrinsic → extrinsic → handeye)
   → capture / dataset_acquisition (collect raw data)
      → process (sync, undistort, reconstruct)
         → inference (6D pose, grasp)
validate/ — sanity-check any stage in isolation
```

See the top-level [CLAUDE.md](../CLAUDE.md) for the "when doing X, look at Y" task map.
