# src/process — Post-Processing Pipelines

Post-capture data processing for the Paradex framework. Each subdirectory processes raw captured data (videos, robot states, charuco frames) from a specific capture pipeline into derived artifacts (synced sensor data, overlay videos, per-frame masks, COLMAP reconstructions, turntable rotations).

## Pipelines
| Directory | Purpose |
|-----------|---------|
| [`miyungpa/`](miyungpa/) | Sync robot arm/hand states to video frames, render robot-mesh overlay videos, compute object-contact maps, and serve a dataset web/3D viewer. |
| [`object_turntable/`](object_turntable/) | Turntable object scan: extract frames from videos, triangulate charuco for per-frame turntable rotation, segment object masks (SAM3), and build a COLMAP reconstruction. |
| [`object_turntable/check/`](object_turntable/check/) | Validation/cleanup helper scripts for the turntable pipeline (image integrity, frame-drop removal, COLMAP/mask presence, camera-pose visualization). |

## Notes
- All scripts hard-code their dataset roots (e.g. `~/shared_data/capture/miyungpa`, `~/paradex_download/capture/object_turntable`) and iterate over `<obj_name>/<index>` subdirectories. Many also hard-code the specific object names they process — edit the loop at the bottom of each script before running.
- Two data roots are used: `shared_dir` (`~/shared_data`, the shared NAS) and `home_path/paradex_download` (local working copy). The turntable pipeline runs against the local copy; the miyungpa pipeline downloads from shared, processes locally, and uploads results back.

## Related
- Capture side: [`src/dataset_acquisition/miyungpa`](../dataset_acquisition/miyungpa), [`src/dataset_acquisition/object_turntable`](../dataset_acquisition/object_turntable)
- paradex modules: `paradex/video/`, `paradex/image/`, `paradex/calibration/`, `paradex/dataset_acqusition/` (note the spelling), `paradex/visualization/`
