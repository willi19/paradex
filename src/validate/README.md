# src/validate — System Validation Harnesses

Standalone scripts that exercise individual subsystems of the distributed Paradex
rig (cameras, signal generator, robot, teleop, networking) to confirm they work
end-to-end before a real capture session. These are diagnostic/smoke-test tools,
not part of the production capture pipeline.

## Subsystems
| Directory | What it validates |
|-----------|-------------------|
| [`calibration/`](calibration/) | Re-evaluates hand-eye / kinematic calibration quality and camera pose drift across sessions |
| [`camera_system/`](camera_system/) | Flir/PySpin camera control, multi-cam loader, frame readers, hardware sync (UTGE900 signal generator + timestamp monitor) |
| [`command_sender/`](command_sender/) | TCP command + data round-trip between main PC and capture PCs |
| [`data_sender/`](data_sender/) | Pub/sub data collection from capture PCs to the main PC |
| `robot/` | Robot arm motion / FK (covered in a separate doc pass) |
| `robot_controller/` | Arm + hand controller plumbing (separate doc pass) |
| `teleop/` | XSens teleop input (separate doc pass) |
| `upload_raw_video/` | Raw video upload path (separate doc pass) |
| `visualizer/` | Viser / Open3D viewer harnesses (separate doc pass) |

> This README covers the camera/network/calibration group. The robot-side
> subsystems (`robot/`, `robot_controller/`, `teleop/`, `upload_raw_video/`,
> `visualizer/`) are documented in a separate pass.

## Distributed-system shape
Most camera/network validators come in a **main-PC / capture-PC pair**: a `*_remote`
or `main`/`stream_remote` script runs on the main PC and SSHes a `*_client`/`client`
script onto the capture PCs (via `paradex.io.capture_pc.ssh.run_script`). Run the
main-PC script; it launches the capture-PC side for you.

## Related
- [`paradex/io/camera_system/`](../../paradex/io/camera_system) — the real camera stack these mirror
- [`paradex/io/capture_pc/`](../../paradex/io/capture_pc) — SSH, command_sender, data_sender
- [`paradex/calibration/`](../../paradex/calibration) — hand-eye solver, calib utils
