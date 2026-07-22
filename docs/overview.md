# System Overview

Paradex is a distributed data pipeline for multi-camera robot experiments. It
coordinates Capture PCs, camera services, hardware-triggered cameras, robot and
hand streams, calibration, session recording, and post-processing. The output is
a recorded session that can be inspected, processed into datasets, or checked for
pose and grasp tasks.

## Core Vocabulary

| Term | Meaning |
|------|---------|
| Rig profile | The selected robot/camera setup under `system/current`, including PC names, camera serials, network settings, and storage paths. |
| Main PC | The operator workstation that launches validation scripts, sends remote camera commands, monitors status, and starts processing jobs. |
| Capture PC | A machine connected to one or more FLIR cameras. It owns camera hardware through a local camera service. |
| Camera service | A long-running process on a Capture PC, such as `src/camera/server_daemon.py`, that owns the camera SDK and responds to remote commands. |
| Hardware trigger | A shared timing signal, typically from the UTGE900, that makes cameras expose frames on the same pulse. |
| Session | One recorded experiment folder containing raw camera data and any robot, hand, teleoperation, timestamp, and calibration snapshots. |
| Intrinsic calibration | Per-camera lens and sensor parameters used for undistortion and projection. |
| Extrinsic calibration | Camera-to-camera or camera-to-scene geometry used to relate views in one coordinate frame. |
| Hand-eye calibration | The camera-to-robot transform used when camera observations must align with robot motion or robot-frame overlays. |
| Post-processing | Work after recording: timestamp matching, undistortion, video encoding, reconstruction, upload, and dataset preparation. |

## Machine Roles

| Role | Runs where | Responsibility |
|------|------------|----------------|
| Main PC | operator workstation | Selects the active rig profile, launches validation and orchestration scripts, sends commands to Capture PCs, aggregates status, and starts processing jobs. |
| Capture PCs | camera machines | Run camera services, own the FLIR/PySpin camera lifecycle, and return frame/status data to the Main PC. |
| Robot/control host | robot-side host or Main PC, depending on setup | Runs arm and hand drivers, robot state streaming, teleoperation, kinematics, and motion checks. |
| Shared storage | usually mounted as `~/shared_data` | Stores calibration files, raw sessions, processed datasets, uploads, and reusable caches. |

The Main PC normally does not open camera hardware directly. It sends
commands to Capture PC camera services, and those services control the camera SDK
lifecycle.

## Pipeline Stages

```text
select rig profile
  -> verify services and transport
    -> calibrate geometry
      -> record synchronized session
        -> post-process data
          -> inspect pose, grasp, overlays, and robot readiness
```

1. **Select rig profile**: choose `system/current`; verify `pc.json`,
   `camera.json`, `network.json`, and shared storage.
2. **Verify services**: check Python imports, SSH launch, command/data transport,
   camera services, and hardware-trigger timing.
3. **Calibrate geometry**: solve intrinsics and extrinsics; add hand-eye
   calibration when camera-to-robot geometry is required.
4. **Record session**: capture synchronized camera, robot, hand, and
   teleoperation streams under shared storage.
5. **Post-process data**: match timestamps, undistort frames, encode videos,
   reconstruct scenes, upload outputs, and prepare datasets.
6. **Inspect results**: check pose estimates, grasp transforms, calibration
   overlays, visualizers, and robot readiness.

## Repository Map

| Path | Role |
|------|------|
| `paradex/` | Reusable library modules: camera IO, Capture PC transport, calibration utilities, robot wrappers, transforms, visualization, and processing helpers. |
| `src/` | Runnable workflows that combine library modules into real tasks. Start with `src/README.md` in the repository. |
| `system/` | Rig configuration. `system/current/` selects the active profile. |
| `rsc/` | Robot URDFs, hand models, meshes, and other static resources. |
| `docs/` | Sphinx guide and generated API pages. |
| `agent_docs/` | Task-oriented subsystem notes for AI coding agents. |

## Practical Setup Order

1. Confirm the active rig profile: `system/current/pc.json`,
   `system/current/camera.json`, `system/current/network.json`, and shared
   storage.
2. Start `src/camera/server_daemon.py` on every Capture PC.
3. Run transport and camera validation from `src/validate/`.
4. Run intrinsic and extrinsic calibration. Add hand-eye calibration when
   robot-frame overlays or robot motion checks depend on camera observations.
5. Record a short session and inspect the folder layout before collecting a full
   dataset.
6. Run post-processing and output checks only after the raw session layout and
   calibration files are known good.

Use {doc}`camera_system` for camera-service failures, {doc}`calibration` for
geometry or projection failures, and the repository's `src/validate/README.md`
for subsystem validation checks.
