# System Overview

Paradex is a distributed pipeline for robot experiment data. It helps an
operator collect synchronized camera and robot data, calibrate the rig, process
recordings, and check pose or grasp results.

## What Paradex Does

Paradex is easiest to understand as the coordination layer around a robot
manipulation rig:

1. It starts from a selected rig profile in `system/current`.
2. It validates that capture PCs, camera services, command transport, and trigger
   timing are working.
3. It calibrates cameras and camera-to-robot geometry.
4. It records synchronized camera, robot, hand, and teleoperation streams.
5. It turns raw sessions into synchronized videos, undistorted frames,
   reconstructions, and upload-ready outputs.
6. It supports pose, grasp, visualization, and robot-readiness checks on top of
   those outputs.

```text
configure rig
  -> validate transport, cameras, and sync
    -> calibrate intrinsic / extrinsic / hand-eye geometry
      -> capture synchronized sessions
        -> process videos, masks, reconstructions, uploads
          -> check pose estimates, grasp transforms, overlays, and robot readiness
```

## Machine Roles

| Role | Runs where | Responsibility |
|---------|------------|----------------|
| Main PC | operator workstation | Orchestrates capture PCs, runs validation entry points, launches remote capture, aggregates status, and starts processing jobs. |
| Capture PCs | camera machines | Run FLIR/PySpin camera services and return frame/status data. |
| Robot/control host | robot-side host or main PC, depending on setup | Runs arm and hand controllers, robot state streaming, teleoperation, kinematics, and motion checks. |
| Shared storage | usually mounted as `~/shared_data` | Stores calibration, raw sessions, processed videos, uploads, and reusable caches. |

The main PC normally does not open camera hardware directly. It sends commands to
capture-PC services, and those services control the camera SDK lifecycle.

## Code Layout

| Path | Role |
|------|------|
| `paradex/` | Reusable library modules: camera IO, capture-PC transport, calibration utilities, robot wrappers, transforms, visualization, processing helpers. |
| `src/` | Runnable workflows that combine library modules into real tasks. Start with `src/README.md` in the repository. |
| `system/` | Rig configuration. `system/current/` selects the active profile. |
| `rsc/` | Robot URDFs, hand models, meshes, and other static resources. |
| `docs/` | Sphinx guide and generated API pages. |
| `agent_docs/` | Task-oriented subsystem notes for AI coding agents. |

## Practical Setup Order

1. Confirm `system/current/{pc,camera,network}.json` and shared storage.
2. Start `src/camera/server_daemon.py` on every capture PC.
3. Run transport and camera validation from `src/validate/`.
4. Run intrinsic and extrinsic calibration; add hand-eye calibration when
   camera-to-robot geometry is required.
5. Run a short capture and inspect the output layout before collecting a full
   dataset.
6. Run processing and output checks only after the raw session layout and calibration
   files are known good.

Use {doc}`camera_system` for camera failures, {doc}`calibration` for geometry or
projection failures, and the repository's `src/validate/README.md` for subsystem
quick checks.
