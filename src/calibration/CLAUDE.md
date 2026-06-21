# CLAUDE.md — src/calibration

## Purpose
Calibration application scripts. Pipeline order is fixed: intrinsic → extrinsic → handeye. This dir's only script patches the xArm URDFs from the controller's factory kinematics.

## Files
- `xarm_kinematic_calibration.py` — `main()` calls `read_xarm_kinematic_params(ip)`, `save_kinematic_yaml(...)`, then `apply_kinematics_to_urdf(urdf, kinematics)` per URDF. Prints per-joint xyz/rpy deltas, warns on large deltas (>20mm / >5deg). `--no_apply` skips patching.

## paradex modules used
- `paradex.robot.xarm_kinematic_calib` — read/save/apply kinematic params.
- `paradex.utils.system.network_info` — default robot IP.

## Data flow & IO
Live xArm (IP) → YAML at `~/shared_data/xarm_kinematics/<robot_name>_kinematics_<suffix>.yaml` → in-place patch of `rsc/robot/{xarm,xarm_allegro,xarm_inspire}.urdf` (backups `*.urdf.original`).

## When working here
- Subdirs `extrinsic/` and `handeye/` each have their own CLAUDE.md — read those when working there.
- `REPO_ROOT` resolved relative to this file; `YAML_DIR` is hardcoded under `~/shared_data`.

## Gotchas
- Patches URDFs in place — re-running re-patches from `.original`-less current files only if no backup logic intervenes; rely on `apply_kinematics_to_urdf` backup behavior.
- Joint-name mismatch yields empty diff ("no joints patched"), not an error.
- Calibration ordering: do not run handeye before a valid extrinsic exists.
