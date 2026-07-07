# CLAUDE.md — src/calibration

## Purpose
Calibration application scripts. Pipeline order is fixed: intrinsic → extrinsic → handeye. This dir also holds the ChArUco board PDF generator and the xArm URDF kinematic patcher.

## Files
- `generate_board.py` — print-ready ChArUco board PDF at **exact physical scale**. Renders with OpenCV's own `cv2.aruco.CharucoBoard.generateImage()` from the **same board object `paradex.image.aruco` detects with** (never hand-drawn), then feeds the raster back through `detect_charuco` and refuses to write unless all corners are recovered (self-check). Page == board size. Defaults to board `"3"` (11×8, 6X6_1000); `--square-mm` (default 50 → 550×400mm, marker 35mm at board 3's 0.7 ratio), `--dpi`, `--margin-mm`, `--out`. **`setLegacyPattern` is taken straight from `charuco_info.json` (the source of truth) — the generator never overrides it, so a generated board can't diverge from what detection expects.** Prints the matching `charuco_info.json` entry (metric) but does **not** modify config. Supersedes `src/util/marker/generate_charuco.py` (that one hand-places markers → legacy-pattern mismatch risk; the burn this fixes).
- `xarm_kinematic_calibration.py` — `main()` calls `read_xarm_kinematic_params(ip)`, `save_kinematic_yaml(...)`, then `apply_kinematics_to_urdf(urdf, kinematics)` per URDF. Prints per-joint xyz/rpy deltas, warns on large deltas (>20mm / >5deg). `--no_apply` skips patching.

## paradex modules used
- `paradex.robot.xarm_kinematic_calib` — read/save/apply kinematic params.
- `paradex.utils.system.network_info` — default robot IP.

## Data flow & IO
Live xArm (IP) → YAML at `~/shared_data/xarm_kinematics/<robot_name>_kinematics_<suffix>.yaml` → in-place patch of `rsc/robot/{xarm,xarm_allegro,xarm_inspire}.urdf` (backups `*.urdf.original`).

## When working here
- Subdirs `intrinsic/`, `extrinsic/`, and `handeye/` each have their own CLAUDE.md — read those when working there.
- `REPO_ROOT` resolved relative to this file; `YAML_DIR` is hardcoded under `~/shared_data`.

## Gotchas
- Patches URDFs in place — re-running re-patches from `.original`-less current files only if no backup logic intervenes; rely on `apply_kinematics_to_urdf` backup behavior.
- Joint-name mismatch yields empty diff ("no joints patched"), not an error.
- Calibration ordering: do not run handeye before a valid extrinsic exists.
