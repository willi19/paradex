# object6d — Validation Scripts

Standalone sanity-check scripts for the 6D object pose capture rig: a minimal robot-data + image capture test, and a camera-to-robot (C2R) calibration validator that overlays the rendered robot onto captured images.

## Scripts
| File | Purpose |
|------|---------|
| `capture_test.py` | Interactive capture session combining robot (arm/hand) data recording (`CaptureSession`) with remote multi-camera image capture; saves cam params + C2R per take. |
| `validate_c2r_simple.py` | Validate C2R calibration by loading the actual arm/hand qpos from a scene, rendering the robot mesh, and overlaying it onto undistorted images via `ImageDict.project_mesh`. |

## Usage

### Capture test (`capture_test.py`)
```bash
python src/object6d/validate/capture_test.py --name my_session --arm xarm --hand inspire
```
Keyboard controls:
- `c` — start a take: begins robot recording, saves cam params + C2R, captures images to `shared_data/<name>/<timestamp>/raw`, then stops.
- `s` — clears the stop flag (used between takes).
- `q` — exit (closes capture session and camera controller).

`--arm` and `--hand` are optional (default `None`); omit to capture images only.

### C2R validation (`validate_c2r_simple.py`)
```bash
python src/object6d/validate/validate_c2r_simple.py --scene_path /abs/path/to/scene --arm xarm --hand inspire
```
Reads real arm/hand poses from the scene if present, otherwise falls back to zero qpos.

## Inputs & Outputs

- **`capture_test.py`**: writes robot data (via `CaptureSession`) + raw images under `shared_data/<name>/<timestamp>/`, plus cam params and `C2R.npy`. Each `c` press is timestamped (`YYYY-MM-DD_HH-MM-SS`).
- **`validate_c2r_simple.py`**:
  - Reads `C2R.npy`, cam params (`load_camparam`), undistorted images (or undistorts raw via `ImageDict` if missing), and optional `raw/arm/position.npy`, `raw/hand/action.npy` / `raw/hand/position.npy`.
  - Hand values are converted from Inspire units (0-2000) to radians via `parse_inspire`.
  - Builds `<arm>_<hand>.urdf` robot mesh, applies the `C2R` transform to the mesh, projects via `ImageDict.project_mesh`.
  - Writes per-camera overlays + `grid.jpg` to `<scene_path>/c2r_validation/`.

## Related
- Parent app: [`../`](..) — `validate_c2r.py` is the zero-qpos `BatchRenderer` sibling of `validate_c2r_simple.py`.
- [`paradex/dataset_acqusition/capture.py`](../../../paradex/dataset_acqusition/capture.py) — `CaptureSession`.
- [`paradex/image/image_dict.py`](../../../paradex/image/image_dict.py) — `ImageDict` (`from_path`, `set_camparam`, `undistort`, `project_mesh`).
- [`paradex/robot/inspire.py`](../../../paradex/robot/inspire.py) — `parse_inspire`.
- [`paradex/visualization/robot.py`](../../../paradex/visualization/robot.py) — `RobotModule`.
- [`paradex/calibration/utils.py`](../../../paradex/calibration/utils.py) — `load_camparam`, `save_current_*`.
