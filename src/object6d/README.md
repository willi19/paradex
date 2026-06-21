# 6D Object Pose Capture & Tracking

Multi-view image capture and 6D object pose estimation for the Paradex rig. Captures synchronized images from all cameras, undistorts them, matches each view against pre-built object templates (LoFTR / template matching), and optimizes a multi-view object pose. Also includes camera-to-robot (C2R) validation tooling.

## Scripts
| File | Purpose |
|------|---------|
| `capture.py` | Main capture + multiview 6D pose pipeline. Optionally captures images (via remote camera controller), undistorts, builds a `Scene`, runs the matcher + optimizer, saves the pose. |
| `image_remote.py` | Interactive keyboard-driven multi-shot capture. Each `c` press captures a timestamped image set, saves cam params + C2R, and undistorts. |
| `capture_template.py` | Template/example variant of `image_remote.py` (saves cam params, no C2R). Treat as a configurable starting point, not a canonical entry point. |
| `undistort_template.py` | Template helper: `undistort()` loads an `ImageDict` from a path, undistorts, and saves cam params + C2R. Runnable via `--save_path`. |
| `loftr_client.py` | Capture-PC client that streams downscaled JPEG camera frames over a `DataPublisher` socket (for a remote LoFTR matching server). No argparse — edit ports in source. |
| `validate_c2r.py` | Validate C2R calibration by rendering the robot at zero qpos and overlaying its mask onto captured images (uses `BatchRenderer` + `overlay_mask`). |

`*_template.py` files are starting points/configs; `capture.py`, `image_remote.py`, `validate_c2r.py`, and `loftr_client.py` are the runnable entry points.

## Usage

All paths are relative to `~/shared_data` (`shared_dir`).

### Capture + pose estimation (`capture.py`) — main PC
Capture fresh images then estimate pose:
```bash
python src/object6d/capture.py --save_path object6d/mug_01 --obj_names mug
```
Or run on an already-captured folder (skips capture):
```bash
python src/object6d/capture.py --image_path object6d/mug_01 --obj_names mug
```
Key flags: `--obj_names` (required, one or more template object names), `--paircount`, `--inliers_threshold`, `--loss_thres`, `--saveimg`, `--vis_final`, `--debug`, `--max_workers`.
Requires the external `_object_6d_tracking` package (Scene, MatchProcessor, CaptureAndOptimizePipeline) on `PYTHONPATH`.

### Interactive image capture (`image_remote.py`) — main PC
```bash
python src/object6d/image_remote.py --save_path object6d/session1
```
Keyboard: `c` = capture one timestamped shot (saves cam param, C2R, raw images, then undistorts); `q` = quit.

### LoFTR streaming client (`loftr_client.py`) — capture PC
Run on each capture PC to stream live downscaled frames to a matching server:
```bash
python src/object6d/loftr_client.py
```
Publishes JPEG frames on port `1234`; listens for an `exit` command on port `6890`. Edit ports in the source.

### Undistort an existing folder (`undistort_template.py`)
```bash
python src/object6d/undistort_template.py --save_path object6d/mug_01
```

### Validate C2R (`validate_c2r.py`)
```bash
python src/object6d/validate_c2r.py --scene_path /abs/path/to/scene --arm xarm --hand inspire --frame 0
```

## Inputs & Outputs

- **Raw capture**: remote cameras write to `shared_data/<save_path>/raw/...` (or `.../<timestamp>/raw`). `save_current_camparam` / `save_current_C2R` write the active intrinsic/extrinsic + camera-to-robot transform alongside.
- **Undistortion**: `ImageDict.from_path(...).undistort(...)` produces an `images/` (or `undistort/`) folder used downstream.
- **Pose pipeline** (`capture.py`): builds a `Scene` (intrinsics, extrinsics, projection, batched renderer), runs `MatchProcessor` (template matching per view) + `CaptureAndOptimizePipeline`, and saves the result to `<scene_path>/obj_T_multiview_matching.pkl` (torch `.pkl`).
- **C2R validation** (`validate_c2r.py`): loads `C2R.npy` + cam params, renders the robot mesh, writes per-camera overlays and a `grid.jpg` to `<scene_path>/c2r_validation/`.
- **LoFTR client**: emits JPEG-encoded frames (downscaled 8x) over the network; no files written locally.

## Related
- [`paradex/image/image_dict.py`](../../paradex/image/image_dict.py) — `ImageDict` load / undistort / `project_mesh`.
- [`paradex/image/projection.py`](../../paradex/image/projection.py) — `BatchRenderer`.
- [`paradex/image/overlay.py`](../../paradex/image/overlay.py), [`paradex/image/grid.py`](../../paradex/image/grid.py) — overlay + grid helpers.
- [`paradex/calibration/utils.py`](../../paradex/calibration/utils.py) — `save_current_camparam`, `save_current_C2R`, `load_camparam`.
- [`paradex/io/camera_system/remote_camera_controller.py`](../../paradex/io/camera_system/remote_camera_controller.py) — orchestrates capture-PC cameras.
- [`paradex/io/camera_system/camera_reader.py`](../../paradex/io/camera_system/camera_reader.py), [`paradex/io/capture_pc/`](../../paradex/io/capture_pc/) — used by `loftr_client.py`.
- [`paradex/visualization/robot.py`](../../paradex/visualization/robot.py) — `RobotModule` (C2R validation).
- Sibling validation scripts: [`validate/`](validate/).
