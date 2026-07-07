# CLAUDE.md — src/calibration/intrinsic

## Purpose
Per-camera intrinsic calibration (K + distortion). Distributed capture like
`extrinsic/`, but with **no save button**: each Capture PC auto-accumulates a
charuco frame whenever the board is novel on that camera's image plane. First
step of the pipeline (intrinsic → extrinsic → handeye).

## Files
- `capture.py` — **Main PC**. `run_script("python src/calibration/intrinsic/client.py")` to start clients; `remote_camera_controller("intrinsic_calibration")` + `DataCollector` + `CommandSender`; `rcc.start("stream", ...)`. Loop: decode JPEG previews, draw current corners (green), `merge_image` display with `img_text` = `kept/target` per camera. `q` → quit; `cs.end()` sends `exit` so clients save. No save key, no `--args`.
- `client.py` — **Capture PC**. `MultiCameraReader` (reads daemon shared memory) + `detect_charuco`; `fixed_corner_array` → `(70,2)` with NaN. `try_accumulate` keeps a frame only if the **full board** is visible (no NaN) and its mean per-corner displacement is >= `NOVELTY_PX` from every kept frame, and `len < MAX_FRAMES` — same rule as the old distributed `should_save` (`len(ids)!=70 → skip`, mean corner dist < 10 → skip). `DataPublisher(port=1234)` streams downscaled JPEG ('image', with `kept`/`target` meta) + current corners ('charuco_detection', /8 float32). `CommandReceiver(port=6890)` with `exit` event. On exit, saves `intrinsic/<serial>/keypoint/<ts>.npy` shape `(N,70,1,2)` then `dp.close()`.
- `calculate.py` — offline. `calibrate_one`: load latest `keypoint/*.npy`, mask NaN per frame, `cv2.calibrateCamera`, `sanity_warn`, write `intrinsic/<serial>/param/<ts>.json` (`{RMS_error,K,distortion,width,height}`). `--serials` optional (defaults to every serial with a `keypoint/` dir); `--board`/`--width`/`--height`/`--min-corners`.

## paradex modules used
- `paradex.image.aruco` — `detect_charuco`, `get_charuco_detector`, `_charuco_board_cache`, `draw_charuco`.
- `paradex.io.camera_system.{remote_camera_controller,camera_reader}`; `paradex.io.capture_pc.{ssh,data_sender,command_sender}`.
- `paradex.calibration.utils.intrinsic_dir` (= `~/shared_data/intrinsic`).

## Data flow & IO
Daemon fills shared memory → `client.py` reads, detects, auto-accumulates per camera, streams preview/count to Main PC → on `q`/`exit`, each Capture PC writes `keypoint/<ts>.npy` locally → `calculate.py` reads latest keypoint per serial → `param/<ts>.json`. `load_current_intrinsic()` later reads the latest `param/*.json` per serial and derives the undistort matrix via `getOptimalNewCameraMatrix`.

## When working here
- `capture.py` and `client.py` are coupled by data types ('image' /8, 'charuco_detection' /8 float32) and ports (1234 publish, 6890 command). Change both together.
- Board default is `"3"` (11×8 → 70 corners, unitless `checkerLength`); object-point scale is irrelevant to intrinsics (only corner geometry matters).
- Capture policy lives in `client.py` constants (`NOVELTY_PX`, `MAX_FRAMES`) — not CLI args. Lower `NOVELTY_PX` for denser coverage. Only full-board frames are kept (matches the old should_save requiring all 70 corners).

## Gotchas
- Distributed: `capture.py` MUST run on the Main PC; it spawns `client.py` on Capture PCs via SSH `run_script`. Camera daemons must already be up (rcc drives them).
- `q` alone doesn't save — the save happens on the `exit` command that `cs.end()` sends; don't remove the `CommandSender`/`cs.end()`.
- Keypoints are full-res; only the preview + streamed corners are downscaled /8.
- `calculate.py` needs ≥5 usable frames per camera or it raises; recapture with more spread if a camera fails.
