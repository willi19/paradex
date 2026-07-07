# Intrinsic Calibration

Per-camera intrinsic calibration (K + distortion). Operators move a Charuco
board through each camera's view; every Capture PC detects the board locally
and **auto-accumulates** a frame whenever the board lands somewhere new on that
camera's image plane. There is **no save button** — coverage is collected
automatically. The Main PC only shows a live preview with each camera's
kept-frame count so you know where to still point the board.

Run this **before** extrinsic (extrinsic's `calculate.py` calls
`load_current_intrinsic()`).

## Scripts
| File | Purpose |
|------|---------|
| `capture.py` | **Main PC** orchestrator. Launches `client.py` on Capture PCs, shows a merged live stream with the current detection (green) and per-camera `kept/target` count. No save key — press **`q`** to finish (this tells the clients to write their keypoint files). |
| `client.py` | **Capture PC** daemon (launched by `capture.py`). Reads cameras, runs Charuco detection, and keeps a frame only when the **full board** is visible and its mean per-corner displacement is > `NOVELTY_PX` from every frame already kept for that camera (until `MAX_FRAMES`) — the proven `should_save` rule from the old distributed version. Publishes downscaled JPEG + current corners + kept count. On `exit`, saves one keypoint `.npy` per camera. |
| `calculate.py` | Offline solver (run where `~/shared_data` is visible). Loads the latest keypoint file per serial, runs `cv2.calibrateCamera` with per-frame NaN masking, prints RMS + sanity warnings, and writes `param/<ts>.json`. |

## Usage
1. On the **Main PC** (Capture PC camera daemons must be up):
   ```bash
   python src/calibration/intrinsic/capture.py
   ```
   In the "Intrinsic Capture" window:
   - move the Charuco board across each camera's full frame — corners, edges, tilts;
   - watch each tile's `kept/target` count climb (frames are accepted automatically as coverage grows);
   - press **`q`** when every camera reads e.g. `60/60`. The Capture PCs then write their keypoint files.
2. Solve (auto-detects every serial that captured, or pass `--serials`):
   ```bash
   python src/calibration/intrinsic/calculate.py
   python src/calibration/intrinsic/calculate.py --serials 25305460 25305462
   ```

## Tuning (in `client.py`)
- `BOARD` — charuco board id from `charuco_info.json` (default `"3"`, 11×8 → 70 corners).
- `NOVELTY_PX` — min mean per-corner displacement (full-res px) to every kept frame; lower = denser capture (default 10). Only full-board detections are kept.
- `MAX_FRAMES` — per-camera cap; capture stops accepting once reached (default 60).

## Inputs & Outputs
- Keypoints: `~/shared_data/intrinsic/<serial>/keypoint/<ts>.npy`, shape `(N, num_corners, 1, 2)`, NaN for undetected corners.
- Result: `~/shared_data/intrinsic/<serial>/param/<ts>.json` = `{K, distortion, RMS_error, width, height}` — the format `paradex.calibration.utils.load_current_intrinsic()` reads back.

## Related
- `paradex/image/aruco.py` — `detect_charuco`, `get_charuco_detector`, `_charuco_board_cache`, `draw_charuco`.
- `paradex/io/camera_system/{remote_camera_controller,camera_reader}.py`, `paradex/io/capture_pc/{ssh,data_sender,command_sender}.py`.
- `paradex.calibration.utils` — `intrinsic_dir`, `load_current_intrinsic`.
- Next step: [`../extrinsic/`](../extrinsic/).
