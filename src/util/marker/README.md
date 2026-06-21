# ChArUco Marker Generation

Generates printable ChArUco calibration boards as A4-sized PDFs.

## Scripts
| File | Purpose |
|------|---------|
| `generate_charuco.py` | Renders a 5x6 ChArUco board (DICT_6X6_250, 60mm squares / 40mm markers) to a 300-DPI PDF centered on A4, plus a parameters `.txt`. |

## Usage
```bash
python src/util/marker/generate_charuco.py
```
No CLI args. Board count and starting marker-ID offset are hard-coded in `__main__` (`num_boards = 1`, `current_marker_id = 0`). Edit `generate_charuco_board_pdf()` to change `squares_x/y`, `square_length`, `marker_length`, or the ArUco dictionary.

## Inputs & Outputs
- Inputs: none (parameters are in code).
- Outputs (relative to CWD):
  - `outputs/charuco/<YYYYMMDD_HHMMSS>/charuco_board_<n>.pdf`
  - `outputs/charuco/<YYYYMMDD_HHMMSS>/charuco_board_<n>_parameters.txt`
  - `outputs/charuco_board_<n>_ids_<start>-<end>.pdf` (copy for easy access)
  - `test.png` (debug raster of the board, written to CWD)

## Related
- Detection / calibration consumers: [`paradex/image/`](../../../paradex/image) (ArUco detection), [`src/calibration/`](../../calibration) (intrinsic/extrinsic workflows).
- The board image is drawn manually with `cv2.aruco.generateImageMarker`; the noted ID-offset is informational only (OpenCV ChArUco IDs are sequential from 0).
