# CLAUDE.md — src/util/marker

## Purpose
One-off generator that produces A4-printable ChArUco board PDFs for camera calibration.

## Files
- `generate_charuco.py` — `generate_charuco_board_pdf(board_number, marker_id_offset)` builds the board as a numpy grayscale array: draws the checkerboard, places `cv2.aruco.generateImageMarker` markers (DICT_6X6_250) into the black squares, then renders it centered on an A4 matplotlib figure saved as PDF. `__main__` loops `num_boards` (currently 1) and increments `marker_id_offset` per board.

## paradex modules used
None — pure `cv2.aruco`, `numpy`, `matplotlib`, `shutil`.

## Data flow & IO
- No inputs. Writes timestamped PDFs + params `.txt` under `outputs/charuco/<ts>/`, a flat copy under `outputs/`, and a `test.png` debug raster. All paths are relative to CWD.

## When working here
- Change board geometry via the constants at the top of `generate_charuco_board_pdf` (`squares_x/y`, `square_length`, `marker_length`).
- Change how many boards via `num_boards` in `__main__`.

## Gotchas
- Marker IDs: the manual draw uses `marker_id_offset`, but the printed comments warn that OpenCV's own ChArUco detector expects IDs starting at 0 — multi-board, non-overlapping IDs require offsetting detection in downstream code.
- `test.png` and `outputs/` are written to the current working directory, not a fixed shared path.
