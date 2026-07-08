# Calibration — Internals (for agents editing this module)

**You are here because you are changing how calibration is *produced*** — the capture or solve scripts
(intrinsic / extrinsic / hand-eye), the Charuco board generation, or the on-disk formats. If you only
want to *read params out* from another program, read [`usage.md`](usage.md) instead.

`paradex/calibration/` is just loaders + solvers ([`utils.py`](../../paradex/calibration/utils.py),
[`colmap.py`](../../paradex/calibration/colmap.py), [`Tsai_Lenz.py`](../../paradex/calibration/Tsai_Lenz.py));
the capture + solve orchestration lives in [`src/calibration/`](../../src/calibration/), one dir per
stage, each with its own `CLAUDE.md`. Pipeline order is fixed: **intrinsic → extrinsic → hand-eye**;
each stage consumes the previous stage's output.

| Stage | src dir | Solver / glue in `paradex/calibration/` | Output store |
|-------|---------|------------------------------------------|--------------|
| 1 intrinsic | [`src/calibration/intrinsic/`](../../src/calibration/intrinsic/CLAUDE.md) | `cv2.calibrateCamera` (in `calculate.py`) | `intrinsic/<serial>/param/<ts>.json` |
| 2 extrinsic | [`src/calibration/extrinsic/`](../../src/calibration/extrinsic/CLAUDE.md) | [`colmap.py`](../../paradex/calibration/colmap.py) (`COLMAPDatabase`, `load_colmap_camparam`) | `cam_param/<name>/{intrinsics,extrinsics}.json` |
| 3 hand-eye | [`src/calibration/handeye/`](../../src/calibration/handeye/CLAUDE.md) | [`Tsai_Lenz.py`](../../paradex/calibration/Tsai_Lenz.py) (`solve_axb_cpu`, `solve_ax_xb`) | `handeye_calibration/<name>/<idx>/C2R.npy` |

---

## Stage 1 — intrinsic (`src/calibration/intrinsic/`)

Distributed, **auto-capture, no save button**. Each Capture PC (`client.py`) reads its cameras from the
daemon's shared memory, detects the Charuco board, and keeps a frame only when the **full board** is
visible and its mean per-corner displacement is ≥ `NOVELTY_PX` from every frame already kept for that
camera (until `MAX_FRAMES`) — the proven `should_save` rule ported from the old distributed version.
Main PC (`capture.py`) shows a merged preview + per-camera `kept/target`; `q` sends `exit`, clients save
`keypoint/<ts>.npy` shape `(N, num_corners, 1, 2)`. `calculate.py` masks NaN per frame and runs
`cv2.calibrateCamera` per serial → `param/<ts>.json`. Board object-point scale is irrelevant to
K/distortion. See [`src/calibration/intrinsic/CLAUDE.md`](../../src/calibration/intrinsic/CLAUDE.md).

## Stage 2 — extrinsic (`src/calibration/extrinsic/`)

Distributed Charuco capture (`capture.py` + `client.py`, manual `c` to save all cameras' full-res
corners/ids/images) → `calculate.py` builds a COLMAP database from the 2D corners
([`COLMAPDatabase`](../../paradex/calibration/colmap.py)), runs `pycolmap.incremental_mapping`,
undistorts, triangulates the board, then a **refine pass** (keep corners with reproj err < 2 px into a
new timestamped dir) and re-solves, finally rescaling translations by `0.06 / mean_board_edge` →
`cam_param/<name>/{intrinsics,extrinsics}.json`. Intrinsics are seeded from `load_current_intrinsic()`
(stage 1). `load_colmap_camparam` reads the reconstruction back out into the intrinsic/extrinsic dicts.
See [`src/calibration/extrinsic/CLAUDE.md`](../../src/calibration/extrinsic/CLAUDE.md).

> **Why `ba_refine_focal_length/principal_point/extra_params=True`.** These are intentionally ON: with
> the stage-1 intrinsics held **fixed**, COLMAP incremental mapping **fails to converge**
> (registration/BA does not settle). Letting BA also refine intrinsics is what makes the solve converge.
> The tradeoff is that per-camera intrinsics can drift from stage 1, so intrinsic quality still matters
> (better stage-1 coverage → less drift). Do **not** naively flip these to False to "trust stage 1" —
> that reintroduces the non-convergence. A real fix for "fix intrinsics AND converge" needs a different
> solver (Charuco PnP-initialized multi-cam BA, where poses are seeded from known board geometry instead
> of SfM bootstrapping), not COLMAP incremental SfM.

## Stage 3 — hand-eye (`src/calibration/handeye/`)

Tsai-Lenz `AX = XB` from paired {camera-observed board pose, robot end-effector pose} → `C2R.npy`. Needs
a valid extrinsic first. [`Tsai_Lenz.py`](../../paradex/calibration/Tsai_Lenz.py) has two solvers:
`solve_axb_cpu(A, B)` is the closed-form Tsai-Lenz (rotation via `log`-map of SO(3) + cross products,
translation via least squares) returning a (4,4); `solve_ax_xb(A_list, B_list)` refines it with a torch
gradient-descent pass (6D rotation param + Gram-Schmidt, orthogonality/det regularizers), initialized
from `solve_axb_cpu`. Both take **lists of (4,4)** with `len(A) == len(B)`.

## Charuco detection (`paradex/image/aruco.py`)

`boardinfo_dict` is loaded from `system/current/charuco_info.json` (board `"3"` = 11×8 → 70 corners is
the default for intrinsic/extrinsic). `get_charuco_detector()` builds one `cv2.aruco.CharucoDetector` per
board (cached), honoring `setLegacyPattern`. `detect_charuco(img)` returns
`{b_id: {checkerCorner (n,2), checkerIDs (n,)}}`; `merge_charuco_detection` concatenates boards with
per-board id offsets. The detector runs with **default params** (no `DetectorParameters`/
`CharucoParameters` tuning) — a lever if detection is flaky.

> **`setLegacyPattern` is a per-board toggle — each board just has to match its own print.** OpenCV
> ≥4.6 flipped the default Charuco layout (top-left cell went from marker → empty checker);
> `setLegacyPattern(True)` restores the old layout. It is set **per board** in `charuco_info.json`, so
> different boards legitimately differ: boards `"1"`/`"2"` are `setLegacyPattern:true` and board `"3"`
> omits it (→ modern/False) — that is fine, not a bug, as long as each physical board was printed with
> the same setting the detector uses for it. The only failure mode is a **mismatch between a board's
> print and its `charuco_info.json` flag**, which makes `detectBoard` fail wholesale or misassign ids.
> `charuco_info.json` is the single source of truth; if you must change a board's pattern, edit it there
> and reprint. To confirm empirically, run `detect_charuco` on one captured image (the flag is already
> baked into the cached detector).

## Board generation (`src/calibration/generate_board.py`)

Print-ready ChArUco PDF at **exact physical scale**, and the anti-mismatch tool. It renders with
OpenCV's own `CharucoBoard.generateImage()` from the **same board object `detect_charuco` uses** (read
straight from `charuco_info.json` — `setLegacyPattern` is never overridden, so a generated board cannot
diverge from what detection expects), then feeds the raster back through `detect_charuco` and **refuses
to write the PDF unless every corner is recovered** (self-check). Page == board size; prints the matching
metric `charuco_info.json` entry to stdout but does not modify config. Defaults reproduce board `"3"` at
`--square-mm 50` (→ 550×400 mm, marker 35 mm at board 3's 0.7 ratio). **Supersedes the old**
`src/util/marker/generate_charuco.py`, which hand-placed markers with matplotlib
(`(row+col)%2==0` = legacy layout) and could silently disagree with the detector — the original burn.

## Format reference

- `intrinsic/<serial>/param/<ts>.json`: `{RMS_error, K (3×3 list), distortion (5,), width, height}`.
- `cam_param/<name>/intrinsics.json` per serial: `{original_intrinsics (9,), intrinsics_undistort (9,), dist_params, height, width}` — note the **flattened** 3×3.
- `cam_param/<name>/extrinsics.json` per serial: 12-length list → reshaped to (3,4).
- `C2R.npy`: (4,4) float.

`load_current_intrinsic` derives `intrinsics_undistort` at **load time** via
`getOptimalNewCameraMatrix(alpha=1)` — it is **not** stored in the per-camera `param` json (only `K` +
`distortion` are). `load_current_camparam`/`load_camparam` reshape the flattened matrices back to
(3,3)/(3,4); `get_cammtx` = `intrinsics_undistort @ extrinsic`.

## Gotchas for editors

- `capture.py`/`client.py` pairs are coupled by stream data types (`'image'` /8 JPEG,
  `'charuco_detection'` /8 float32) and ports (1234 publish, 6890 command). Change both together.
- `extrinsic/calculate.py` writes the refined model into a **new** timestamped extrinsic dir but the
  final `cam_param` uses the original `<name>` — trace `__main__` carefully before changing paths.
- `load_current_C2R` loads the **first sorted index** (`sorted(os.listdir(...))[0]`) under the latest
  `handeye_calibration/<name>/` — not "the latest index".
- `colmap.py` `load_colmap_camparam` has a **dead bug** worth knowing: it writes `dist_height` twice
  (h then w, so width is lost into that key) — harmless because the real `height`/`width` come from
  `orig_size`, but don't copy the pattern.
- Typo `dataset_acqusition` (missing 'i') is intentional repo-wide — don't "fix" imports.
