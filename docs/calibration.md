# Calibration

Overview of how Paradex turns a rig of cameras into metric, robot-referenced measurement devices,
and where every calibration parameter is stored and read back. Read this for the mental model; for
exact signatures see the {doc}`API reference <calibration_api>`.

- Core library: `paradex/calibration/` (`utils.py`, `colmap.py`, `Tsai_Lenz.py`)
- Application scripts: `src/calibration/` (`intrinsic/`, `extrinsic/`, `handeye/`, `generate_board.py`)
- Main consumer of the results: {doc}`ImageDict <image>`

:::{note}
Everything is keyed by **camera serial string**. The pipeline order is fixed:
**intrinsic → extrinsic → hand-eye**; each stage consumes the previous stage's output.
:::

---

## 1. The three parameter stores

Everything lives under `~/shared_data`. "Current" loaders pick the **latest** timestamped dir.

```{mermaid}
flowchart LR
    subgraph S1["intrinsic (per camera)"]
      I["intrinsic/&lt;serial&gt;/param/&lt;ts&gt;.json<br/>K, distortion"]
    end
    subgraph S2["extrinsic (rig solve)"]
      C["cam_param/&lt;name&gt;/<br/>intrinsics.json + extrinsics.json"]
    end
    subgraph S3["hand-eye"]
      R["handeye_calibration/&lt;name&gt;/&lt;idx&gt;/C2R.npy<br/>4×4 camera→robot"]
    end
    I -->|seeds| S2
    C -->|board poses + robot poses| S3
```

| Store | Path | Written by | Read by |
|-------|------|-----------|---------|
| Per-camera intrinsics | `intrinsic/<serial>/param/<ts>.json` | `src/calibration/intrinsic/calculate.py` | `load_current_intrinsic()` |
| Combined cam params | `cam_param/<name>/{intrinsics,extrinsics}.json` | `src/calibration/extrinsic/calculate.py` | `load_current_camparam()`, `load_camparam(demo_path)` |
| Camera→robot | `handeye_calibration/<name>/<idx>/C2R.npy` | `src/calibration/handeye/` | `load_current_C2R()` |

**Why two intrinsic stores.** `intrinsic/<serial>/param/*.json` is the raw per-camera calibration
(one camera, `cv2.calibrateCamera`). The extrinsic solve reads those as seed intrinsics, runs
COLMAP, and re-emits them **plus poses** into `cam_param/<name>/`. Downstream code almost always
wants `cam_param/` via `load_current_camparam` — not the raw per-camera dir.

---

## 2. Using the parameters

The one call you usually want:

```python
from paradex.calibration.utils import load_current_camparam, load_current_C2R, get_cammtx
intrinsic, extrinsic = load_current_camparam()   # or load_camparam(demo_path) for a saved dataset
C2R = load_current_C2R()                          # 4×4 camera→robot
P   = get_cammtx(intrinsic, extrinsic)            # {serial: 3×4 projection}
```

`intrinsic[serial]` → `{original_intrinsics (3×3), intrinsics_undistort (3×3), dist_params (5,),
height, width}`; `extrinsic[serial]` → **(3×4)** world→camera `[R|t]` (metric, meters).

:::{warning}
**Distorted vs. undistorted — pick one and stay consistent.** Raw frames are distorted → project
with `original_intrinsics` + `dist_params`. Undistort first (`cv2.undistort`) → then use
`intrinsics_undistort` with **no** distortion. `get_cammtx` uses `intrinsics_undistort`, so feed it
undistorted 2D. Mixing the two is the most common silent calibration bug.
:::

### `ImageDict` — the main consumer
Most vision code never calls the loaders directly; it uses {doc}`ImageDict <image>`, which wraps a
capture dir *and its `cam_param/`*: `ImageDict.from_path(path)` auto-loads calibration when
`cam_param/` is present, or `.load_current_camparam()` attaches the latest global one. Its
`undistort` / `triangulate_*` / `project_*` methods then run through `get_cammtx`.

---

## 3. Boards & `setLegacyPattern` (the thing that burns you)

Boards are defined in `system/current/charuco_info.json` — one entry per board id with `numX`,
`numY`, `checkerLength`, `markerLength`, `dict_type`, `markerIDs`, and optional `setLegacyPattern`.
`paradex.image.aruco.get_charuco_detector()` builds one cached `CharucoDetector` per board from
exactly these fields.

:::{important}
`setLegacyPattern` is a **per-board toggle**, not a global mode. OpenCV ≥4.6 flipped the default
ChArUco layout (top-left cell: marker → empty checker); `setLegacyPattern(True)` restores the old
layout. Different boards may legitimately differ — in the shipped config boards `"1"`/`"2"` are
`true` and board `"3"` omits it (modern/False), and that is **fine**. The only failure mode is a
**mismatch between a physical board's print and its `charuco_info.json` flag**: then `detectBoard`
fails wholesale or misassigns corner ids. `charuco_info.json` is the single source of truth.
:::

### Generating a board: `src/calibration/generate_board.py`
Renders a print-ready ChArUco PDF at **exact physical scale** and is the anti-mismatch tool:

```{mermaid}
flowchart LR
    CFG["charuco_info.json<br/>(source of truth)"] --> BLD["CharucoBoard object<br/>(same one detect_charuco uses)"]
    BLD --> GEN["generateImage()<br/>OpenCV's own renderer"]
    GEN --> CHK{"detect_charuco<br/>recovers ALL corners?"}
    CHK -->|no| STOP["refuse to write PDF"]
    CHK -->|yes| PDF["exact-size PDF<br/>+ metric config entry printed"]
```

It never hand-draws markers and never overrides `setLegacyPattern` (reads it from config), then
round-trips the rendered raster through `detect_charuco` and **refuses to write unless every corner
is recovered**. So a PDF it produces is detectable by this repo's detector by construction.
Defaults reproduce board `"3"` at `--square-mm 50` (→ 550×400 mm). It supersedes the old
`src/util/marker/generate_charuco.py`, which hand-placed markers and could silently disagree with
the detector.

---

## 4. The pipeline stages

```{mermaid}
flowchart TB
    B["generate_board.py<br/>print ChArUco"] --> INTR
    INTR["intrinsic/<br/>distributed auto-capture → cv2.calibrateCamera"] --> EXTR
    EXTR["extrinsic/<br/>COLMAP incremental_mapping + refine + metric rescale"] --> HE
    HE["handeye/<br/>Tsai-Lenz AX=XB"] --> OUT["cam_param + C2R"]
```

**Intrinsic** (`src/calibration/intrinsic/`) — distributed, **auto-capture, no save button**: each
Capture PC keeps a frame only when the full board is visible and its mean per-corner displacement is
novel vs. frames already kept for that camera; `q` triggers save; `calculate.py` runs
`cv2.calibrateCamera` per serial.

**Extrinsic** (`src/calibration/extrinsic/`) — distributed ChArUco capture → `calculate.py` builds a
COLMAP database, `pycolmap.incremental_mapping`, a reproj-error refine pass, then rescales
translations so the mean board edge = 0.06 m (metric).

:::{warning}
Extrinsic BA runs with `ba_refine_focal_length/principal_point/extra_params=True` **on purpose**:
holding the seed intrinsics fixed makes COLMAP fail to converge. The tradeoff is that per-camera
intrinsics can drift, so intrinsic quality (coverage) still matters. Flipping these to False to
"trust stage 1" reintroduces the non-convergence.
:::

**Hand-eye** (`src/calibration/handeye/`) — Tsai-Lenz `AX = XB` (`Tsai_Lenz.solve_ax_xb`) from paired
{camera-observed board pose, robot end-effector pose} → `C2R.npy`. Needs a valid extrinsic first.

---

## 5. Conventions

- Extrinsic translations and `C2R` are **metric (meters)**.
- Camera frames follow the OpenCV convention (+Z forward, +X right, +Y down).
- Board object-point scale is irrelevant to intrinsic K/distortion (only corner geometry matters).
- The typo `dataset_acqusition` (elsewhere in the repo) is intentional.
