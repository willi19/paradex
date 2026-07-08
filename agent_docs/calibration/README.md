# agent_docs/calibration — agent orientation

Docs for **AI agents working with `paradex/calibration/`** — the layer that produces and loads the
**camera matrices, camera poses, and the camera→robot transform** that the rest of the rig depends on.
Calibration is a **fixed three-stage pipeline** — **intrinsic → extrinsic → hand-eye** — where each
stage consumes the previous stage's output. `paradex/calibration/` itself is only the **loaders +
solvers** ([`utils.py`](../../paradex/calibration/utils.py), [`colmap.py`](../../paradex/calibration/colmap.py),
[`Tsai_Lenz.py`](../../paradex/calibration/Tsai_Lenz.py)); the *capture + solve orchestration* lives in
[`src/calibration/`](../../src/calibration/). Read the **one** file for your task.

Mental model: **three parameter stores under `~/shared_data`, keyed by camera serial (string).**
Each stage writes one store; `load_current_*` picks the **latest** timestamped dir and hands you numpy.
You almost never call the loaders directly — [`ImageDict`](../../paradex/image/image_dict.py) pulls the
calibration in for you. Intrinsics come in a **distorted** and an **undistorted** flavor; picking the
wrong one is the single most common silent bug.

| Your task | Read |
|-----------|------|
| **Consume** params: load K / poses / `C2R`, project, undistort, feed `ImageDict` | [`usage.md`](usage.md) |
| **Edit** the capture or solve scripts (intrinsic/extrinsic/hand-eye), board gen, formats | [`internals.md`](internals.md) |
| Detect / project / undistort **images** with the loaded params | [`agent_docs/image`](../image/README.md) |

Rule of thumb: **reading calibration out** → `usage.md`; **producing or changing it** → `internals.md`.

## The three parameter stores
Everything lives under `~/shared_data`. `<ts>`/`<name>` are `YYYYMMDD_HHMMSS`; "current" loaders pick the
**latest** dir via `find_latest_directory`. All helpers are in [`utils.py`](../../paradex/calibration/utils.py).

| Store | Path | Written by (stage) | Read by |
|-------|------|--------------------|---------|
| Per-camera intrinsics (raw) | `intrinsic/<serial>/param/<ts>.json` | `src/calibration/intrinsic/calculate.py` (1) | `load_current_intrinsic()` |
| Combined cam params | `cam_param/<name>/{intrinsics,extrinsics}.json` | `src/calibration/extrinsic/calculate.py` (2) | `load_current_camparam()`, `load_camparam(demo_path)` |
| Camera→robot | `handeye_calibration/<name>/<idx>/C2R.npy` | `src/calibration/handeye/calculate.py` (3) | `load_current_C2R()` |

**Why two intrinsic stores:** `intrinsic/<serial>/param/*.json` is the raw per-camera calibration
(one camera, `cv2.calibrateCamera`); the extrinsic solve reads those as seed intrinsics, runs COLMAP,
and re-emits them **plus poses** into `cam_param/<name>/`. Downstream code almost always wants
`cam_param/` (via `load_current_camparam`), not the raw per-camera dir.

## File map
| File | What it is |
|------|-----------|
| [`utils.py`](../../paradex/calibration/utils.py) | The public API: `load_current_camparam` / `load_camparam` / `load_current_intrinsic` / `load_current_C2R` / `get_cammtx` + the store-path constants. |
| [`colmap.py`](../../paradex/calibration/colmap.py) | COLMAP glue for stage 2: `COLMAPDatabase` (SQLite schema), `get_two_view_geometries`, `load_colmap_camparam`. |
| [`Tsai_Lenz.py`](../../paradex/calibration/Tsai_Lenz.py) | Hand-eye `AX = XB` solvers: `solve_axb_cpu` (Tsai-Lenz closed form) + `solve_ax_xb` (torch refine). |
| `__init__.py` | Empty — import from the submodules directly. |

Consumers: [`ImageDict`](../../paradex/image/image_dict.py) is the primary one;
`src/calibration/{intrinsic,extrinsic,handeye}/` are the producers (each has its own `CLAUDE.md`).
Typo `dataset_acqusition` (missing 'i') is intentional repo-wide — don't "fix" it.
