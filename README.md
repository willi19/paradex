# Paradex

A distributed **multi-camera vision + robot control framework** for dexterous manipulation research.

🌐 **Project page:** open [`index.html`](index.html) (deployable via GitHub Pages) &nbsp;·&nbsp; 📚 **API reference:** [`docs/index.html`](docs/index.html)

---

## Overview

```
paradex/
├── paradex/   # Core library (general-purpose, reusable modules)
├── src/       # Applications (combine paradex modules for specific tasks)
├── system/    # System configs (camera, network, PC info) — active config in system/current/
├── rsc/       # Resources (robot URDFs, object meshes, hand models)
└── docs/      # Generated Sphinx API reference
```

- **`paradex/`** — calibration, camera IO, robot control, transforms, simulation, retargeting, visualization. Install with `pip install -e .`
- **`src/`** — concrete workflows: calibrate → capture → process → infer. **Every app group has a `README.md` (humans) + `CLAUDE.md` (Claude).** See the [application index](src/README.md).

> **Distributed system:** 6 capture PCs + 1 main PC. Many scripts pair a **Capture-PC** daemon/client (waits for commands) with a **Main-PC** orchestrator (`_remote` / `_main`, sends commands over SSH/TCP). Hardware sync via a UTGE900 signal generator.

---

## Installation

```bash
git clone https://github.com/willi19/paradex.git
cd paradex
pip install -e .
```

System configuration lives under [`system/`](system/) — the active profile is the `system/current/` symlink (e.g. `paradex1`, `paradex2`).

---

## Quick Start

### 1. Calibration — run in order: `intrinsic → extrinsic → handeye`

| Step | Command | Notes |
|------|---------|-------|
| Extrinsic | `python src/calibration/extrinsic/capture.py` → `calculate.py` | Charuco + COLMAP. Press `c` to capture, `q` to quit. |
| Hand-eye | `python src/calibration/handeye/capture.py --arm xarm` → `calculate.py --arm xarm` | Camera→robot (`C2R.npy`). Needs extrinsic first. |

Re-run rules: aperture/focal length changed → redo from intrinsic; extrinsic changed → redo hand-eye.
Details: [`src/calibration/`](src/calibration/README.md).

### 2. Capture (distributed)

```bash
# on each capture PC
python src/camera/server_daemon.py

# on the main PC
python src/capture/camera/image_remote.py --save_path dataset/001
python src/capture/camera/video_remote.py --save_path dataset/001 --sync_mode
```

Details: [`src/capture/`](src/capture/README.md) · [`src/camera/`](src/camera/README.md).

### 3. Dataset acquisition / Process / Inference

- Build datasets: [`src/dataset_acquisition/`](src/dataset_acquisition/README.md)
- Post-process (sync, undistort, COLMAP): [`src/process/`](src/process/README.md)
- 6D pose & grasp: [`src/object6d/`](src/object6d/README.md) · [`src/inference/`](src/inference/README.md)

### Validation

Sanity-check any subsystem in isolation: [`src/validate/`](src/validate/README.md) (cameras, sync, robots, teleop, calibration quality).

---

## Applications

| Group | Purpose |
|-------|---------|
| [calibration/](src/calibration/README.md) | Intrinsic/extrinsic + hand-eye calibration |
| [camera/](src/camera/README.md) | Capture-PC camera daemons |
| [capture/](src/capture/README.md) | Image/video/stream + robot teleop capture |
| [dataset_acquisition/](src/dataset_acquisition/README.md) | Per-dataset capture pipelines |
| [process/](src/process/README.md) | Post-capture processing (sync, COLMAP) |
| [object6d/](src/object6d/README.md) | Multi-view 6D object pose |
| [inference/](src/inference/README.md) | Grasp inference & evaluation |
| [util/](src/util/README.md) | Charuco gen, object registration, URDF tools, upload |
| [validate/](src/validate/README.md) | System validation harnesses |

Full index: [`src/README.md`](src/README.md).

---

## Documentation

- **Project landing page:** [`index.html`](index.html) — host with GitHub Pages (deploy from branch → root). `.nojekyll` is included so the Sphinx `_static`/`_modules` dirs serve correctly.
- **API reference (Sphinx):** [`docs/index.html`](docs/index.html)
- **Project guide & task map for Claude:** [`CLAUDE.md`](CLAUDE.md)

---

## Contributing

1. Create a feature branch: `git checkout -b feature/your-feature`
2. **`paradex/`** — keep functions general and reusable.
3. **`src/`** — combine paradex modules for specific applications; add/update the app's `README.md` + `CLAUDE.md`.
