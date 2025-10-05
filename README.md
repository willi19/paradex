# Paradex
Multi-camera system + Robot control framework

---

## Overview
Paradex provides a general-purpose library for multi-camera vision and robot control.

```
paradex/
├── paradex/        # Core library (general-purpose, reusable modules)
├── src/            # Applications (combines paradex modules for specific tasks)
├── config/         # Configuration files (camera, environment, data acquisition)
└── rsc/            # Resources (robot URDFs, object meshes, hand models)
```

---

## Quick Start

### Validation
[To be added]

### Calibration
Required sequence: `intrinsic → extrinsic → handeye_calibration`  
Optional: `eef` (if using end effector)

**Important:**
- Camera aperture/focal length changed → Re-run from intrinsic
- Before data capture → Must run extrinsic
- Extrinsic changed → Re-run handeye_calibration

**Calibration types:**
- **intrinsic**: Camera internal parameters
- **extrinsic**: Camera-to-camera relative positions
- **eef**: Robot last link → end effector transformation
- **handeye_calibration**: Camera coordinate → robot position
- **all_in_one**: (In development) intrinsic + handeye combined

**Detailed guide:** [`src/calibration/calibration_readme.md`](src/calibration/calibration_readme.md)

### Capture

#### Camera
Captures image and video data from cameras in a distributed system.

**Image Capture:**
1. On Capture PC: `python src/capture/camera/image.py` (wait for commands)
2. On Main PC: `python src/capture/camera/image_main.py` (send commands)

**Video Capture:**
1. On Capture PC: `python src/capture/camera/video.py` (wait for commands)
2. On Main PC: `python src/capture/camera/video_main.py` (send commands)

**Detailed guide:** [`src/capture/capture_readme.md`](src/capture/capture_readme.md)

#### Robot
[To be added]

#### Tactile
[To be added]

### Utilities

Essential utilities for system operation:

- **`util/git_pull.py`**: Synchronize code across Capture PCs
- **`util/kill_process.py`**: Terminate dangling client processes after Main PC emergency stop
- **`util/video_process/process_video_main.py`**: Post-process captured videos (undistort + frame drop correction)

---

## Advanced Usage

### Examples
- **Dataset Acquisition:** Custom dataset creation (see `src/dataset_acquisition/`)
- **Inference:** Model execution examples (see `src/inference/`)
- **Process:** Data preprocessing pipelines (see `src/process/`)

[Detailed documentation to be added]

---

## System Setup

### Installation
```bash
pip install -e .
```

### Configuration
Edit configuration files in `config/` directory:

```bash
# Camera settings
config/camera/camera.json           # Camera parameters
config/camera/camera_index.json     # Camera ID mapping
config/camera/lens_info.json        # Lens calibration

# Environment settings
config/environment/network.json     # Network configuration
config/environment/pc.json          # PC information
```

See [`config/README.md`](config/README.md) for detailed configuration guide.

**Quick reference:**
- **Camera:** `config/camera/`
- **Environment:** `config/environment/`
- **Robot URDFs:** `rsc/robot/`

---

## Contributing
1. Create feature branch: `git checkout -b feature/your-feature`
2. **paradex/**: Keep functions general and reusable
3. **src/**: Combine paradex modules for specific applications

---

## License
[Add license info]