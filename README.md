# Paradex

Multi-camera system + Robot control framework

---

## Overview

Paradex provides a general-purpose library for multi-camera vision and robot control.

```
paradex/
├── paradex/    # Core library (general-purpose, reusable modules)
├── src/        # Applications (combines paradex modules for specific tasks)
├── config/     # Configuration files (camera, environment, data acquisition)
└── rsc/        # Resources (robot URDFs, object meshes, hand models)
```

---

## Quick Start

### Installation

```bash
pip install -e .
```

Basic Usage
1. Configure your system:
Edit configuration files in config/ directory:
bash# Camera settings
config/camera/camera.json          # Camera parameters
config/camera/camera_index.json    # Camera ID mapping
config/camera/lens_info.json       # Lens calibration

# Environment settings
config/environment/network.json    # Network configuration
config/environment/pc.json         # PC information
See config/README.md for detailed configuration guide.
2. Follow task-specific guides:

Camera Capture: src/capture/capture_readme.md
Calibration: src/calibration/calibration_readme.md
Image Processing: paradex/image/readme.md
Video Processing: paradex/video/readme.md

See detailed guides in:
- **Camera Capture:** [`src/capture/capture_readme.md`](src/capture/capture_readme.md)
- **Calibration:** [`src/calibration/calibration_readme.md`](src/calibration/calibration_readme.md)
- **Image Processing:** [`paradex/image/readme.md`](paradex/image/readme.md)
- **Video Processing:** [`paradex/video/readme.md`](paradex/video/readme.md)
