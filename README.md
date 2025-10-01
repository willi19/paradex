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

### Basic Usage

**1. Configure your system**

Edit configuration files in `config/` directory:

```bash
# Camera settings
config/camera/camera.json          # Camera parameters
config/camera/camera_index.json    # Camera ID mapping
config/camera/lens_info.json       # Lens calibration

# Environment settings
config/environment/network.json    # Network configuration
config/environment/pc.json         # PC information
```

See [`config/README.md`](config/README.md) for detailed configuration guide.

**2. Follow task-specific guides**

- **Camera Capture:** [`src/capture/capture_readme.md`](src/capture/capture_readme.md)
- **Calibration:** [`src/calibration/calibration_readme.md`](src/calibration/calibration_readme.md)
- **Image Processing:** [`paradex/image/readme.md`](paradex/image/readme.md)
- **Video Processing:** [`paradex/video/readme.md`](paradex/video/readme.md)

**Additional Configuration**

See [`config/README.md`](config/README.md) for detailed configuration guide.

Quick reference:
- **Camera:** `config/camera/`
- **Environment:** `config/environment/`
- **Robot URDFs:** `rsc/robot/`

---

## Frequently Asked Questions

**Q: How do I capture images from cameras?**

See [`src/capture/capture_readme.md`](src/capture/capture_readme.md)

**Q: How do I control the robot?**

Check example scripts in `src/capture/robot/` or use:
```python
from paradex.robot import RobotWrapper
```

**Q: How do I register a new object?**

See examples in `src/register_object/` (e.g., `pringles.py`)

**Q: How do I perform calibration?**

See [`src/calibration/calibration_readme.md`](src/calibration/calibration_readme.md)

---

## Contributing

1. Create feature branch: `git checkout -b feature/your-feature`
2. **paradex/**: Keep functions general and reusable
3. **src/**: Combine paradex modules for specific applications

---

## License

[Add license info]