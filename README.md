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

See detailed guides in:
- **Camera Capture:** [`src/capture/capture_readme.md`](src/capture/capture_readme.md)
- **Calibration:** [`src/calibration/calibration_readme.md`](src/calibration/calibration_readme.md)
- **Image Processing:** [`paradex/image/readme.md`](paradex/image/readme.md)
- **Video Processing:** [`paradex/video/readme.md`](paradex/video/readme.md)

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

## Configuration

- **Camera:** `config/camera/camera.json`
- **Environment:** `config/environment/`
- **Robot URDFs:** `rsc/robot/`
