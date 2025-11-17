# Configuration

This directory contains all configuration files for the Paradex system.

## Structure

```
config/
├── $SYSTEM_NAME/              # Name of the system(paradex1, paradex2)
│   ├── camera_index.json    # Camera ID mapping
│   └── camera.json          # Camera parameters and settings
│   
│
├── current/         # Environment setup
│   ├── charuco_info.json    # Charuco board parameters
│   ├── marker.json          # ArUco marker configurations
│   ├── network.json         # Network settings for distributed capture
│   └── pc.json              # PC/workstation information
│
└── colmap_options.yaml  # COLMAP reconstruction options
```

---

## Camera Configuration

### camera_index.json
Maps camera IDs to their physical devices or network addresses.

### camera.json
Main camera configuration including:
- Resolution and frame rate
- Synchronization settings
- Capture parameters

### lens_info.json
Lens distortion parameters for camera calibration:
- Intrinsic matrix
- Distortion coefficients
- Image dimensions

---

## Environment Configuration

### charuco_info.json
Charuco calibration board specifications:
- Board dimensions
- Square size
- Marker size
- Dictionary type

### marker.json
ArUco marker configurations for object tracking and environment setup.

### network.json
Network settings for multi-PC camera synchronization:
- IP addresses
- Port configurations
- Sync protocols

### pc.json
Workstation information and hardware specifications.

---

## COLMAP Options

`colmap_options.yaml` contains parameters for 3D reconstruction:
- Feature extraction settings
- Matching parameters
- Bundle adjustment options

---

## Usage

Configuration files are loaded by modules in `paradex/` and used by scripts in `src/`.

Example:
```python
from paradex.utils import load_config

camera_config = load_config('config/camera/camera.json')
```