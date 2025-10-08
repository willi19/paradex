# Image Processing Module

Image manipulation and computer vision operations for multi-camera systems.

---

## Overview

This module provides essential image processing utilities for working with multi-camera setups, including marker detection, camera calibration support, projection operations, and image manipulation.

---

## Modules

### `aruco.py`
ArUco and CharUco marker detection and tracking.

**Key Functions:**
- `detect_aruco(img, dict_type)` - Detect ArUco markers in an image
- `detect_charuco(img, boardinfo)` - Detect CharUco board corners and markers
- `triangulate_marker(img_dict, intrinsic, extrinsic, dict_type)` - Triangulate 3D marker positions from multiple camera views
- `draw_aruco(img, kypt, ids, color)` - Visualize detected ArUco markers
- `draw_charuco(image, corners, color, radius, thickness, ids)` - Visualize detected CharUco corners

**Supported ArUco Dictionaries:**
- 4X4: 50, 100, 250, 1000 markers
- 5X5: 50, 100, 250, 1000 markers
- 6X6: 50, 100, 250, 1000 markers
- 7X7: 50, 100, 250, 1000 markers

**Usage Example:**
```python
from paradex.image.aruco import detect_aruco, triangulate_marker

# Detect markers in a single image
corners, ids = detect_aruco(image, dict_type='6X6_1000')

# Triangulate 3D positions from multiple cameras
image_dict = {serial_num: image for serial_num, image in camera_images.items()}
marker_3d = triangulate_marker(image_dict, intrinsics, extrinsics, dict_type='6X6_1000')
```

---

### `merge.py`
Image grid composition for multi-camera visualization.

**Key Functions:**
- `merge_image(image_dict)` - Merge multiple camera images into a single grid view

**Input Format:**
- `image_dict`: `Dict[str, np.ndarray]` - Dictionary mapping camera serial numbers to images

**Usage Example:**
```python
from paradex.image.merge import merge_image

image_dict = {
    "camera_01": img1,
    "camera_02": img2,
    "camera_03": img3,
}
grid_image = merge_image(image_dict)
```

---

### `undistort.py`
Lens distortion correction.

**Key Functions:**
- `undistort_img(img, intrinsic)` - Remove lens distortion from an entire image
- `undistort_points(pts, intrinsic)` - Undistort specific pixel coordinates
- `remap_corners(corners, cammtx, dist_coef, sn, img)` - Remap corner pixels with optimal camera matrix

**Usage Example:**
```python
from paradex.image.undistort import undistort_img, undistort_points

# Undistort entire image
undistorted_image = undistort_img(image, intrinsic)

# Undistort specific points
undistorted_pts = undistort_points(corner_points, intrinsic)
```

---

### `projection.py`
Camera projection and 3D rendering operations.

**Key Functions:**
- `get_cammtx(intrinsic, extrinsic)` - Compute projection matrices for all cameras
- `project_point(verts, cammtx, image, color)` - Project 3D points onto 2D image
- `project_mesh(image, mesh, intrinsic, extrinsic, obj_T, renderer)` - Render 3D mesh onto image using PyRender
- `project_mesh_nvdiff(object, renderer)` - GPU-accelerated mesh rendering using NVDiffRast

**Usage Example:**
```python
from paradex.image.projection import get_cammtx, project_point

# Get projection matrices
cam_matrices = get_cammtx(intrinsics, extrinsics)

# Project 3D points
image_with_points = project_point(vertices_3d, cam_matrices[serial_num], image)
```

---

### `overlay.py`
Image overlay and masking operations.

**Key Functions:**
- `overlay_mask(image, mask, color, alpha)` - Overlay a colored mask on an image with transparency

**Usage Example:**
```python
from paradex.image.overlay import overlay_mask

# Apply semi-transparent red mask
masked_image = overlay_mask(image, mask, color=(255, 0, 0), alpha=0.7)
```

---

## Data Structures

### Image Dictionary Format
Many functions use a standardized dictionary format for multi-camera images:

```python
image_dict = {
    "serial_num_1": np.ndarray,  # Camera 1 image
    "serial_num_2": np.ndarray,  # Camera 2 image
    ...
}
```

### Intrinsic Parameters Format
```python
intrinsic = {
    "serial_num": {
        "original_intrinsics": np.ndarray,      # 3x3 camera matrix
        "dist_params": np.ndarray,              # Distortion coefficients
        "intrinsics_undistort": np.ndarray,     # Undistorted camera matrix
        "width": int,
        "height": int
    }
}
```

### Extrinsic Parameters Format
```python
extrinsic = {
    "serial_num": np.ndarray  # 3x4 or 4x4 transformation matrix [R|t]
}
```

---

## Dependencies

- OpenCV (cv2)
- NumPy
- PyRender (for mesh projection)
- Trimesh
- Open3D
- PyTorch (for some operations)

---

## Notes

- All images are expected in BGR format (OpenCV default)
- Coordinate system follows OpenCV conventions
- Distortion models use OpenCV's radial-tangential model (k1, k2, p1, p2)
- Projection matrices are computed as: `P = K @ [R|t]` where K is intrinsic, [R|t] is extrinsic