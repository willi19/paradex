# database.py

### 🗄️ Table Schema Overview

| Table | Columns (main) | Purpose |
|-------|----------------|---------|
| **`cameras`** | `camera_id, model, width, height, params BLOB` | Intrinsics per camera. |
| **`images`** | `image_id, name, camera_id, prior_q*, prior_t*` | Image metadata & optional priors. |
| **`keypoints`** | `image_id, rows, cols, data BLOB` | SIFT/ORB/etc. keypoints. |
| **`descriptors`** | `image_id, rows, cols, data BLOB` | 128‑D SIFT descriptors (uint8). |
| **`matches`** | `pair_id, rows, cols, data BLOB` | Raw feature matches. |
| **`two_view_geometries`** | `pair_id, rows, cols, data, config, F, E, H` | Inlier matches + epipolar geometry. |

The schema is created via the constant **`CREATE_ALL`**.

---

### 🔑 Helper Functions

| Function | Description |
|----------|-------------|
| `image_ids_to_pair_id(id1, id2)` | Deterministic mapping **(id1,id2) → pair_id** using `MAX_IMAGE_ID`. |
| `pair_id_to_image_ids(pair_id)` | Inverse mapping. |
| `array_to_blob(arr)` / `blob_to_array(blob, dtype, shape)` | Zero‑copy serialization between NumPy and SQLite BLOB. |

---

### 💾 Accessor Methods

| Method | Returns |
|--------|---------|
| `get_camera()` | `{camera_id: {model, width, height, params}}` |
| `get_images()` | `{image_id: {"name", "camera_id"}}` |
| `get_keypoints()` | `{image_id: np.ndarray (N,2)}` *(x,y)* |
| `get_corners()` | same as `get_keypoints()` but reshaped `(N,2)` |
| `get_descriptors()` | `{image_id: np.ndarray (N,128)}` |
| `get_matches()` | `{(id1,id2): np.ndarray (M,2)}` |
| `get_two_view_geometries()` | `{(id1,id2): {"data", "config", "F", "E", "H"}}` |

---

### ⚠️ Notes
* All coordinate & geometry matrices default to **identity** when not supplied.
* `array_to_blob` uses `.tostring()` which is deprecated in NumPy 1.24+;
  replace with `.tobytes()` if needed.
* The wrapper assumes SIFT dimension sizes (keypoints 6‑float, descriptors 128‑byte);
  modify the assertions for other feature types.
