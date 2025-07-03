# file_io.py
### File‑IO Helper Utilities

A collection of convenience functions for loading camera parameters, trajectories, meshes, and miscellaneous files used in the **Paradex** project.

---

#### 📂 Path Constants

| Variable | Description |
|----------|-------------|
| `rsc_path` | Root to static resources (`rsc/`). |
| `home_path` | User home directory (`~`). |
| `shared_dir` | `~/shared_data` (network‑shared data). |
| `calib_path_list` | `[~/captures1/calibration, ~/captures2/calibration]` |
| `capture_path_list` | `[~/captures1, ~/captures2]` |
| `download_dir` | `~/download` |
| `cam_param_dir` | `~/shared_data/cam_param` |
| `handeye_calib_path` | `~/shared_data/handeye_calibration` |
| `config_dir` | `.../config` (relative to package root) |

---

## 🔍 Video Helpers


| Function | Purpose |
|----------|---------|
| `get_video_list(video_dir)` | Scan `video_dir` for `.avi` or `.mp4` files and return a list of tuples **`(video_path, timestamp_json_path)`**. If the corresponding timestamp JSON is absent, `None` is placed in the second position. |
| `find_latest_directory(directory)` | Return the lexicographically newest sub‑directory name under `directory`; if none exist, returns `"0"`. |
| `find_latest_index(directory)` | Same as above but treats sub‑directory names as integers and returns the highest value. If none exist, returns `-1`. |
---

## 🎥 Camera Parameter Loaders

| Function | Purpose |
|----------|---------|
| `load_cam_param_prev(name=None)` | **Deprecated**: legacy field names (`Intrinsics`, `Intrinsics_warped`, `dist_param`). |
| `load_cam_param(name=None)` | Current JSON layout (`original_intrinsics`, `intrinsics_undistort`, `dist_params`). |
| `load_cam_param_temp(name=None)` | Temporary extrinsics filename `extrinsics_temp.json`. |
| `load_camparam(demo_path)` | Load per‑demo intrinsics / extrinsics in `<demo>/cam_param/`. |
| `load_intrinsic()` | Scan `~/shared_data/intrinsic/<cam>/param` and build undistorted intrinsics for each camera. |
| `load_colmap_camparam(path)` | Parse COLMAP reconstruction and convert to intrinsics/extrinsics dicts. |

All loaders return two dictionaries: `intrinsics`, `extrinsics`.  
*Intrinsics* dict maps **serial → {intrinsic matrices, dist_params, H, W}**.  
*Extrinsics* dict maps **serial → (3×4 or 4×4) matrix**.

---

## 🤖 Trajectory & Contact Loaders

| Function | File(s) Loaded | Shape / Type |
|----------|---------------|--------------|
| `load_obj_traj(demo_path)` | `obj_traj.pickle` | arbitrary |
| `load_robot_traj(demo_path)` | `arm/state.npy` + `hand/state.npy` | `(T, D_arm + D_hand)` |
| `load_robot_traj_prev(demo_path)` | `robot_qpos.npy` | legacy |
| `load_robot_target_traj(demo_path)` | `arm/action.npy` + `hand/action.npy` | `(T, D_arm + D_hand)` |
| `load_contact_value(demo_path)` | `contact/data.npy` | `(T, C)` |
| `load_c2r(demo_path)` | `C2R.npy` | `(4, 4)` |

---

## 🖼️ Image & Mesh Helpers

* **`is_image_file(file)`** – Returns `True` for `.png/.jpg/.jpeg`.  
* **`load_images(dir)`** – List comprehension wrapper that returns absolute image paths inside `dir`.  
* **`load_mesh(obj_name)`** – Loads an `.obj` mesh from `rsc/<obj_name>/<obj_name>.obj` using *Open3D*.

