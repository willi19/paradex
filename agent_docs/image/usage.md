# paradex/image — How to Use

The 2D multi-camera image layer. Everything is keyed by **camera serial string**. The hub is
`ImageDict`; the standalone functions (`detect_*`, `undistort_*`, `merge_image`) work on raw
arrays if you don't need the container.

> Editing these modules (detector caches, the renderer, undistort maths)? Read
> [`internals.md`](internals.md). Calibration param *shapes* (K, dist, `intrinsics_undistort`,
> distorted-vs-undistorted) live in `agent_docs/calibration/` — this doc assumes them.

---

## ImageDict — the container

```python
from paradex.image.image_dict import ImageDict

# from a saved capture dir: loads images/ (or raw/images/) and, if cam_param/ exists, calibration
imgs = ImageDict.from_path(path)
# or build directly from a serial→ndarray dict:
imgs = ImageDict({"25305460": img0, "25305462": img1})
imgs.load_current_camparam()                 # attach the latest global calibration
imgs.set_camparam(intrinsic, extrinsic)      # or inject explicitly
```

Dict-like: `imgs[serial]`, `for serial, img in imgs:`, `len(imgs)`, `serial in imgs`,
`imgs.serial_list`, `.keys()/.values()/.items()`.

### Batch ops (the reason to use it)
| call | returns | notes |
|------|---------|-------|
| `imgs.undistort(save_path=None)` | new `ImageDict` | per-serial `cv2.remap`; caches the map. Needs `intrinsic`. |
| `imgs.apply(func, use_camparam=False, **kw)` | `{serial: result}` | run `func(img, **kw)`; `use_camparam=True` also passes `intrinsic=`/`extrinsic=`. |
| `imgs.map_images(func, ...)` | new `ImageDict` | like `apply` but rewraps images. |
| `imgs.triangulate_markers(dict_type='6X6_1000')` | `(marker_2d, marker_3d)` | detect ArUco in every view → triangulate per id. |
| `imgs.triangulate_charuco()` | `{board_id: {checkerIDs, checkerCorner}}` | same for ChArUco corners. |
| `imgs.traingulate_points(pts2d)` | `(N,3)` | `pts2d` = `{serial: (N,2)}` (note the typo in the method name). |
| `imgs.project_pointcloud(pts3d)` | `{serial: (N,2)}` | project world points into every camera. |
| `imgs.project_mesh(obj, color=(0,255,0), alpha=0.5)` | new `ImageDict` | GPU-render a mesh + overlay its mask (needs CUDA + nvdiffrast). |
| `imgs.merge(image_text={})` | one `np.ndarray` | labeled grid preview. |
| `imgs.draw_keypoint({serial:(N,2)}, ...)` | new `ImageDict` | draw dots. |
| `imgs.save(path)` / `imgs.save_camparam(path)` | — | write `images/*.png` / `cam_param/*.json`. |

Triangulation/projection all go through `get_cammtx(intrinsic, extrinsic)` (3×4 per serial,
using `intrinsics_undistort`) — so **triangulate/project on undistorted geometry**. Detect on
raw frames only if you undistorted the *points* too. The proj matrices are cached on first use;
if you mutate `intrinsic`/`extrinsic` after that, make a fresh `ImageDict`.

---

## aruco — detection (works on bare arrays)

```python
from paradex.image.aruco import detect_aruco, detect_charuco, merge_charuco_detection

corners, ids = detect_aruco(img, dict_type='6X6_1000')   # ids is (N,1) int, [] / (0,1) if none
det = detect_charuco(img)          # {board_id: {"checkerCorner": (n,2), "checkerIDs": (n,)}}
flat = merge_charuco_detection(det)  # concat all boards with per-board id offsets → {checkerCorner, checkerIDs}
```

- Correspondence helpers: `find_common_indices(ids1, ids2) -> (idx1, idx2)` (match two views by
  id), `get_board_cor()` (board 3D object points).
- Drawing: `draw_aruco`, `draw_charuco`, `draw_keypoint` (all mutate/return the image).
- **Board config / which board / object-point scale / `setLegacyPattern`** are calibration
  concerns — see `agent_docs/calibration/README.md`, not this doc.

---

## undistort {#undistort}

```python
from paradex.image.undistort import undistort_img, undistort_points
und_img = undistort_img(img, intrinsic[serial])          # one-shot cv2.undistort
und_pts = undistort_points(pts, intrinsic[serial])       # (N,2) distorted → undistorted pixels
```

For many frames from one camera, precompute the map once:
```python
from paradex.image.undistort import precomute_undistort_map, apply_undistort_map
new_K, mapx, mapy = precomute_undistort_map(intrinsic[serial])
und = apply_undistort_map(img, mapx, mapy)               # cv2.remap, fast
```
GPU variant for video pipelines: `precompute_undistort_map_torch` + `apply_undistort_torch`
(`grid_sample`). After undistorting, use `intrinsics_undistort` with **no** distortion.

---

## merge / grid / overlay {#merge}

```python
from paradex.image.merge import merge_image        # labeled square grid (serials as captions)
grid = merge_image(image_dict, image_text={serial: "extra caption"})

from paradex.image.grid import make_image_grid      # bare tiling of a LIST of images (no labels)
from paradex.image.overlay import overlay_mask      # in-place alpha blend of a bool mask
```
`merge_image` assumes ~2048×1536 source frames (hardcoded canvas). Use it for live previews /
sync checks (that's what `src/validate/camera_system/` does), not for archival output.

---

## Gotchas
- Serial keys are **strings**; multi-cam images are `Dict[str, np.ndarray]` everywhere.
- Pick **one** of distorted / undistorted and stay consistent — `get_cammtx` (hence every
  triangulate/project op) uses `intrinsics_undistort`, so feed it undistorted 2D. Mixing is the
  classic silent bug (see calibration doc).
- `imgs.triangulate_points` is misspelled **`traingulate_points`** in the API.
- `ImageDict.__repr__` is broken (references a nonexistent attr) — don't `print(imgs)` / rely on
  repr; use `imgs.serial_list`. `from_path` on an empty dir also raises the wrong way. Both are
  known (see internals) — treat them as landmines, not examples to copy.
- `project_mesh` / `BatchRenderer` need CUDA + `nvdiffrast` installed, else import fails.
- Detectors run with **default** OpenCV params (no tuning) — first lever if detection is flaky.
