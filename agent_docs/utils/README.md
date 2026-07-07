# agent_docs/utils — agent orientation

Docs for **AI agents working on `paradex/utils/`** — the repo's utility grab-bag and its
**most-imported** module. Config loading, hard-coded filesystem paths, small file/pickle/yaml
loaders, an rsync wrapper, a stdin keyboard listener, and the shared logger all live here.
Nothing in this module is "an algorithm"; it's the plumbing every other module and `src/` script
pulls in on line 1. Read this one file — the whole module is seven short files. Focus is **which
helper do I call for X**, not internals.

Mental model: **`path.py` = where things are on disk, `system.py` = who/what this PC is
(from `system/current/`), the rest = tiny loaders + I/O helpers.** Most of the repo's "import
paradex" starts here, so a rename ripples widely — treat these as a public API.

## File map
| File | What it is |
|------|-----------|
| `path.py` | Module-level path constants: `home_path`, `pc_name`, `shared_dir`, `capture_path_list`, `download_dir`, `model_dir`, `rsc_path`. No functions — just `os.path.join`ed strings evaluated at import. |
| `system.py` | Reads `system/current/{pc,network,camera}.json` at import into `config_dir`, `pc_info`, `network_info`, `pc_name`. Helpers: `get_pc_ip`, `get_camera_list`, `get_pc_list`, `get_camera_config`. |
| `file_io.py` | Grab-bag loaders: `load_yaml`, `find_latest_directory`, `find_latest_index`, `load_images`/`is_image_file`, `remove_home`, and demo-trajectory loaders (`load_robot_traj`, `load_obj_traj`, `load_robot_target_traj`, `load_contact_value`, `load_mesh`). |
| `log.py` | `get_logger(name)` — console + NAS-file logger, PC/date-namespaced under `shared_dir/log/`. Degrades to console-only if the NAS is down. |
| `upload_file.py` | `rsync_copy(src, dst, ...)` — safe rsync wrapper (archive, checksum, `--partial`, optional move). Plus `check_rsync_installed`, `get_file_size`, `format_size`, `estimate_time`. |
| `keyboard_listener.py` | `listen_keyboard(event_dict)` / `stop_listening()` — background thread that maps typed stdin keys to `threading.Event.set()`. Used by interactive capture scripts. |
| `__init__.py` | Empty. `paradex.utils` is a namespace only; always import the submodule (`paradex.utils.path`, not `paradex.utils`). |

All paths relative to [`paradex/utils/`](../../paradex/utils/).

---

## Who calls this / most-used helpers
Counts from `grep` over `src/` + `paradex/` import lines (consolidated by symbol).

| Helper | Import | ~uses | What it gives you |
|--------|--------|------:|-------------------|
| `shared_dir` | `from paradex.utils.path import shared_dir` | 45+ | `~/shared_data` — the NAS root where every dataset / calibration / log lives. **The single most-imported name in the repo.** |
| `network_info` | `from paradex.utils.system import network_info` | 19 | Parsed `system/current/network.json` (ports/IPs for the distributed PCs). |
| `listen_keyboard` | `from paradex.utils.keyboard_listener import listen_keyboard` | 17 | Non-blocking stdin key → event mapping for interactive scripts. |
| `home_path` | `from paradex.utils.path import home_path` | 15+ | `os.path.expanduser("~")`. |
| `rsc_path` | `from paradex.utils.path import rsc_path` | 10+ | Repo `rsc/` dir (URDFs, meshes). |
| `get_logger` | `from paradex.utils.log import get_logger` | 10 | Shared console+NAS logger. |
| `get_pc_list` / `get_pc_ip` | `from paradex.utils.system import ...` | 8 | Enumerate the capture PCs / resolve a PC's IP. |
| `rsync_copy` | `from paradex.utils.upload_file import rsync_copy` | 7 | Copy/move files to the NAS. |
| `config_dir` | `from paradex.utils.system import config_dir` | 4 | Path to `system/current/`. |
| `find_latest_directory` / `find_latest_index` | `from paradex.utils.file_io import ...` | 7 | Newest timestamped / numeric-named subdir. |

Callers span nearly all of `src/` and much of `paradex/` (camera_system, calibration, robot,
dataset_acqusition, io/capture_pc). This is foundational plumbing — assume any change is a
wide-blast-radius change.

---

## The 3-4 you'll actually touch

### `path.py` — filesystem constants (no functions)
Everything is a module-level string built from `os.path.expanduser("~")` and evaluated **at
import time**:
- `home_path` = `~`; `pc_name` = basename of `~` (the PC identity, e.g. `paradex1`).
- `shared_dir` = `~/shared_data` — the NAS mount; the root of `cam_param/`, datasets, `log/`, etc.
- `capture_path_list` = `[~/captures1, ~/captures2]` — local raw-capture roots (hard-coded to 2).
- `download_dir` = `~/download`; `model_dir` = repo `../../model`; `rsc_path` = repo `rsc/`.

Because these are import-time constants, they reflect the machine and `$HOME` of the running
process — there is no override argument. `rsc_path`/`model_dir` are relative to this file's
location, so they resolve to the repo checkout regardless of cwd.

### `system.py` — this PC's config (from `system/current/`)
At import it `json.load`s `pc.json` and `network.json` from `config_dir`
(`paradex/../../system/current`, the active-config symlink). **Import fails if those files are
missing** (`camera.json` is optional — `get_camera_config` returns `{}` when absent).
- `pc_info` / `network_info` — the parsed JSONs.
- `config_dir` — path to `system/current/`.
- `get_pc_ip(pc_name)`, `get_camera_list(pc_name=<this pc>)`, `get_pc_list()`, `get_camera_config()`.

Note `pc_name` is defined **both** here and in `path.py` (same value, computed independently).

### `file_io.py` — loaders grab-bag
- `load_yaml(path)` → dict (the canonical yaml loader used across the repo; per CLAUDE.md).
- `find_latest_directory(dir)` → newest child by string sort (`None` if empty).
- `find_latest_index(dir)` → newest child by **int** sort (`-1` if missing/empty) — for numeric
  session dirs; will raise on non-numeric names.
- `is_image_file` / `load_images(dir)` → list of image paths in a dir.
- `remove_home(path)` → path made relative to `~` (used to derive NAS `root_name` keys).
- Demo loaders (`load_robot_traj`, `load_robot_target_traj`, `load_obj_traj`,
  `load_contact_value`, `load_robot_traj_prev`) read the fixed `arm/`, `hand/`, `contact/`,
  `obj_traj.pickle` layout of a demo dir.

### `log.py` — `get_logger(name)`
Console + NAS-file logger. File goes to `~/shared_data/log/<pc_name>/<YYYYMMDD>/<name>_<HHMMSS>.log`
(PC/date-namespaced so the capture PCs don't clobber each other). Idempotent per process: a second
`get_logger("x")` returns the same logger/file. `name` is both logger name and file stem — keep it
short and filesystem-safe. If the NAS isn't mounted, the file handler is skipped and it logs to
console only rather than crashing.

---

## Gotchas for editors
- **Import-time side effects.** `system.py` reads three JSONs and `path.py`/`system.py` compute
  paths *at import*. Importing `paradex.utils.system` on a box without `system/current/pc.json`
  and `network.json` **raises immediately** — before any of your code runs. Keep this in mind
  when writing tests or running off a capture PC.
- **`capture_path_list` is hard-coded to two dirs** (`captures1`, `captures2`, `range(1,3)`). A PC
  with a different number of capture disks is not configurable here.
- **`file_io.load_mesh(obj_name)` is latently broken**: it references `rsc_path` but `file_io.py`
  never imports it → `NameError` if called. It's dead/unused; don't assume it works.
- **Two consumer import paths don't exist in this module** — `from paradex.utils.env import
  get_serial_list` and `from paradex.utils.file_io import get_robot_urdf_path`. There is no
  `paradex/utils/env.py`, and `file_io.py` defines neither `get_robot_urdf_path`, `rsc_path`,
  `config_dir`, nor `home_path`. Those symbols actually live in
  [`paradex/io/camera_system/pyspin.py`](../../paradex/io/camera_system/pyspin.py) (`get_serial_list`)
  and [`paradex/robot/utils.py`](../../paradex/robot/utils.py) (`get_robot_urdf_path`). Those
  imports are stale/broken — don't propagate them, and don't "add" the symbols here to satisfy them.
- **`pc_name` lives in two files** (`path.py` and `system.py`). If you change how PC identity is
  derived, change both.
- **These names are a de-facto public API.** `shared_dir`, `home_path`, `network_info`,
  `listen_keyboard`, `get_logger`, `rsync_copy` are imported by name in 100+ places across
  `src/`. Renaming or changing a signature is a repo-wide edit, not a local one.
- **`rsync_copy` shells out to `rsync`** and, with `move=True`, passes `--remove-source-files`
  (delete-on-success). It returns `False` (does not raise) when rsync is missing or the source
  doesn't exist — check the return value.
- **`keyboard_listener` uses a module-level `stop_event`** shared across all listeners in the
  process; `stop_listening()` stops every listener, not just one.
