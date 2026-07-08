# agent_docs/utils — agent orientation

Docs for **AI agents working on `paradex/utils/`** — the repo's utility grab-bag and its
**most-imported** module. Config loading, hard-coded filesystem paths, small file/pickle/yaml
loaders, an rsync wrapper, a stdin keyboard listener, and the shared logger all live here.
Nothing in this module is "an algorithm"; it's the plumbing every other module and `src/` script
pulls in on line 1. These are seven short, independent files — organized **by helper, not by a
flow**. Focus is **which helper do I call for X**, not internals.

Mental model: **`path.py` = where things are on disk, `system.py` = who/what this PC is
(from `system/current/`), the rest = tiny loaders + I/O helpers.** Most of the repo's
`import paradex` starts here, so a rename ripples widely — treat these names as a **public API**.

| Your task | Read |
|-----------|------|
| **Call** a helper: config load, path constant, `load_yaml`/`load_pickle`, `get_logger`, keyboard listener, rsync | [`usage.md`](usage.md) |
| **Edit / understand** the tricky ones: config+symlink resolution, logger NAS behavior, `find_latest_*` semantics, listener threading, hard-coded-path traps | [`internals.md`](internals.md) |
| Just need the disk root `~/shared_data` | [`usage.md`](usage.md) — `from paradex.utils.path import shared_dir` |

Rule of thumb: **calling** a helper → `usage.md`; **changing** one (or hunting a path/import trap) → `internals.md`.

## File map
| File | What it is |
|------|-----------|
| [`path.py`](../../paradex/utils/path.py) | Module-level path **constants** (no functions): `home_path`, `pc_name`, `shared_dir`, `capture_path_list`, `download_dir`, `model_dir`, `rsc_path`. `os.path.join`ed strings evaluated **at import**. |
| [`system.py`](../../paradex/utils/system.py) | Reads `system/current/{pc,network}.json` **at import** into `config_dir`, `pc_info`, `network_info`, `pc_name`. Helpers: `get_pc_ip`, `get_camera_list`, `get_pc_list`, `get_camera_config` (`camera.json` is lazy/optional). |
| [`file_io.py`](../../paradex/utils/file_io.py) | Loader grab-bag: `load_yaml`, `find_latest_directory`, `find_latest_index`, `load_images`/`is_image_file`, `remove_home`, and demo-trajectory loaders (`load_robot_traj`, `load_obj_traj`, `load_robot_target_traj`, `load_contact_value`, `load_mesh`). |
| [`log.py`](../../paradex/utils/log.py) | `get_logger(name)` — console + NAS-file logger, PC/date-namespaced under `shared_dir/log/`. Degrades to console-only if the NAS is down. |
| [`keyboard_listener.py`](../../paradex/utils/keyboard_listener.py) | `listen_keyboard(event_dict)` / `stop_listening()` — background thread mapping typed stdin keys to `threading.Event.set()`. Used by interactive capture scripts. |
| [`upload_file.py`](../../paradex/utils/upload_file.py) | `rsync_copy(src, dst, ...)` — safe rsync wrapper (archive, checksum, `--partial`, optional move). Plus `check_rsync_installed`, `get_file_size`, `format_size`, `estimate_time`. |
| [`__init__.py`](../../paradex/utils/__init__.py) | Empty. `paradex.utils` is a namespace only; always import the submodule (`paradex.utils.path`, not `paradex.utils`). |

All paths relative to [`paradex/utils/`](../../paradex/utils/).

## Most-used names (import counts across `src/` + `paradex/`)
| Helper | Import | ~uses |
|--------|--------|------:|
| `shared_dir` | `from paradex.utils.path import shared_dir` | 45+ — **single most-imported name in the repo** (`~/shared_data` NAS root). |
| `network_info` | `from paradex.utils.system import network_info` | 19 — parsed `network.json` (ports/IPs for the distributed PCs). |
| `listen_keyboard` | `from paradex.utils.keyboard_listener import listen_keyboard` | 17 |
| `home_path` | `from paradex.utils.path import home_path` | 15+ |
| `rsc_path` | `from paradex.utils.path import rsc_path` | 10+ (repo `rsc/` — URDFs, meshes) |
| `get_logger` | `from paradex.utils.log import get_logger` | 10 |
| `get_pc_list` / `get_pc_ip` | `from paradex.utils.system import ...` | 8 |
| `rsync_copy` | `from paradex.utils.upload_file import rsync_copy` | 7 |
| `config_dir` | `from paradex.utils.system import config_dir` | 4 (path to `system/current/`) |
| `find_latest_directory` / `find_latest_index` | `from paradex.utils.file_io import ...` | 7 |

Callers span nearly all of `src/` and much of `paradex/` (camera_system, calibration, robot,
`dataset_acqusition`, io/capture_pc). This is foundational plumbing — **assume any change is
wide-blast-radius**.

## Cross-links
- `log.py`'s `get_logger` is the **same** logger the robot and camera stacks use — see the fault-log
  contract in [`agent_docs/robot_controller/internals.md`](../robot_controller/internals.md) §3 and
  [`agent_docs/camera_system/`](../camera_system/README.md). Don't fork it.
- `network_info` / `pc_info` come from the active `system/current/` symlink; the camera and robot
  factories resolve device params out of them.
- Note the intentional repo-wide typo `dataset_acqusition` (missing `i`) — do **not** "fix" it.
