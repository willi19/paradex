# paradex/utils — How to Call Each Helper

Recipes for the utility grab-bag. Every symbol below is imported **from its submodule**
(`paradex.utils.path`, not `paradex.utils` — the package `__init__.py` is empty).

> Editing one of these helpers, or hunting a path/import trap? Read [`internals.md`](internals.md).

---

## 1. Path constants — [`path.py`](../../paradex/utils/path.py)

No functions, no arguments — just module-level strings computed **at import** from `$HOME`.

```python
from paradex.utils.path import (
    home_path,          # os.path.expanduser("~")
    pc_name,            # basename of ~  → this PC's identity, e.g. "paradex1"
    shared_dir,         # ~/shared_data  → the NAS root (datasets, cam_param, log, ...)
    capture_path_list,  # [~/captures1, ~/captures2]  (hard-coded to 2)
    download_dir,       # ~/download
    model_dir,          # <repo>/model   (relative to path.py, cwd-independent)
    rsc_path,           # <repo>/rsc     (URDFs, meshes)
)

session = os.path.join(shared_dir, "my_dataset", "001")
```

`shared_dir` is the single most-imported name in the repo — everything on the NAS hangs off it.
`rsc_path`/`model_dir` resolve relative to `path.py`'s location, so they point at the repo checkout
regardless of your working directory. There is **no override argument** for any of these.

---

## 2. This-PC config — [`system.py`](../../paradex/utils/system.py)

Reads `system/current/pc.json` and `network.json` **at import** (see the symlink note in
[`internals.md`](internals.md)). `import` **fails immediately** on a box missing those files.

```python
from paradex.utils.system import (
    config_dir,     # path to system/current/
    pc_info,        # parsed pc.json      (dict keyed by PC name)
    network_info,   # parsed network.json (ports/IPs for the distributed rig)
    pc_name,        # this PC's identity (same value as path.pc_name, computed independently)
    get_pc_ip, get_camera_list, get_pc_list, get_camera_config,
)

ip       = get_pc_ip("paradex2")              # -> pc_info["paradex2"]["ip"]           (KeyError if absent)
cams     = get_camera_list()                  # -> pc_info[<this pc>]["cam_list"]; pass pc_name= for another PC
all_pcs  = get_pc_list()                      # -> list(pc_info.keys())
cam_cfg  = get_camera_config()                # -> parsed camera.json, or {} if the file is absent (lazy, optional)
```

| Call | Returns | Notes |
|------|---------|-------|
| `get_pc_ip(pc_name)` | `str` | raw index `pc_info[pc_name]["ip"]` — `KeyError` on unknown PC |
| `get_camera_list(pc_name=<this pc>)` | `list[str]` | camera serials for a PC; defaults to this PC |
| `get_pc_list()` | `list[str]` | all PC names in `pc.json` |
| `get_camera_config()` | `dict` | `camera.json` if present, else `{}` — the only lazy/optional read |

---

## 3. File & data loaders — [`file_io.py`](../../paradex/utils/file_io.py)

```python
from paradex.utils.file_io import (
    load_yaml, find_latest_directory, find_latest_index,
    is_image_file, load_images, remove_home,
    load_robot_traj, load_robot_target_traj, load_obj_traj,
    load_contact_value, load_robot_traj_prev,
)

cfg   = load_yaml("system/current/lens_info.yaml")   # -> dict (canonical yaml loader; uses yaml.Loader)
newest = find_latest_directory(shared_dir)           # newest child by STRING sort, or None if empty
idx    = find_latest_index(session_dir)              # newest child by INT sort, or -1 if missing/empty
imgs   = load_images(frame_dir)                      # [paths] of *.png/*.jpg/*.jpeg in dir (unsorted)
key    = remove_home("/home/mingi/shared_data/x")    # -> "shared_data/x" (path relative to ~)
```

| Loader | Reads | Returns |
|--------|-------|---------|
| `load_yaml(path)` | a `.yaml` **path (str)** | `dict` — **only** handles `str` input (see gotcha) |
| `find_latest_directory(dir)` | `os.listdir` | newest name by `max(..., key=str)`, `None` if empty |
| `find_latest_index(dir)` | `os.listdir` | newest name by `max(..., key=int)`, `-1` if dir missing/empty; **raises** on non-numeric names |
| `is_image_file(name)` | filename | `bool` (`.png/.jpg/.jpeg`) |
| `load_images(dir)` | `os.listdir` | `list[str]` of image paths |
| `remove_home(path)` | path | `str` relative to `Path.home()` — used to derive NAS `root_name` keys |

Demo-trajectory loaders expect the fixed demo layout (`arm/`, `hand/`, `contact/`, `obj_traj.pickle`):

| Loader | File(s) read | Returns |
|--------|--------------|---------|
| `load_robot_traj(demo)` | `arm/state.npy` + `hand/state.npy` | concatenated `(T, arm+hand)` |
| `load_robot_target_traj(demo)` | `arm/action.npy` + `hand/action.npy` | concatenated targets |
| `load_obj_traj(demo)` | `obj_traj.pickle` | unpickled object trajectory |
| `load_contact_value(demo)` | `contact/data.npy` | contact array |
| `load_robot_traj_prev(demo)` | `robot_qpos.npy` (legacy layout) | arm qpos |

> **Gotchas.** `load_yaml` only assigns its result when the arg is a `str`; passing a dict (despite
> the docstring) raises `UnboundLocalError`. `load_mesh(obj_name)` references `rsc_path` that
> `file_io.py` never imports → `NameError` if called (dead helper). Details in [`internals.md`](internals.md).

---

## 4. Shared logger — [`log.py`](../../paradex/utils/log.py)

```python
from paradex.utils.log import get_logger

logger = get_logger("my_app")        # name = logger name AND file stem — keep it short & fs-safe
logger.info("started")
```

Writes to **console + a NAS file**:

```
~/shared_data/log/<pc_name>/<YYYYMMDD>/<my_app>_<HHMMSS>.log
```

- **Idempotent per process**: a second `get_logger("my_app")` returns the same logger and file
  (it checks `logger.handlers`). One process run == one file.
- **Level `INFO`**, `propagate=False` (won't double-log through the root logger).
- **NAS down?** The file handler is skipped and it logs a warning + continues console-only — never
  raises. This is the same logger the robot/camera stacks use; route faults through it, not `print`.

---

## 5. Keyboard listener — [`keyboard_listener.py`](../../paradex/utils/keyboard_listener.py)

Non-blocking stdin → `threading.Event` mapping for interactive capture scripts.

```python
import threading
from paradex.utils.keyboard_listener import listen_keyboard, stop_listening

events = {"s": threading.Event(), "q": threading.Event()}
listen_keyboard(events)              # spawns a daemon thread reading input() lines

if events["s"].is_set():             # poll your events in the main loop
    events["s"].clear()
    ...

stop_listening()                     # stop the listener thread(s)
```

- Keys are matched **case-insensitively and stripped** (`input().strip().lower()`); typed key must
  be a full line + Enter (it uses blocking `input()`, one key per line).
- Unknown keys print `Unknown key: <k>`; on match it prints `[k] event triggered`.
- The thread is a `daemon` and exits on `EOFError`/`KeyboardInterrupt` or when `stop_listening()`
  is called. **`stop_listening()` stops every listener in the process** (shared module-level event).

---

## 6. rsync copy/move — [`upload_file.py`](../../paradex/utils/upload_file.py)

```python
from paradex.utils.upload_file import rsync_copy

ok = rsync_copy(src, dst, move=False, resume=True, dry_run=False,
                compress=False, checksum=True, verbose=False)
if not ok:
    ...   # rsync missing, src missing, or non-zero exit — it returns False, does NOT raise
```

| Arg | Default | Effect |
|-----|---------|--------|
| `move` | `False` | adds `--remove-source-files` → **deletes source on success** |
| `resume` | `True` | adds `--inplace` (write in place, resumable) |
| `dry_run` | `False` | `--dry-run`, no writes |
| `compress` | `False` | `-z` (useful on slow links) |
| `checksum` | `True` | `--checksum` verification |
| `verbose` | `False` | `-v --stats`; otherwise `--quiet` |

Always runs `-a -h --info=progress2 --partial --no-owner --no-group --progress`. Auto-creates the
destination dir (a `dst` with a suffix is treated as a file → makes `dst.parent`). Returns `True`/`False`.

Companions: `check_rsync_installed()` (bool), `get_file_size(path)` (bytes, recurses dirs),
`format_size(bytes)` (`"1.50 GB"`), `estimate_time(src)` (prints an ETA assuming ~100 MB/s NAS).
