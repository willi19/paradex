# paradex/utils — Internals (for agents editing this module)

**You are here to change or reason about the tricky helpers** (config resolution, the NAS logger,
`find_latest_*`, the keyboard listener) or to understand a path/import trap. If you only want to
*call* a helper, read [`usage.md`](usage.md).

These are seven independent files, so this is organized by helper, not by a flow. The through-line
is: **almost everything is import-time and rooted at `$HOME`/`system/current/`** — that's where the
subtlety and the traps live.

---

## 1. Config resolution + the `system/current/` symlink

`system.py` locates config by **path relative to itself**, then reads it eagerly:

```python
config_dir  = os.path.join(os.path.dirname(__file__), "..", "..", "system", "current")
pc_info     = json.load(open(os.path.join(config_dir, "pc.json"), "r"))      # at IMPORT
network_info= json.load(open(os.path.join(config_dir, "network.json"), "r")) # at IMPORT
pc_name     = os.path.basename(os.path.expanduser("~"))
```

- `system/current/` is a **symlink** to the active system config (`paradex1`, `paradex2`, …). Config
  resolution is therefore "whatever `current` points at" — swapping rigs is a symlink change, not a
  code change. `config_dir` never uses `$HOME`; it's anchored to the repo checkout.
- **Only `pc.json` and `network.json` are read at import — two files, not three.** `camera.json` is
  read lazily and optionally by `get_camera_config()` (returns `{}` if absent). Missing `pc.json`
  **or** `network.json` makes `import paradex.utils.system` raise **before any caller code runs**.
- The file handles from `json.load(open(...))` are never explicitly closed (relies on GC). Harmless
  at import scope; don't copy the pattern into a loop.
- `pc_name` is derived from `basename($HOME)` — the PC's identity **is its home-dir name**
  (`/home/paradex1` → `paradex1`). This is defined **independently in both `system.py` and
  `path.py`**; if you change how identity is derived, change both.

The device factories (camera_system, robot_controller) index `network_info`/`pc_info` directly —
see the config-schema-inconsistency note in
[`agent_docs/robot_controller/internals.md`](../robot_controller/internals.md) §7.

---

## 2. `path.py` — import-time constants, rooted two ways

```python
home_path         = os.path.expanduser("~")
pc_name           = os.path.basename(home_path)
shared_dir        = os.path.join(home_path, "shared_data")
capture_path_list = [os.path.join(home_path, f"captures{i}") for i in range(1,3)]  # captures1, captures2
model_dir = os.path.join(os.path.dirname(__file__), "..", "..", "model")
rsc_path  = os.path.join(os.path.dirname(__file__), "..", "..", "rsc")
```

Two rooting schemes coexist: `home_path`/`shared_dir`/`capture_path_list`/`download_dir` follow the
**running user's `$HOME`** (machine-dependent, no override); `model_dir`/`rsc_path` follow **this
file's location** (repo checkout, cwd-independent). Everything is evaluated once at import —
changing `$HOME` afterward has no effect.

- `capture_path_list` is **hard-coded to two dirs** via `range(1,3)`. A capture PC with a different
  number of local disks is not configurable here — you'd edit this line.

---

## 3. `log.py` — NAS-file logger behavior

`get_logger(name)` returns a console+file logger; the file path is **per-PC, per-day**:

```
~/shared_data/log/<pc_name>/<YYYYMMDD>/<name>_<HHMMSS>.log
```

- The PC/date namespacing is deliberate: the 6 capture PCs all write to the same NAS and must not
  clobber each other. `pc_name` here comes from `path.py`.
- **Idempotent per process.** First call configures handlers; the guard `if logger.handlers: return
  logger` means later `get_logger("x")` calls reuse the same logger **and the same file** (the
  `<HHMMSS>` stamp is fixed at first call). One process run == one file per name.
- **Degrades, never crashes.** The `FileHandler` construction is wrapped in `try/except OSError`; if
  the NAS isn't mounted it logs a warning and keeps the console handler only. Note it catches
  **`OSError` specifically** — a different failure class would still propagate.
- `setLevel(INFO)`, `propagate=False` (no double-logging via root), fixed format
  `"%(asctime)s [%(name)s] %(levelname)-5s %(message)s"`.
- This is the shared logger the robot and camera stacks depend on for fault diagnosis
  ([`robot_controller/internals.md`](../robot_controller/internals.md) §3). Don't add `print()`
  paths or change the file layout without checking those consumers.

---

## 4. `file_io.py` — `find_latest_*` semantics & latent bugs

- **`find_latest_directory(dir)`** = `max(os.listdir(dir), key=str)` → newest by **string** sort,
  `None` if empty. Works for zero-padded timestamps; for unpadded numbers `"9" > "10"` bites you.
  Does **not** check `os.path.exists(dir)` first, so a missing dir raises inside `os.listdir`.
- **`find_latest_index(dir)`** = `max(os.listdir(dir), key=int)` → newest by **integer** sort;
  returns `-1` if the dir is missing or empty. It **raises `ValueError`** the moment any child name
  isn't an int — use it only on purely numeric session dirs.
- **`load_yaml` only handles `str`.** The body assigns `yaml_params` only inside
  `if isinstance(file_path, str)`, so the docstring's "if file_path is a dictionary, return as is"
  is **not implemented** — passing a dict raises `UnboundLocalError`. Uses `yaml.Loader` (full, not
  safe). Fix by adding an `else: yaml_params = file_path` if you ever need the dict passthrough.
- **`load_mesh(obj_name)` is broken.** It builds `os.path.join(rsc_path, "object", ...)` but
  `file_io.py` never imports `rsc_path` → `NameError` on call. It's dead/unused; don't assume it
  works and don't rely on it.
- `load_images` is **unsorted** (`os.listdir` order) — sort at the call site if order matters.
- Demo loaders hard-code the demo dir layout (`arm/state.npy`, `hand/state.npy`, `arm/action.npy`,
  `hand/action.npy`, `contact/data.npy`, `obj_traj.pickle`, and the legacy `robot_qpos.npy`). A
  missing sub-path raises from `np.load`/`pickle.load`; there's no graceful fallback.

---

## 5. `keyboard_listener.py` — threading model

```python
stop_event = threading.Event()          # MODULE-LEVEL, shared by every listener in the process
def listen_keyboard(event_dict):
    def run(event_dict):
        while not stop_event.is_set():
            key = input().strip().lower() # BLOCKING read, one line per key
            if key in event_dict: event_dict[key].set()
    threading.Thread(target=run, args=(event_dict,), daemon=True).start()
```

- `stop_event` is **module-global** → `stop_listening()` stops **all** listeners started in the
  process, not just one. There's no per-listener handle.
- `run` calls blocking `input()`, so the loop only re-checks `stop_event` **after the next line of
  stdin arrives** — a listener can outlive `stop_listening()` until the user presses Enter. It also
  exits on `EOFError`/`KeyboardInterrupt` (e.g. piped/closed stdin).
- The thread is `daemon=True`, so it won't block process exit; callers poll the `Event`s they passed
  in (the listener only ever `.set()`s them — it never clears them, that's the caller's job).

---

## 6. `upload_file.py` — the rsync wrapper

- Builds an `rsync` argv and runs it via `subprocess.run(..., check=True)`. Returns **`False`
  (never raises)** when: source doesn't exist, `rsync` isn't installed
  (`check_rsync_installed()` probes `rsync --version` with a 2 s timeout), or rsync exits non-zero
  (`CalledProcessError` is caught). **Always check the return value.**
- Always-on flags: `-a -h --info=progress2 --partial --no-owner --no-group --progress`. Optional:
  `--remove-source-files` (`move=True`, **delete-on-success**), `--inplace` (`resume=True`,
  default), `-z` (`compress`), `--checksum` (default on), `--dry-run`, and `-v --stats` vs `--quiet`.
- Destination handling: if `dst` has a suffix it's treated as a **file** (creates `dst.parent`);
  otherwise as a **dir** (creates `dst`). Comments are in Korean — leave them.
- `get_file_size` recurses dirs via `rglob('*')`; `estimate_time` just prints an ETA at a hard-coded
  ~100 MB/s NAS assumption (advisory only).

---

## 7. Traps that look like bugs (and the ones that are)

**Intentional / leave alone:**
- **`dataset_acqusition` typo** (missing `i`) is intentional and repo-wide — never "fix" imports.
- `pc_name` defined twice (`path.py` + `system.py`) with the same value, computed independently.
- Unclosed `json.load(open(...))` handles at module scope in `system.py` — import-time only.

**Real breakage — fix at the call site, don't paper over it here:**
- **Stale consumer imports that don't exist in this module:**
  `from paradex.utils.env import get_serial_list` (there is **no** `paradex/utils/env.py`) and
  `from paradex.utils.file_io import get_robot_urdf_path` (`file_io.py` defines no
  `get_robot_urdf_path`, and no `rsc_path`/`config_dir`/`home_path` either). Those symbols actually
  live in [`paradex/io/camera_system/pyspin.py`](../../paradex/io/camera_system/pyspin.py)
  (`get_serial_list`) and [`paradex/robot/utils.py`](../../paradex/robot/utils.py)
  (`get_robot_urdf_path`). Don't propagate the broken imports and don't "add" the symbols here.
- **`load_yaml` dict passthrough** and **`load_mesh` `NameError`** (see §4) — latent, callable bugs.

**Blast radius — these names are a de-facto public API:**
- `shared_dir`, `home_path`, `network_info`, `listen_keyboard`, `get_logger`, `rsync_copy` are
  imported **by name** in 100+ places across `src/` and `paradex/`. Renaming one, or changing a
  signature/return, is a repo-wide edit, not a local one.
- **Hard-coded-path assumptions everywhere.** `shared_dir == ~/shared_data` and
  `pc_name == basename($HOME)` are baked in; there is no config/env override. Code that imports
  `paradex.utils.system` **requires** `system/current/pc.json` + `network.json` to exist at import
  time — running off a bare box (CI, a laptop) fails before your code starts.
