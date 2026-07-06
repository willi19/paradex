# AGENTS.md

Cross-tool guide for AI coding agents (Codex, Cursor, Claude, …) working in this
repo. (Claude also reads `CLAUDE.md`.)

## What this repo is

Paradex — a distributed multi-camera vision + robot control framework.
`paradex/` is the reusable library; `src/` are application scripts; `system/current/`
holds the active machine config; `docs/` is the generated site
(`willi19.github.io/paradex`).

## Using the camera system (read this before capturing images/video)

Most callers waste time rediscovering this. **On the main PC, use
`remote_camera_controller` (`rcc`)** — do not instantiate `Camera`/`CameraLoader`
directly. Full recipe, modes, error handling, gotchas:

→ **[`paradex/io/camera_system/README.md`](paradex/io/camera_system/README.md)**

Minimal:
```python
from paradex.io.camera_system.remote_camera_controller import remote_camera_controller
rcc = remote_camera_controller("my_app")
rcc.start("image", False, save_path="dataset/001/raw")   # image/video/stream/full
rcc.stop(); rcc.end()                                     # end() releases the lock
# health: rcc.get_status() -> {'error', 'stalled', 'pc': {...}}
```

## Key docs by topic

| Topic | Where |
|-------|-------|
| Camera system (how it works) | online docs `camera_system.html`; API `docs/camera_system_api.md` |
| Camera usage recipe | `paradex/io/camera_system/README.md` |
| Robot / capture / process subsystems | `docs/{robot,capture,process}.md` (+ `_api`) |
| Per-app run steps | each `src/<app>/README.md` |
| Camera redesign roadmap / known limits | `design/camera-recording-redesign.md` |

## Conventions

- Camera serial numbers are string keys everywhere.
- Module typo `dataset_acqusition` (missing 'i') is intentional — do not "rename fix" it.
- `system/current/` is per-machine config (not in git); don't hardcode PC lists / serials.
- No test suite — validate via `src/validate/` scripts.
- Docs site is built from `docs/` via GitHub Actions on push to `main`; the `docs/`
  markdown is Sphinx (MyST) — `make html` in `docs/` to rebuild.
