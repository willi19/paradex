# AGENTS.md

Thin index for AI agents (Codex / Cursor / Claude). **Read this, then open ONLY the
one doc for your task — do not scan the whole repo.** Detailed guidance lives next to
the code it describes (each `src/<app>/` and key module has its own README/CLAUDE);
prefer the local one over anything global.

Paradex = distributed multi-camera vision + robot framework. `paradex/` = library,
`src/` = apps, `system/current/` = per-machine config, `docs/` = generated site.

## Where to look (open only what you need)

| Your task | Open |
|-----------|------|
| Capture images/video from code (use `rcc`) | `agent_docs/camera_system/usage.md` |
| Change camera internals (daemon / acquire / pyspin) | `agent_docs/camera_system/internals.md` |
| Record a session / align sensor logs (`CaptureSession`, `match_sync`) | `agent_docs/dataset_acquisition/README.md` |
| Multi-cam image ops (`ImageDict`, aruco detect, undistort, mesh render) | `agent_docs/image/README.md` |
| Post-capture video processing (raw→undistort→H.264→NAS upload) | `agent_docs/video/README.md` |
| Camera design / roadmap / known limits | `design/camera-recording-redesign.md` |
| Robot / capture / process subsystems | `docs/robot.md` · `docs/capture.md` · `docs/process.md` |
| Run/understand a specific app | `src/<app>/README.md` (+ `CLAUDE.md`) |
| Full doc site | `willi19.github.io/paradex` |

## Global conventions
- Camera serial numbers are string keys everywhere.
- Module typo `dataset_acqusition` (missing 'i') is intentional — don't "fix" it.
- `system/current/` is per-machine config (not in git); never hardcode PC lists / serials.
- No test suite — validate via `src/validate/` scripts.
