# agent_docs/camera_system — agent orientation

Docs for **AI agents working on the camera system** (routed here from the repo-root
`AGENTS.md`). Read **one** file for your task; don't scan everything.

| Your task | Read |
|-----------|------|
| Capture images/video from *another* program (use `rcc`) | [`usage.md`](usage.md) |
| Change camera **internals** (daemon / acquisition / pyspin / lifecycle) | [`internals.md`](internals.md) |
| Understand a specific function | its NumPy-style docstring in the `.py` (start from `internals.md` §5 to find which one) |
| Design / roadmap / known limits (redesign) | `design/camera-recording-redesign.md` (repo root — deliberately not here) |

Rule of thumb: **calling** the camera system → `usage.md`; **editing** it → `internals.md`.
Both exist so agents stop rediscovering the layer structure from scratch every session.
