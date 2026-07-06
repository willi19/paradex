# agent_docs/

Docs written **for AI agents** (Claude / Codex / Cursor) working on this repo — kept
here, separate from the source tree and from `docs/` (the human-facing generated site),
so agents stop rediscovering subsystem structure from scratch every session.

- One subdirectory per subsystem (e.g. [`camera_system/`](camera_system/)).
- The repo-root [`AGENTS.md`](../AGENTS.md) is the thin router; it points here per task.
- **Read only the one file for your task** — don't scan the whole tree.

| Subsystem | Start at |
|-----------|----------|
| Camera system | [`camera_system/README.md`](camera_system/README.md) |

Design/roadmap docs stay in `design/` (repo root); generated API reference stays in
`docs/`. This directory is orientation, not the plan and not the reference.
