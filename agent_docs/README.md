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
| Calibration (params & pipeline) | [`calibration/README.md`](calibration/README.md) |
| Dataset acquisition (`CaptureSession`, `match_sync`) | [`dataset_acquisition/README.md`](dataset_acquisition/README.md) |
| Image layer (`ImageDict`, aruco, undistort, render) | [`image/README.md`](image/README.md) |
| Capture-PC orchestration (SSH/rsync, command & telemetry transport) | [`capture_pc/README.md`](capture_pc/README.md) |
| Batch-processing framework (`Job`/`Processor`/`run_jobs`, distributed) | [`process/README.md`](process/README.md) |
| Robot controllers (arms/hands IO — `get_arm`/`get_hand`, control loop, fault handling) | [`robot_controller/README.md`](robot_controller/README.md) |
| Robot kinematics/planning (`RobotWrapper`, `CuroboPlanner`, URDF) | [`robot/README.md`](robot/README.md) |
| Teleop input (XSens / Oculus motion-capture receivers) | [`teleop/README.md`](teleop/README.md) |
| Retargeting (human hand pose → robot hand/arm joints) | [`retargetor/README.md`](retargetor/README.md) |
| Simulator (IsaacGym / PhysX physics wrapper) | [`simulator/README.md`](simulator/README.md) |
| Transforms (rigid-align `SOLVE_XA_B`, triangulation, device frames) | [`transforms/README.md`](transforms/README.md) |
| Utils (config/paths/file-I/O grab-bag — `shared_dir`, `network_info`) | [`utils/README.md`](utils/README.md) |
| Video post-processing (raw→undistort→H.264→NAS upload) | [`video/README.md`](video/README.md) |
| Visualization (`ViserScene`/`Open3DScene` — 3D viewer + figure/video render) | [`visualization/README.md`](visualization/README.md) |

Design/roadmap docs stay in `design/` (repo root); generated API reference stays in
`docs/`. This directory is orientation, not the plan and not the reference.
