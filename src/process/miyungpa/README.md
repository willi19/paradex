# Miyungpa Demo Processing

Post-processing for the "miyungpa" robot-manipulation demos (XArm + Inspire hand). Syncs arm/hand state to camera frames, renders robot-mesh overlay videos, computes object-contact maps, and serves dataset viewers.

## Scripts
| File | Purpose |
|------|---------|
| `process.py` | Standalone single-machine processing of demos: download videos, sync sensors, render overlay + merged videos, upload results. Has timing instrumentation. Processes the last 2 objects in the capture dir. |
| `process_main.py` | Orchestrator (main PC). Builds the demo list and distributes `process:<demo_path>` tasks over ZMQ (port 5555) to a fixed pool of capture PCs (`capture13–16,18`). |
| `process_client.py` | Worker (capture PC). ZMQ `REP` server that handles `process:<demo_path>` tasks: download → match_sync → overlay → upload. Same logic as `process.py` but driven remotely. |
| `compute_tactile.py` | Viser GUI click-handler fragment: for the current timestep, finds nearest hand-link↔object vertices, transfers contact to object mesh, writes colored/categorized contact meshes and pickles. **Not a standalone script** (no imports/header). |
| `visualizer.py` | Loads a demo + optimized object pose pickle, opens a `ViserViewer` with the robot trajectory, object mesh, floor, and contact module. Hard-coded to object `clock`. |
| `web.py` | Flask dataset gallery (port 5000): scans `shared_data/capture/miyungpa/<obj>/<date>/thumbnail.jpg` and shows a searchable thumbnail grid. |

## Usage
Distributed processing (main PC + capture PCs):
```bash
# On each capture PC:
python src/process/miyungpa/process_client.py
# On the main PC:
python src/process/miyungpa/process_main.py
```

Single-machine processing (no distribution):
```bash
python src/process/miyungpa/process.py   # edit the obj loop at bottom first
```

Viewers:
```bash
python src/process/miyungpa/web.py          # http://0.0.0.0:5000 thumbnail gallery
python src/process/miyungpa/visualizer.py   # viser 3D view of robot+object+contact
```

Per-demo processing order (in `process_demo`): `download_dir` → `match_sync` → `overlay` → `upload_output`.

## Inputs & Outputs
Input demo layout under `shared_data/capture/miyungpa/<obj>/<index>/`:
- `videos/<serial>.avi` — per-camera videos
- `raw/timestamps/{timestamp.npy,frame_id.npy}` — frame sync
- `raw/arm/`, `raw/hand/` — sensor `.npy` + `time.npy`
- camera params + cam-to-robot (loaded via `load_camparam`, `load_c2r`)
- `<obj>_optim/final/obj_output_after_optim_total_fibonacci_5_.pickle` — optimized object pose (visualizer only)
- `mesh/<obj>.ply` (under `shared_dir/mesh`) — object mesh (visualizer only)

Outputs:
- `<demo>/arm/`, `<demo>/hand/` — synced sensor `.npy` (one per source array)
- `<demo>/overlay/<serial>.avi` — per-camera robot-mesh overlays
- `<demo>/merged.mp4` (and `.avi`) — merged multi-view overlay video
- `<demo>/contact/` — `debug_*.obj`, `object_min_dist*.pickle`, `robot_pose_in_contact.pickle` (from `compute_tactile.py`)

Note: `process.py` and `process_client.py` differ slightly in where they rsync outputs (videos dir vs. shared demo dir) and whether they skip already-processed demos.

## Related
- paradex: `paradex.dataset_acqusition.match_sync` (`get_synced_data`, `fill_framedrop`), `paradex.image.image_dict.ImageDict` (`project_mesh`), `paradex.image.merge.merge_image`, `paradex.visualization.robot.RobotModule`, `paradex.visualization.visualizer.viser.ViserViewer`, `paradex.robot.inspire.parse_inspire`, `paradex.video.util.convert_avi_to_mp4`, `paradex.utils.upload_file.rsync_copy`, `paradex.io.capture_pc.ssh.run_script`
- Capture side: [`src/dataset_acquisition/miyungpa`](../../dataset_acquisition/miyungpa)
