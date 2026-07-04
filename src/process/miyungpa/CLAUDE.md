# CLAUDE.md — src/process/miyungpa

## Purpose
Process miyungpa robot demos (XArm + Inspire hand): sync sensors to video, render robot-mesh overlay/merged videos, compute object contact, serve viewers.

## Files
- `process.py` — single-machine driver. `process_demo` = download_dir → match_sync → overlay → upload_output. `overlay` reads per-cam videos, runs `RobotModule.get_robot_mesh()` per frame, projects via `ImageDict.project_mesh`, writes per-cam + merged video, converts avi→mp4. Has per-stage timing. Loops over `os.listdir(...)[-2:]`.
- `process_main.py` — `ProcessorMain` ZMQ `REQ` client; round-robins `process:<demo_path>` to PCs `capture13/14/15/16/18` on port 5555, threaded distribute loop, sends `quit` when done.
- `process_client.py` — ZMQ `REP` server on `tcp://*:5555`; same match_sync/overlay/upload logic as process.py. Skips demos that already have `merged.mp4`.
- `compute_tactile.py` — a `@compute_contact_button.on_click` handler body (fragment, depends on outer-scope vars: `server`, `robot_module`, `mesh_dictionary`, `contact_tg`, `sensororder`, etc.). cKDTree nearest-neighbor hand↔object, writes contact meshes/pickles to `<scene>/contact/`. Not runnable alone.
- `visualizer.py` — loads c2r, arm/hand positions, optimized object pose pickle, mesh ply; builds ViserViewer with robot/object/traj/floor/contact_module. Hard-coded `obj_name='clock'`.
- `web.py` — Flask app, scans `BASE_PATH` for `<obj>/<date>/thumbnail.jpg`, renders searchable grid + modal. `BASE_PATH` hard-coded to `/home/temp_id/shared_data/capture/miyungpa`.

## paradex modules used
`paradex.dataset_acqusition.match_sync` (get_synced_data, fill_framedrop), `paradex.image.image_dict.ImageDict`, `paradex.image.merge.merge_image`, `paradex.calibration.utils` (load_camparam, load_c2r), `paradex.visualization.robot.RobotModule`, `paradex.visualization.visualizer.viser.ViserViewer`, `paradex.robot.inspire.parse_inspire`, `paradex.robot.utils.get_robot_urdf_path`, `paradex.video.util.convert_avi_to_mp4`, `paradex.utils.upload_file.rsync_copy`, `paradex.io.capture_pc.ssh.run_script`, `paradex.utils.system` (get_pc_list, get_pc_ip).

## Data flow & IO
- Source: `shared_dir/capture/miyungpa/<obj>/<index>/` with `videos/`, `raw/timestamps/`, `raw/arm`, `raw/hand`.
- match_sync writes synced `<demo>/arm/*.npy`, `<demo>/hand/*.npy`.
- overlay works in `home_path/paradex_download/<demo>/`, writes `overlay/<serial>.avi`, `merged.{avi,mp4}`, then rsync back to shared.
- hand state passed through `parse_inspire`; robot cfg = concat(arm, hand); mesh transformed by c2r.

## When working here
- Distributed mode: start `process_client.py` on each capture PC, then `process_main.py` on main PC.
- `arm`/`hand` position.npy must already be synced for overlay (overlay loads `<demo>/hand/position.npy`); process_demo runs match_sync before overlay so it self-satisfies.

## Gotchas
- `paradex.dataset_acqusition` misspelled (missing 'i') — intentional.
- `process.py` loops `[-2:]`; `visualizer.py` fixed to `clock`; `web.py` BASE_PATH absolute to `/home/temp_id`. Edit before reuse.
- `compute_tactile.py` is a fragment, not a module; `process_main.py`'s `run_script(...)` call to launch clients is commented out.
- ZMQ sockets use `RCVTIMEO=-1` (block forever) — a stuck client hangs the orchestrator.
