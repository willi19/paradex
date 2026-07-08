# CLAUDE.md — src/process/miyungpa

## Purpose
Process miyungpa robot demos (XArm + Inspire hand): sync sensors to video, render robot-mesh overlay/merged videos, compute object contact, serve viewers.

## Files
- `worker.py` — **the processing job**, on `paradex.process`. `discover()` = one `Job` per `capture/miyungpa/<obj>/<index>` (skip when `merged.mp4` exists on NAS); `process(job, ctx)` = download_dir → match_sync → overlay → upload_output. `overlay` reads per-cam videos, runs `RobotModule.get_robot_mesh()` per frame, projects via `ImageDict.project_mesh`, writes per-cam + merged video, converts avi→mp4, and reports `ctx.status(frame=i, total=N)` for frame-level ETA. Run `--local` (single machine) or bare (distributed shard). Uses the "local IO in meta" pattern — no framework inputs/outputs, does its own rsync.
- `main.py` — main-PC orchestrator: `run_distributed("python src/process/miyungpa/worker.py")`. Replaces the old ZMQ REQ/REP `process_main.py`/`process_client.py` (removed) with the shared framework dashboard (per-PC counts, per-demo frame progress, rig ETA).
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
- Distributed mode: just run `python src/process/miyungpa/main.py` on the main PC — it SSH-launches `worker.py` (its shard) on every capture PC and monitors. Single machine: `python src/process/miyungpa/worker.py --local`.
- `arm`/`hand` position.npy must already be synced for overlay (overlay loads `<demo>/hand/position.npy`); `process()` runs match_sync before overlay so it self-satisfies.
- Framework guarantees (skip-if-done, exception capture, status/ETA, sharding) come from `paradex.process` — see [`src/process/template/`](../template/CLAUDE.md). To tune parallelism, edit `num_workers` in `worker.py`'s `__main__` (kept low: overlay is GPU/mem-heavy).

## Gotchas
- `paradex.dataset_acqusition` misspelled (missing 'i') — intentional.
- `visualizer.py` fixed to `clock`; `web.py` BASE_PATH absolute to `/home/temp_id`. Edit before reuse.
- `compute_tactile.py` is a fragment, not a module.
