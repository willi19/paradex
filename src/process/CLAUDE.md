# CLAUDE.md — src/process

## Purpose
Post-capture processing scripts. Turn raw capture sessions into derived artifacts: synced sensor data, robot-overlay videos, contact maps, turntable rotations, object masks, COLMAP reconstructions.

## Subdirectories
- `miyungpa/` — robot-demo dataset: sync arm/hand to video, render overlay/merged videos, compute object contact, web + viser viewers. **On `paradex.process`** (`worker.py` + `main.py`, distributed across capture PCs).
- `object_turntable/` — turntable object scan: extract_video → extract_charuco → get_rotation → extract_mask_sam3 → generate_colmap. **On `paradex.process`** (`worker.py`, single-machine/GPU).
- `object_turntable/check/` — validation/cleanup scripts (not part of the main flow).
- `template/` — copy-me example for the `paradex.process` batch framework (see [`paradex/process/`](../../paradex/process/)); clone it to start a new processing job.

## Data flow & IO
- Two roots: `shared_dir` = `~/shared_data` (NAS), `home_path/paradex_download` = local working copy.
- Datasets keyed as `capture/<pipeline>/<obj_name>/<index>/`.
- miyungpa: download from shared → process local → upload back.
- turntable: operates entirely on the local `paradex_download` copy.

## When working here
- Both pipelines now run through `paradex.process` via a `worker.py` (discover+process). miyungpa is distributed (`main.py` → `run_distributed`); object_turntable is single-machine (`worker.py` local). See [`template/`](template/CLAUDE.md) and [`paradex/process/`](../../paradex/process/) for the framework (skip/cache/upload/status/ETA/sharding).
- The individual object_turntable stage scripts still hard-code roots in their `__main__`; the `worker.py` supersedes hand-running them in sequence.

## Gotchas
- Module `paradex.dataset_acqusition` is misspelled (missing 'i') — intentional, do not "fix".
- `object_turntable/check/check_colmap copy.py` is a duplicate of `check_colmap.py` (filename has a space) — leftover, not used.
- `object_turntable/deprecated/` holds old mask-extraction experiments — stale, ignore.
