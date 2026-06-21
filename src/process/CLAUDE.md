# CLAUDE.md — src/process

## Purpose
Post-capture processing scripts. Turn raw capture sessions into derived artifacts: synced sensor data, robot-overlay videos, contact maps, turntable rotations, object masks, COLMAP reconstructions.

## Subdirectories
- `miyungpa/` — robot-demo dataset: sync arm/hand to video, render overlay/merged videos, compute object contact, web + viser viewers.
- `object_turntable/` — turntable object scan: extract_video → extract_charuco → get_rotation → extract_mask_sam3 → generate_colmap.
- `object_turntable/check/` — validation/cleanup scripts (not part of the main flow).

## Data flow & IO
- Two roots: `shared_dir` = `~/shared_data` (NAS), `home_path/paradex_download` = local working copy.
- Datasets keyed as `capture/<pipeline>/<obj_name>/<index>/`.
- miyungpa: download from shared → process local → upload back.
- turntable: operates entirely on the local `paradex_download` copy.

## When working here
- Each script's `__main__`/bottom block hard-codes dataset roots and frequently a single object name (e.g. `['clock']`, `['big_green_spray']`). Edit the loop before running.
- These are research scripts, not a library — no CLI args on most, no shared entry point.

## Gotchas
- Module `paradex.dataset_acqusition` is misspelled (missing 'i') — intentional, do not "fix".
- `extract_mask_sam3.py` has a stray non-Python line (Korean comment as bare statement) on line 159 — file will not import/run as-is; it documents intent for the realdex `load_mask` path.
- `object_turntable/check/check_colmap copy.py` is a duplicate of `check_colmap.py` (filename has a space) — leftover, not used.
- `object_turntable/deprecated/` holds old mask-extraction experiments — stale, ignore.
