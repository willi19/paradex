# CLAUDE.md — src/inference/grasp_eval

## Purpose
Grasp evaluation for the pringles can using DexGraspNet grasps. Estimate object 6D pose
from the rig, then visualize (Viser) or execute (GUI on real robot) the grasp.

## Files
- `real/` — the only subdir; runs against the physical XArm + Allegro. See `real/CLAUDE.md`.

## paradex modules used
See `real/CLAUDE.md` — same backbone as the rest of `src/inference`.

## Data flow & IO
- Grasps from `dexgraspnet/results/pringles/<index>/{qpos,wrist_6d}.npy`.
- Marker offsets from `shared_dir/object/marker_offset/pringles/0/marker_offset.npy`.
- Captures to `~/shared_data/inference/grasp_eval/<timestamp>/`.

## When working here
- Hardcoded object `pringles` and grasp `index` (6 in current scripts).
- Run from repo root so `dexgraspnet/results/...` resolves.

## Gotchas
- `real/asdf.py` is broken scratch (undefined symbols) — do not treat as runnable.
