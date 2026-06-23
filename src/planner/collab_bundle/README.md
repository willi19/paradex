# Teaser bundle: pringles_new

`predictions.npz` was produced by our Wan 2.1 + v5_taeksoo decoder.
The included `make_teaser.py` is the version this run was built against — use
it directly (don't substitute a copy from a different branch).

## Quick start

```bash
bash run.sh
```

Produces `teaser_pred.mp4`: front view + side view of the predicted hands,
camera trail, frame-0 point cloud backdrop, and obj mesh trajectory in 3D.

## Dependencies

Python env (any Python ≥ 3.9, e.g. `conda create -n teaser python=3.10`):

```bash
pip install numpy opencv-python pyrender trimesh scipy smplx
pip install torch torchvision  # CPU is fine; GPU optional
pip install git+https://github.com/microsoft/MoGe.git    # MoGe-2, for the PCD
```

System:
- `PYOPENGL_PLATFORM=egl` (already set in run.sh) or a working OpenGL context
- MANO model files (license-gated — download from https://mano.is.tue.mpg.de/):
  place `MANO_LEFT.pkl` and `MANO_RIGHT.pkl` under
  `<repo_root>/data_processing/HaWoR/_DATA/data/mano/` *or* set
  `MANO_DIR=/abs/path/to/mano` before running and patch `make_teaser.py`'s
  hard-coded `mano_dir` accordingly.

## What `run.sh` runs

```bash
python make_teaser.py \
    --npz predictions.npz \
    --video source_video.mp4 \
    --output teaser_pred.mp4 \
    --view_back 0.20 --view_up 0.10 --view_right 0.05 \
    --obj_models_dir obj_models \
    --rescale_pcd_to_hands
```

`--rescale_pcd_to_hands` aligns the MoGe-derived point cloud to the metric
of the hand poses (HaWoR/MegaSaM). Without it, hands and PCD live in slightly
different scales and the teaser looks like the apple+hands float relative to
the scene. Recommended for any SAM-3D-derived run.

## NPZ contents

- Hands: `left_trans`, `left_root_orient`, `left_hand_pose`, `left_betas`,
  `left_valid` (and `right_*`) — MANO params in world frame (≈ frame-0 cam).
- Camera: `cam_traj` (T, 7), `cam_scale`, `cam_focal`, `cam_center`,
  `pred_cam` (T, 10) — frame-0 cam frame.
- Object: `obj_pred` (T, 9, rows-convention 6D + trans), `obj_id=1`,
  `obj_name="pringles"`. The mesh in `obj_models/` matches `obj_id`.

6D rotation is pytorch3d's *rows* convention (first two rows of R, flat).
`make_teaser.py` decodes it correctly — earlier copies of make_teaser.py
(pre-2026-05-04) used columns and produced visibly transposed obj rotations.
