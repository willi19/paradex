#!/bin/bash
set -euo pipefail
export PYOPENGL_PLATFORM=egl

# --rescale_pcd_to_hands: aligns the MoGe point cloud to HaWoR/MegaSaM metric
# using the right wrist as a reference. Required when the obj anchor was
# rescaled (SAM-3D / cross-metric runs) so the PCD background sits in the
# same world as hands+obj.
python make_teaser.py \
    --npz predictions.npz \
    --video source_video.mp4 \
    --output teaser_pred.mp4 \
    --view_back 0.20 --view_up 0.10 --view_right 0.05 \
    --obj_models_dir obj_models \
    --rescale_pcd_to_hands
