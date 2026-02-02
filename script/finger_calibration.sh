#!/usr/bin/env bash
set -euo pipefail

BASE_DIR="/home/temp_id/shared_data/0127_inspire_pinky_calibration_2"
HAND_JSON="/home/temp_id/shared_data/0127_inspire_pinky_calibration_2/998/lookup_arm_to_hand.json"
HAND_STATE_KEY="998"

for ep_dir in "$BASE_DIR"/*; do
  if [[ -d "$ep_dir" ]]; then
    ep="$(basename "$ep_dir")"
    python src/util/inspire/calibration_finger_via_silhouette.py \
      --base-dir "$BASE_DIR" \
      --ep "$ep" \
      --gd-steps 1000 \
      --debug-every 1000 \
      --hand-json "$HAND_JSON" \
      --hand-state-key "$HAND_STATE_KEY"
  fi
done
