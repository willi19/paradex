set -euo pipefail

ROOT="/home/temp_id/shared_data/capture/hri_inspire_left"
ARM="xarm"
HAND="inspire"

find "$ROOT" -mindepth 2 -maxdepth 2 -type d | while read -r EP_DIR; do
  OBJ=$(basename "$(dirname "$EP_DIR")")
  EP=$(basename "$EP_DIR")

  OUT_DIR="/home/temp_id/shared_data/capture/hri_inspire_left"

  echo "Processing $OBJ/$EP ..."
  python src/util/robot/project_robot.py \
    --arm "$ARM" --hand "$HAND" \
    --object "$OBJ" --episode "$EP" \
    --start-frame 0 --stride 1 \
    --output-dir "$OUT_DIR" \
    --overlay-option "position"
done