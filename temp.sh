set -euo pipefail

ROOT="/home/temp_id/shared_data/capture/hri_inspire_left"
ARM="xarm"
HAND="inspire"

HOSTS=(
  "capture12"
  "capture13"
  "capture14"
  "capture15"
  "capture16-B760M-AORUS-ELITE"
  "capture18"
)
HOSTNAME_SHORT="$(hostname -s)"
HOST_IDX=-1
for i in "${!HOSTS[@]}"; do
  if [[ "${HOSTS[$i]}" == "$HOSTNAME_SHORT" ]]; then
    HOST_IDX="$i"
    break
  fi
done
if [[ "$HOST_IDX" -lt 0 ]]; then
  echo "Unknown host: $HOSTNAME_SHORT"
  exit 1
fi
NUM_HOSTS="${#HOSTS[@]}"

# Assign objects to hosts by index modulo NUM_HOSTS.
mapfile -t OBJECTS < <(find "$ROOT" -mindepth 1 -maxdepth 1 -type d -printf '%f\n' | sort)
OBJ_COUNT=0
for OBJ in "${OBJECTS[@]}"; do
  if (( (OBJ_COUNT % NUM_HOSTS) != HOST_IDX )); then
    OBJ_COUNT=$((OBJ_COUNT + 1))
    continue
  fi
  OBJ_COUNT=$((OBJ_COUNT + 1))
  find "$ROOT/$OBJ" -mindepth 1 -maxdepth 1 -type d | while read -r EP_DIR; do
    EP=$(basename "$EP_DIR")

    OUT_DIR="/home/temp_id/shared_data/capture/hri_inspire_left"

    echo "[$HOSTNAME_SHORT] Processing $OBJ/$EP ..."
    python src/util/robot/project_robot.py \
      --arm "$ARM" --hand "$HAND" \
      --object "$OBJ" --episode "$EP" \
      --start-frame 0 --stride 1 \
      --output-dir "$OUT_DIR" \
      --overlay-option "position"
  done
done
