#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  project_all.sh --object <name> --object-mesh-name <mesh_name> [options] [-- <extra args to project_all.py>]

Required:
  --object              Object name (e.g. red_ramen)
  --object-mesh-name    Mesh name (e.g. red_ramen_von)

Options:
  --capture-root <name> Capture root under ~/shared_data/capture (default: hri_xarm_f1)
  --grid-scale <float>  Grid scale for project_all.py (default: 0.25)
  --python <bin>        Python executable (default: python)
  --dry-run             Print commands only
  -h, --help            Show this help

Examples:
  ./project_all.sh --object red_ramen --object-mesh-name red_ramen_von
  ./project_all.sh --object red_ramen --object-mesh-name red_ramen_von -- --output-type video
EOF
}

OBJECT=""
OBJECT_MESH_NAME=""
CAPTURE_ROOT="hri_xarm_f1"
GRID_SCALE="0.25"
PYTHON_BIN="python"
DRY_RUN=0
EXTRA_ARGS=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --object)
      OBJECT="${2:-}"
      shift 2
      ;;
    --object-mesh-name)
      OBJECT_MESH_NAME="${2:-}"
      shift 2
      ;;
    --capture-root)
      CAPTURE_ROOT="${2:-}"
      shift 2
      ;;
    --grid-scale)
      GRID_SCALE="${2:-}"
      shift 2
      ;;
    --python)
      PYTHON_BIN="${2:-}"
      shift 2
      ;;
    --dry-run)
      DRY_RUN=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    --)
      shift
      EXTRA_ARGS+=("$@")
      break
      ;;
    *)
      EXTRA_ARGS+=("$1")
      shift
      ;;
  esac
done

if [[ -z "$OBJECT" || -z "$OBJECT_MESH_NAME" ]]; then
  echo "Error: --object and --object-mesh-name are required." >&2
  usage
  exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../../../.." && pwd)"
PROJECT_PY="${REPO_ROOT}/src/util/robot/project/project_all.py"
CAPTURE_BASE="${HOME}/shared_data/capture/${CAPTURE_ROOT}/${OBJECT}"

if [[ ! -f "$PROJECT_PY" ]]; then
  echo "Error: project_all.py not found: ${PROJECT_PY}" >&2
  exit 1
fi

if [[ ! -d "$CAPTURE_BASE" ]]; then
  echo "Error: object directory not found: ${CAPTURE_BASE}" >&2
  exit 1
fi

mapfile -t CAPTURE_EPS < <(
  find "$CAPTURE_BASE" -mindepth 1 -maxdepth 1 -type d -printf '%f\n' \
    | grep -E '^[0-9]+$' \
    | sort -V
)

if [[ "${#CAPTURE_EPS[@]}" -eq 0 ]]; then
  echo "Error: no capture-ep directories found under ${CAPTURE_BASE}" >&2
  exit 1
fi

echo "Object: ${OBJECT}"
echo "Mesh: ${OBJECT_MESH_NAME}"
echo "Capture root: ${CAPTURE_ROOT}"
echo "Found capture-ep: ${#CAPTURE_EPS[@]}"

for idx in "${!CAPTURE_EPS[@]}"; do
  ep="${CAPTURE_EPS[$idx]}"
  output_mp4="${CAPTURE_BASE}/${ep}/overlay_position/grid/grid_overlay.mp4"

  if [[ -f "$output_mp4" ]]; then
    printf '[%d/%d] capture-ep=%s (skip: exists %s)\n' \
      "$((idx + 1))" "${#CAPTURE_EPS[@]}" "$ep" "$output_mp4"
    continue
  fi

  cmd=(
    "$PYTHON_BIN" "$PROJECT_PY"
    --object "$OBJECT"
    --object-mesh-name "$OBJECT_MESH_NAME"
    --capture-root "$CAPTURE_ROOT"
    --capture-ep "$ep"
    --project-object
    --project-robot
    --grid-scale "$GRID_SCALE"
  )
  cmd+=("${EXTRA_ARGS[@]}")

  printf '[%d/%d] capture-ep=%s\n' "$((idx + 1))" "${#CAPTURE_EPS[@]}" "$ep"
  printf '  '
  printf '%q ' "${cmd[@]}"
  printf '\n'

  if [[ "$DRY_RUN" -eq 0 ]]; then
    "${cmd[@]}"
  fi
done
