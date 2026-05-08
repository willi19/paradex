#!/usr/bin/env bash

if [[ $# -eq 0 ]]; then
  echo "Usage: $0 <library1> <library2> ..."
  exit 1
fi

CONDA_BASE="$HOME/anaconda3"

ENV_NAMES=(
  base DGBench MarkerGen bodex camera dexnet dexnet_table docs
  flir_env flir_env_2 isaacsim lygra object6d paradex planner tmp
)

LIBRARIES=("$@")

source "$CONDA_BASE/etc/profile.d/conda.sh"

for ENV in "${ENV_NAMES[@]}"; do
  echo "========================================"
  echo "🔹 Environment: $ENV"

  conda activate "$ENV" 2>/dev/null || {
    echo "❌ Cannot activate $ENV"
    continue
  }

  for LIB in "${LIBRARIES[@]}"; do
    if pip show "$LIB" &>/dev/null; then
      echo "✅ $LIB installed"
    else
      echo "❌ $LIB missing"
    fi
  done

  conda deactivate
done
echo "========================================"