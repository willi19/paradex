#!/bin/bash
# Launch franka_daemon inside the libfranka Docker image (host-side wrapper).
#
# The daemon holds the 1 kHz real-time FCI connection to the FR3 and exposes
# ZMQ command(5555)/state(5556) sockets on the host network for the Python
# FrankaController to connect to.
#
# Usage:
#   ./run_daemon.sh [FCI_IP] [CMD_PORT] [STATE_PORT]
# Defaults target this machine's 2nd Franka (장자).
#
# Prereqs (see docs/franka.md):
#   - Franka Desk: Execution mode, joints unlocked, FCI activated
#   - sudo ufw allow from <FCI_IP>
#   - Real-time control needs a PREEMPT_RT host kernel; without it libfranka
#     may fail with "Communication constraints violation".
set -e

FCI_IP="${1:-172.17.1.11}"
CMD_PORT="${2:-5555}"
STATE_PORT="${3:-5556}"

IMAGE="franka3/real:bimanual_control_libfranka0.18.0_v2"
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"   # paradex repo root

echo "=== franka_daemon launcher ==="
echo "  FCI IP     : ${FCI_IP}"
echo "  cmd/state  : ${CMD_PORT}/${STATE_PORT}"
echo "  repo mount : ${REPO} -> /workspace/paradex"
echo "  image      : ${IMAGE}"

docker run -it --rm \
  --network host \
  --cap-add=SYS_NICE \
  --ulimit rtprio=99 \
  --ulimit memlock=-1:-1 \
  -v "${REPO}":/workspace/paradex \
  "${IMAGE}" bash -lc "
    set -e
    cd /workspace/paradex/cpp/franka_daemon
    if [ ! -x build/franka_daemon ]; then
      echo '[build] daemon binary missing -> building (one-time)'
      bash build_daemon.sh
    fi
    echo '[run] starting franka_daemon'
    exec ./build/franka_daemon ${FCI_IP} --command_port ${CMD_PORT} --state_port ${STATE_PORT}
  "
