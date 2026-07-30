#!/bin/bash
# Force-kill the franka_daemon container.
#
# The daemon's SIGINT handler can hang ("shutting down..." on repeat) while tearing
# down an active velocity stream / control loop, so Ctrl+C in run_daemon.sh may not
# return. This SIGKILLs the container outright.
IMAGE="franka3/real:bimanual_control_libfranka0.18.0_v2"

ids=$(docker ps -q --filter "ancestor=${IMAGE}")
if [ -z "$ids" ]; then
    echo "[kill] no running franka_daemon container"
    exit 0
fi

for id in $ids; do
    echo "[kill] docker kill $id"
    docker kill "$id" >/dev/null
done
echo "[kill] done"
