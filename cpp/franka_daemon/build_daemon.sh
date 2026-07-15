#!/bin/bash
# Build script for FrankaDaemon
# Run this inside the Docker container

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUILD_DIR="${SCRIPT_DIR}/build"

echo "=== Building FrankaDaemon ==="
echo "Source dir: ${SCRIPT_DIR}"
echo "Build dir: ${BUILD_DIR}"

# Create build directory
mkdir -p "${BUILD_DIR}"
cd "${BUILD_DIR}"

# Configure
cmake \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_PREFIX_PATH="/opt/libfranka" \
    ..

# Build
make -j$(nproc)

echo ""
echo "=== Build Complete ==="
echo "Binary: ${BUILD_DIR}/franka_daemon"
echo ""
echo "To run:"
echo "  ${BUILD_DIR}/franka_daemon <robot-ip>"
echo ""
echo "Example:"
echo "  ${BUILD_DIR}/franka_daemon 172.17.1.11"
