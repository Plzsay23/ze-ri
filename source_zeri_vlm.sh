#!/usr/bin/env bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export ZERI_ROOT="$SCRIPT_DIR"
export ZERI_TOOLS="$ZERI_ROOT/tools"

source /opt/ros/jazzy/setup.bash

if [ -f "$ZERI_ROOT/.venv/bin/activate" ]; then
  source "$ZERI_ROOT/.venv/bin/activate"
fi

export PYTHONPATH="$ZERI_ROOT/src:${PYTHONPATH:-}"

echo "[Ze-Ri VLM]"
echo "  ZERI_ROOT=$ZERI_ROOT"
echo "  ZERI_TOOLS=$ZERI_TOOLS"
echo "  python=$(which python)"
echo "  ROS_DISTRO=${ROS_DISTRO:-unknown}"
