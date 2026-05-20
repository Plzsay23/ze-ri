#!/usr/bin/env bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export ZERI_ROOT="$SCRIPT_DIR"
export NBYTICS_ROOT="$ZERI_ROOT/NBYtics"

source /opt/ros/jazzy/setup.bash

if [ -f "$NBYTICS_ROOT/install/setup.bash" ]; then
  source "$NBYTICS_ROOT/install/setup.bash"
fi

if [ -f "$NBYTICS_ROOT/.venv/bin/activate" ]; then
  source "$NBYTICS_ROOT/.venv/bin/activate"
fi

export PYTHONPATH="$NBYTICS_ROOT/src:$ZERI_ROOT/src:${PYTHONPATH:-}"

echo "[Ze-Ri NBYtics]"
echo "  ZERI_ROOT=$ZERI_ROOT"
echo "  NBYTICS_ROOT=$NBYTICS_ROOT"
echo "  python=$(which python)"
echo "  ROS_DISTRO=${ROS_DISTRO:-unknown}"
