#!/usr/bin/env bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export NBYTICS_ROOT="$SCRIPT_DIR"

source /opt/ros/jazzy/setup.bash

if [ -f "$NBYTICS_ROOT/install/setup.bash" ]; then
  source "$NBYTICS_ROOT/install/setup.bash"
fi

if [ -f "$NBYTICS_ROOT/.venv/bin/activate" ]; then
  source "$NBYTICS_ROOT/.venv/bin/activate"
fi

export PYTHONPATH="$NBYTICS_ROOT/src:${PYTHONPATH:-}"

echo "[NBYtics]"
echo "  NBYTICS_ROOT=$NBYTICS_ROOT"
echo "  python=$(which python)"
echo "  ROS_DISTRO=${ROS_DISTRO:-unknown}"
