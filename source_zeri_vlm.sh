#!/usr/bin/env bash

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export ZERI_ROOT="$SCRIPT_DIR"
export ZERI_TOOLS="$ZERI_ROOT/tools"
export ZERI_ROS_WS="$ZERI_ROOT/ros2_ws"

# CUDA 13.0 for Jetson Thor
export CUDA_HOME=/usr/local/cuda
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"

source /opt/ros/jazzy/setup.bash

if [ -f "$ZERI_ROOT/.venv/bin/activate" ]; then
  source "$ZERI_ROOT/.venv/bin/activate"
fi

if [ -f "$ZERI_ROS_WS/install/setup.bash" ]; then
  source "$ZERI_ROS_WS/install/setup.bash"
fi

VENV_SITE="$ZERI_ROOT/.venv/lib/python3.12/site-packages"

export PYTHONPATH="$VENV_SITE:$ZERI_ROOT/src:${PYTHONPATH:-}"

echo "[Ze-Ri Unified]"
echo "  ZERI_ROOT=$ZERI_ROOT"
echo "  ZERI_ROS_WS=$ZERI_ROS_WS"
echo "  ZERI_TOOLS=$ZERI_TOOLS"
echo "  python=$(which python)"
echo "  ROS_DISTRO=${ROS_DISTRO:-unknown}"
