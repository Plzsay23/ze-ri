#!/usr/bin/env bash

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export ZERI_ROOT="$SCRIPT_DIR"
export ZERI_TOOLS="$ZERI_ROOT/tools"
export NBYTICS_ROOT="$ZERI_ROOT/NBYtics"

# CUDA 13.0 for Jetson Thor
export CUDA_HOME=/usr/local/cuda
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"

source /opt/ros/jazzy/setup.bash

if [ -f "$ZERI_ROOT/.venv/bin/activate" ]; then
  source "$ZERI_ROOT/.venv/bin/activate"
fi

if [ -f "$NBYTICS_ROOT/install/setup.bash" ]; then
  source "$NBYTICS_ROOT/install/setup.bash"
fi

VENV_SITE="$ZERI_ROOT/.venv/lib/python3.12/site-packages"

NBYTICS_PYTHONPATH="$NBYTICS_ROOT/src/nb_voice_stt:$NBYTICS_ROOT/src/nb_odom:$NBYTICS_ROOT/src/nb_base_bridge:$NBYTICS_ROOT/src"
export PYTHONPATH="$VENV_SITE:$NBYTICS_PYTHONPATH:$ZERI_ROOT/src:${PYTHONPATH:-}"

echo "[Ze-Ri Unified]"
echo "  ZERI_ROOT=$ZERI_ROOT"
echo "  NBYTICS_ROOT=$NBYTICS_ROOT"
echo "  ZERI_TOOLS=$ZERI_TOOLS"
echo "  python=$(which python)"
echo "  ROS_DISTRO=${ROS_DISTRO:-unknown}"
