#!/usr/bin/env bash

set +u

source /opt/ros/jazzy/setup.bash

export ZERI_ROOT="$HOME/ze-ri"
export ZERI_ROS_WS="$HOME/ze-ri/ros2_ws"
export ZERI_CONFIG_DIR="$HOME/ze-ri/configs"
export PYTHONPATH="$ZERI_ROOT/src:${PYTHONPATH:-}"

if [ -f "$ZERI_ROS_WS/install/setup.bash" ]; then
  source "$ZERI_ROS_WS/install/setup.bash"
fi
