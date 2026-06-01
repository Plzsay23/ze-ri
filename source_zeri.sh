#!/usr/bin/env bash

set +u

source /opt/ros/jazzy/setup.bash

if [ -f "$HOME/ze-ri/ros2_ws/install/setup.bash" ]; then
  source "$HOME/ze-ri/ros2_ws/install/setup.bash"
fi

export ZERI_ROOT="$HOME/ze-ri"
export ZERI_ROS_WS="$HOME/ze-ri/ros2_ws"
export ZERI_CONFIG_DIR="$HOME/ze-ri/configs"
