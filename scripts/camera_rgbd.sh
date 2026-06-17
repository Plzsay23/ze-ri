#!/usr/bin/env bash

set -e

source "$HOME/ze-ri/source_zeri.sh"

# pyrealsense2가 .venv에 있으므로 경로 강제 추가
export PYTHONPATH="$HOME/ze-ri/.venv/lib/python3.12/site-packages:$PYTHONPATH"

SERIAL="${1:-944122071303}"

RGB_TOPIC="/zeri/vlm/input_rgb"
DEPTH_TOPIC="/zeri/vlm/input_depth"

echo "[Ze-Ri Camera RGBD]"
echo "  serial=$SERIAL"
echo "  rgb_topic=$RGB_TOPIC"
echo "  depth_topic=$DEPTH_TOPIC"

NODE="$HOME/ze-ri/ros2_ws/install/zeri_camera/lib/zeri_camera/realsense_rgbd_node"
if [ ! -x "$NODE" ]; then
  NODE="$HOME/ze-ri/ros2_ws/install/zeri_camera/bin/realsense_rgbd_node"
fi

if [ ! -x "$NODE" ]; then
  echo "[ERROR] realsense_rgbd_node not found."
  echo "        checked:"
  echo "          $HOME/ze-ri/ros2_ws/install/zeri_camera/lib/zeri_camera/realsense_rgbd_node"
  echo "          $HOME/ze-ri/ros2_ws/install/zeri_camera/bin/realsense_rgbd_node"
  exit 1
fi

exec "$NODE" --ros-args \
  -p serial_number:="'$SERIAL'" \
  -p width:=640 \
  -p height:=480 \
  -p fps:=30 \
  -p rgb_topic:="$RGB_TOPIC" \
  -p depth_topic:="$DEPTH_TOPIC" \
  -p align_depth_to_color:=true
