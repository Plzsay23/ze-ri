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

# 현재 zeri_camera는 ros2 run이 아니라 bin 직접 실행 기준
exec "$HOME/ze-ri/ros2_ws/install/zeri_camera/bin/realsense_rgbd_node" --ros-args \
  -p serial_number:="'$SERIAL'" \
  -p width:=640 \
  -p height:=480 \
  -p fps:=30 \
  -p rgb_topic:="$RGB_TOPIC" \
  -p depth_topic:="$DEPTH_TOPIC" \
  -p align_depth_to_color:=true
