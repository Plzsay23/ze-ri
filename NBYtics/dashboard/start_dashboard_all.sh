#!/usr/bin/env bash
set -e

source /opt/ros/humble/setup.bash

ROOT="$HOME/NBYtics/dashboard"
cd "$ROOT"

cleanup() {
  echo ""
  echo "[INFO] shutting down all processes..."
  jobs -p | xargs -r kill
}
trap cleanup EXIT INT TERM

echo "[INFO] starting realsense..."
ros2 launch realsense2_camera rs_launch.py > realsense.log 2>&1 &

sleep 3

echo "[INFO] starting rosbridge..."
ros2 run rosbridge_server rosbridge_websocket > rosbridge.log 2>&1 &

sleep 2

echo "[INFO] starting web_video_server..."
ros2 run web_video_server web_video_server > web_video_server.log 2>&1 &

sleep 2

echo "[INFO] starting depth viz..."
python3 "$ROOT/depth_viz_node.py" > depth_viz.log 2>&1 &

sleep 2

echo "[INFO] starting dashboard http server..."
python3 -m http.server 8000
