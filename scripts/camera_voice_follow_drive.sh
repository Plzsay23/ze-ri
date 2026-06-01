#!/usr/bin/env bash

set -e

source "$HOME/ze-ri/source_zeri.sh"
export PYTHONPATH="$HOME/ze-ri/.venv/lib/python3.12/site-packages:$PYTHONPATH"

LOG_DIR="$HOME/ze-ri/logs/camera_voice_follow_drive_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$LOG_DIR"

LIDAR_PORT="${LIDAR_PORT:-/dev/lidar}"
ARDUINO_PORT="${ARDUINO_PORT:-/dev/arduino}"

VAD_POLL_HZ="${VAD_POLL_HZ:-40.0}"

TARGET_DISTANCE_M="${TARGET_DISTANCE_M:-1.00}"
STOP_DISTANCE_M="${STOP_DISTANCE_M:-0.55}"
CENTER_DEADBAND_NORM="${CENTER_DEADBAND_NORM:-0.16}"

FORWARD_SPEED="${FORWARD_SPEED:-0.16}"
TURN_KP="${TURN_KP:-0.65}"
MIN_TURN_SPEED="${MIN_TURN_SPEED:-0.32}"
MAX_TURN_SPEED="${MAX_TURN_SPEED:-0.45}"
INVERT_TURN="${INVERT_TURN:-false}"

VOICE_HOLD_SEC="${VOICE_HOLD_SEC:-2.0}"

LIDAR_STOP_DISTANCE="${LIDAR_STOP_DISTANCE:-0.45}"
LIDAR_CLEAR_DISTANCE="${LIDAR_CLEAR_DISTANCE:-0.65}"

START_RVIZ="false"
if [ "${1:-}" = "--rviz" ]; then
  START_RVIZ="true"
fi

PIDS=()

cleanup() {
  echo
  echo "[Ze-Ri Camera Voice Follow] stopping..."

  ros2 topic pub --once /cmd_vel_raw geometry_msgs/msg/Twist \
    "{linear: {x: 0.0, y: 0.0, z: 0.0}, angular: {x: 0.0, y: 0.0, z: 0.0}}" >/dev/null 2>&1 || true

  ros2 topic pub --once /cmd_vel geometry_msgs/msg/Twist \
    "{linear: {x: 0.0, y: 0.0, z: 0.0}, angular: {x: 0.0, y: 0.0, z: 0.0}}" >/dev/null 2>&1 || true

  for pid in "${PIDS[@]}"; do
    if kill -0 "$pid" 2>/dev/null; then
      kill "$pid" 2>/dev/null || true
    fi
  done

  sleep 0.5

  for pid in "${PIDS[@]}"; do
    if kill -0 "$pid" 2>/dev/null; then
      kill -9 "$pid" 2>/dev/null || true
    fi
  done

  echo "[Ze-Ri Camera Voice Follow] stopped"
}

trap cleanup EXIT INT TERM

start_node() {
  local name="$1"
  shift

  echo "[START] $name"
  echo "        log: $LOG_DIR/$name.log"

  "$@" > "$LOG_DIR/$name.log" 2>&1 &
  local pid=$!
  PIDS+=("$pid")

  sleep 0.8

  if ! kill -0 "$pid" 2>/dev/null; then
    echo "[ERROR] $name exited early. Check log:"
    echo "        $LOG_DIR/$name.log"
    tail -80 "$LOG_DIR/$name.log" || true
    exit 1
  fi
}

require_file() {
  local path="$1"
  if [ ! -f "$path" ]; then
    echo "[ERROR] missing file: $path"
    exit 1
  fi
}

echo "[Ze-Ri Camera Voice Follow Drive]"
echo "  Camera publisher is NOT started here."
echo "  Start camera separately:"
echo "    ~/ze-ri/scripts/camera_rgbd.sh"
echo
echo "  LIDAR_PORT=$LIDAR_PORT"
echo "  ARDUINO_PORT=$ARDUINO_PORT"
echo "  TARGET_DISTANCE_M=$TARGET_DISTANCE_M"
echo "  STOP_DISTANCE_M=$STOP_DISTANCE_M"
echo "  CENTER_DEADBAND_NORM=$CENTER_DEADBAND_NORM"
echo "  INVERT_TURN=$INVERT_TURN"
echo "  LOG_DIR=$LOG_DIR"
echo "  RVIZ=$START_RVIZ"
echo

require_file "$HOME/ze-ri/ros2_ws/install/zeri_voice/bin/respeaker_vad_doa_node"
require_file "$HOME/ze-ri/ros2_ws/install/zeri_voice/bin/voice_stop_guard_node"
require_file "$HOME/ze-ri/ros2_ws/install/zeri_camera/bin/camera_depth_follow_node"

start_node "01_lidar" \
  ros2 run ydlidar_ros2_driver ydlidar_ros2_driver_node --ros-args \
    -p port:="$LIDAR_PORT" \
    -p frame_id:=laser_frame_raw \
    -p baudrate:=115200 \
    -p lidar_type:=1 \
    -p device_type:=0 \
    -p isSingleChannel:=true \
    -p intensity:=false

start_node "02_lidar_tf" \
  ros2 run tf2_ros static_transform_publisher \
    --x 0.20 \
    --y 0.00 \
    --z 0.10 \
    --roll 0.0 \
    --pitch 0.0 \
    --yaw 3.14159 \
    --frame-id base_link \
    --child-frame-id laser_frame_raw

start_node "03_scan_front_filter" \
  ros2 run zeri_lidar scan_front_filter_node --ros-args \
    -p input_topic:=/scan \
    -p output_topic:=/scan_front \
    -p min_angle_deg:=-90.0 \
    -p max_angle_deg:=90.0 \
    -p lidar_yaw_deg:=180.0 \
    -p min_keep_range:=0.45 \
    -p max_keep_range:=6.0 \
    -p fixed_bins:=720

start_node "04_voice_stop_guard" \
  "$HOME/ze-ri/ros2_ws/install/zeri_voice/bin/voice_stop_guard_node" --ros-args \
    -p input_cmd_topic:=/cmd_vel_raw \
    -p output_cmd_topic:=/cmd_vel \
    -p scan_topic:=/scan_front \
    -p stop_distance:="$LIDAR_STOP_DISTANCE" \
    -p clear_distance:="$LIDAR_CLEAR_DISTANCE"

start_node "05_base_key_odom" \
  ros2 run zeri_base base_key_odom_serial_node --ros-args \
    -p port:="$ARDUINO_PORT" \
    -p baudrate:=115200 \
    -p cmd_topic:=/cmd_vel \
    -p odom_topic:=/odom \
    -p ticks_per_rev:=3464.0 \
    -p wheel_radius:=0.075 \
    -p lx:=0.1575 \
    -p ly:=0.2125 \
    -p log_sent_key:=true \
    -p log_encoder_line:=false

start_node "06_slam_toolbox" \
  ros2 launch slam_toolbox online_async_launch.py \
    use_sim_time:=false \
    slam_params_file:="$HOME/ze-ri/configs/slam/mapper_params_online_async.yaml"

start_node "07_respeaker_vad" \
  "$HOME/ze-ri/ros2_ws/install/zeri_voice/bin/respeaker_vad_doa_node" --ros-args \
    -p poll_hz:="$VAD_POLL_HZ" \
    -p vad_topic:=/zeri/audio/vad \
    -p doa_topic:=/zeri/audio/doa_deg \
    -p state_topic:=/zeri/audio/state \
    -p log_active_only:=true

start_node "08_camera_depth_follow" \
  "$HOME/ze-ri/ros2_ws/install/zeri_camera/bin/camera_depth_follow_node" --ros-args \
    -p depth_topic:=/zeri/vlm/input_depth \
    -p vad_topic:=/zeri/audio/vad \
    -p cmd_topic:=/cmd_vel_raw \
    -p state_topic:=/zeri/camera_follow/state \
    -p use_vad_gate:=true \
    -p voice_hold_sec:="$VOICE_HOLD_SEC" \
    -p target_distance_m:="$TARGET_DISTANCE_M" \
    -p stop_distance_m:="$STOP_DISTANCE_M" \
    -p center_deadband_norm:="$CENTER_DEADBAND_NORM" \
    -p forward_speed:="$FORWARD_SPEED" \
    -p turn_kp:="$TURN_KP" \
    -p min_turn_speed:="$MIN_TURN_SPEED" \
    -p max_turn_speed:="$MAX_TURN_SPEED" \
    -p invert_turn:="$INVERT_TURN"

if [ "$START_RVIZ" = "true" ]; then
  start_node "09_rviz2" rviz2
fi

echo
echo "[READY] Camera-based voice follow drive is running."
echo
echo "Check:"
echo "  ros2 topic echo /zeri/camera_follow/state"
echo "  ros2 topic echo /zeri/voice_stop_guard/state"
echo "  ros2 topic echo /cmd_vel"
echo
echo "Logs:"
echo "  $LOG_DIR"
echo
echo "Press Ctrl+C to stop."

while true; do
  sleep 1
done
