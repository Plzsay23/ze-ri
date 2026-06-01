#!/usr/bin/env bash

set -e

source "$HOME/ze-ri/source_zeri.sh"

export PYTHONPATH="$HOME/ze-ri/.venv/lib/python3.12/site-packages:$PYTHONPATH"

LOG_DIR="$HOME/ze-ri/logs/voice_follow_drive_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$LOG_DIR"

LIDAR_PORT="${LIDAR_PORT:-/dev/lidar}"
ARDUINO_PORT="${ARDUINO_PORT:-/dev/arduino}"

FRONT_ANGLE_DEG="${FRONT_ANGLE_DEG:-245.0}"
INVERT_TURN="${INVERT_TURN:-false}"

VAD_POLL_HZ="${VAD_POLL_HZ:-40.0}"

LISTEN_SEC="${LISTEN_SEC:-0.8}"
MIN_VOICE_SAMPLES="${MIN_VOICE_SAMPLES:-5}"
MAX_SAMPLE_AGE_SEC="${MAX_SAMPLE_AGE_SEC:-1.2}"

MOVE_BURST_SEC="${MOVE_BURST_SEC:-0.45}"
POST_MOVE_STOP_SEC="${POST_MOVE_STOP_SEC:-0.25}"

ALIGN_ENTER_DEG="${ALIGN_ENTER_DEG:-50.0}"

FORWARD_SPEED="${FORWARD_SPEED:-0.16}"
MIN_TURN_SPEED="${MIN_TURN_SPEED:-0.34}"
MAX_TURN_SPEED="${MAX_TURN_SPEED:-0.42}"
TURN_KP="${TURN_KP:-0.012}"

STOP_DISTANCE="${STOP_DISTANCE:-0.65}"
CLEAR_DISTANCE="${CLEAR_DISTANCE:-0.85}"

START_RVIZ="false"
if [ "${1:-}" = "--rviz" ]; then
  START_RVIZ="true"
fi

PIDS=()

cleanup() {
  echo
  echo "[Ze-Ri Voice Follow] stopping..."

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

  echo "[Ze-Ri Voice Follow] stopped"
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

echo "[Ze-Ri Voice Follow Drive - Listen/Move/Listen]"
echo "  LIDAR_PORT=$LIDAR_PORT"
echo "  ARDUINO_PORT=$ARDUINO_PORT"
echo "  FRONT_ANGLE_DEG=$FRONT_ANGLE_DEG"
echo "  INVERT_TURN=$INVERT_TURN"
echo "  VAD_POLL_HZ=$VAD_POLL_HZ"
echo "  LISTEN_SEC=$LISTEN_SEC"
echo "  MOVE_BURST_SEC=$MOVE_BURST_SEC"
echo "  POST_MOVE_STOP_SEC=$POST_MOVE_STOP_SEC"
echo "  STOP_DISTANCE=$STOP_DISTANCE"
echo "  CLEAR_DISTANCE=$CLEAR_DISTANCE"
echo "  LOG_DIR=$LOG_DIR"
echo "  RVIZ=$START_RVIZ"
echo

require_file "$HOME/ze-ri/ros2_ws/install/zeri_voice/bin/respeaker_vad_doa_node"
require_file "$HOME/ze-ri/ros2_ws/install/zeri_voice/bin/voice_follow_cmd_node"
require_file "$HOME/ze-ri/ros2_ws/install/zeri_voice/bin/voice_stop_guard_node"

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
    -p stop_distance:="$STOP_DISTANCE" \
    -p clear_distance:="$CLEAR_DISTANCE"

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

start_node "07_respeaker_vad_doa" \
  "$HOME/ze-ri/ros2_ws/install/zeri_voice/bin/respeaker_vad_doa_node" --ros-args \
    -p poll_hz:="$VAD_POLL_HZ" \
    -p vad_topic:=/zeri/audio/vad \
    -p doa_topic:=/zeri/audio/doa_deg \
    -p state_topic:=/zeri/audio/state \
    -p log_active_only:=true

start_node "08_voice_follow_cmd" \
  "$HOME/ze-ri/ros2_ws/install/zeri_voice/bin/voice_follow_cmd_node" --ros-args \
    -p vad_topic:=/zeri/audio/vad \
    -p doa_topic:=/zeri/audio/doa_deg \
    -p cmd_topic:=/cmd_vel_raw \
    -p state_topic:=/zeri/audio/follow_state \
    -p front_angle_deg:="$FRONT_ANGLE_DEG" \
    -p invert_turn:="$INVERT_TURN" \
    -p listen_sec:="$LISTEN_SEC" \
    -p min_voice_samples:="$MIN_VOICE_SAMPLES" \
    -p max_sample_age_sec:="$MAX_SAMPLE_AGE_SEC" \
    -p move_burst_sec:="$MOVE_BURST_SEC" \
    -p post_move_stop_sec:="$POST_MOVE_STOP_SEC" \
    -p align_enter_deg:="$ALIGN_ENTER_DEG" \
    -p forward_speed:="$FORWARD_SPEED" \
    -p min_turn_speed:="$MIN_TURN_SPEED" \
    -p max_turn_speed:="$MAX_TURN_SPEED" \
    -p turn_kp:="$TURN_KP"

if [ "$START_RVIZ" = "true" ]; then
  start_node "09_rviz2" rviz2
fi

echo
echo "[READY] Listen-Move-Listen voice-follow drive is running."
echo
echo "States:"
echo "  LISTEN : robot stopped, collecting voice DOA"
echo "  MOVE   : robot moves briefly, DOA ignored"
echo "  PAUSE  : robot stopped, motor noise settles"
echo
echo "Check:"
echo "  ros2 topic echo /zeri/audio/follow_state"
echo "  ros2 topic echo /zeri/voice_stop_guard/state"
echo "  ros2 topic echo /cmd_vel"
echo
echo "Logs:"
echo "  $LOG_DIR"
echo
echo "Press Ctrl+C to stop all launched nodes."

while true; do
  sleep 1
done
