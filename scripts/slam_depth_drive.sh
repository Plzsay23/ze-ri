#!/usr/bin/env bash

set -e

source "$HOME/ze-ri/source_zeri.sh"

# depth guard가 numpy 등을 쓸 수 있게 venv 경로도 추가
export PYTHONPATH="$HOME/ze-ri/.venv/lib/python3.12/site-packages:$PYTHONPATH"

LOG_DIR="$HOME/ze-ri/logs/slam_depth_drive_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$LOG_DIR"

LIDAR_PORT="${LIDAR_PORT:-/dev/lidar}"
ARDUINO_PORT="${ARDUINO_PORT:-/dev/arduino}"
DEPTH_TOPIC="${DEPTH_TOPIC:-/zeri/vlm/input_depth}"

START_RVIZ="false"
if [ "${1:-}" = "--rviz" ]; then
  START_RVIZ="true"
fi

PIDS=()

cleanup() {
  echo
  echo "[Ze-Ri] stopping..."
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

  echo "[Ze-Ri] stopped"
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
}

echo "[Ze-Ri SLAM Depth Drive]"
echo "  LIDAR_PORT=$LIDAR_PORT"
echo "  ARDUINO_PORT=$ARDUINO_PORT"
echo "  DEPTH_TOPIC=$DEPTH_TOPIC"
echo "  LOG_DIR=$LOG_DIR"
echo "  RVIZ=$START_RVIZ"
echo

# 1. LiDAR driver
start_node "01_lidar" \
  ros2 run ydlidar_ros2_driver ydlidar_ros2_driver_node --ros-args \
    -p port:="$LIDAR_PORT" \
    -p frame_id:=laser_frame_raw \
    -p baudrate:=115200 \
    -p lidar_type:=1 \
    -p device_type:=0 \
    -p isSingleChannel:=true \
    -p intensity:=false

# 2. base_link -> laser_frame_raw TF
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

# 3. /scan -> /scan_front
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

# 4. LiDAR + Depth obstacle avoidance
start_node "04_lidar_depth_guard" \
  ros2 run zeri_lidar lidar_depth_guard_node --ros-args \
    -p input_cmd_topic:=/cmd_vel_raw \
    -p output_cmd_topic:=/cmd_vel \
    -p scan_topic:=/scan_front \
    -p depth_topic:="$DEPTH_TOPIC" \
    -p stop_distance:=0.55 \
    -p clear_distance:=0.85 \
    -p side_min_clearance:=0.45 \
    -p avoid_lateral_speed:=0.20 \
    -p avoid_min_time_sec:=0.7 \
    -p avoid_max_time_sec:=3.0 \
    -p left_is_positive_y:=true \
    -p lidar_weight:=0.70 \
    -p depth_weight:=0.30 \
    -p use_depth:=true \
    -p log_state_change:=true \
    -p log_scores:=true

# 5. Arduino base serial + encoder odom
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

# 6. SLAM toolbox
start_node "06_slam_toolbox" \
  ros2 launch slam_toolbox online_async_launch.py \
    use_sim_time:=false \
    slam_params_file:="$HOME/ze-ri/configs/slam/mapper_params_online_async.yaml"

# 7. Optional RViz
if [ "$START_RVIZ" = "true" ]; then
  start_node "07_rviz2" rviz2
fi

echo
echo "[READY]"
echo "  Camera is NOT started by this script."
echo "  Start camera separately:"
echo "    ~/ze-ri/scripts/camera_rgbd.sh"
echo
echo "  Test forward:"
echo "    ros2 topic pub -r 10 /cmd_vel_raw geometry_msgs/msg/Twist \"{linear: {x: 0.20, y: 0.0, z: 0.0}, angular: {x: 0.0, y: 0.0, z: 0.0}}\""
echo
echo "  Stop:"
echo "    ros2 topic pub --once /cmd_vel_raw geometry_msgs/msg/Twist \"{linear: {x: 0.0, y: 0.0, z: 0.0}, angular: {x: 0.0, y: 0.0, z: 0.0}}\""
echo
echo "  Logs:"
echo "    $LOG_DIR"
echo
echo "Press Ctrl+C to stop all launched nodes."

while true; do
  sleep 1
done
