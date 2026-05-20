#!/usr/bin/env bash
set -u

# ============================================================
# Temporary Ze-Ri Mapping / Autonomy Stack Launcher
# - LiDAR
# - LiDAR TF
# - /scan_front filter
# - Encoder Odom
# - SLAM Toolbox
#
# Stop: Ctrl+C
# Logs: ~/ze-ri/logs/tmp_mapping_stack/
# ============================================================

ZERI_ROOT="${ZERI_ROOT:-$HOME/ze-ri}"
SOURCE_FILE="$ZERI_ROOT/source_zeri_vlm.sh"
LOG_DIR="$ZERI_ROOT/logs/tmp_mapping_stack"

mkdir -p "$LOG_DIR"

if [ ! -f "$SOURCE_FILE" ]; then
  echo "[ERROR] source file not found: $SOURCE_FILE"
  exit 1
fi

source "$SOURCE_FILE"

if [ -z "${NBYTICS_ROOT:-}" ]; then
  echo "[WARN] NBYTICS_ROOT is not set after sourcing $SOURCE_FILE"
  echo "[WARN] fallback: NBYTICS_ROOT=$ZERI_ROOT"
  export NBYTICS_ROOT="$ZERI_ROOT"
fi

PIDS=()
NAMES=()

start_proc() {
  local name="$1"
  local cmd="$2"
  local log_file="$LOG_DIR/${name}.log"

  echo ""
  echo "============================================================"
  echo "[START] $name"
  echo "[LOG]   $log_file"
  echo "============================================================"

  # setsid로 별도 process group 생성
  # Ctrl+C 시 자식 프로세스까지 같이 종료하기 쉽게 하기 위함
  setsid bash -lc "
    cd '$ZERI_ROOT'
    source '$SOURCE_FILE'
    exec $cmd
  " > "$log_file" 2>&1 &

  local pid=$!
  PIDS+=("$pid")
  NAMES+=("$name")

  echo "[PID]   $pid"
  sleep 1.0
}

cleanup() {
  echo ""
  echo "============================================================"
  echo "[STOP] stopping temporary mapping stack..."
  echo "============================================================"

  for i in "${!PIDS[@]}"; do
    local pid="${PIDS[$i]}"
    local name="${NAMES[$i]}"

    if kill -0 "$pid" 2>/dev/null; then
      echo "[STOP] $name pid=$pid"
      # process group 전체에 SIGINT
      kill -INT "-$pid" 2>/dev/null || kill -INT "$pid" 2>/dev/null || true
    fi
  done

  sleep 2

  for i in "${!PIDS[@]}"; do
    local pid="${PIDS[$i]}"
    local name="${NAMES[$i]}"

    if kill -0 "$pid" 2>/dev/null; then
      echo "[KILL] $name pid=$pid"
      kill -TERM "-$pid" 2>/dev/null || kill -TERM "$pid" 2>/dev/null || true
    fi
  done

  echo "[DONE] stopped."
}

trap cleanup INT TERM EXIT

echo ""
echo "============================================================"
echo " Ze-Ri temporary mapping stack"
echo " ZERI_ROOT=$ZERI_ROOT"
echo " NBYTICS_ROOT=$NBYTICS_ROOT"
echo " LOG_DIR=$LOG_DIR"
echo "============================================================"

# 1. LiDAR
start_proc "01_lidar" \
"ros2 run ydlidar_ros2_driver ydlidar_ros2_driver_node --ros-args \
  -p port:=/dev/lidar \
  -p frame_id:=laser_frame_raw \
  -p baudrate:=115200 \
  -p lidar_type:=1 \
  -p device_type:=0 \
  -p isSingleChannel:=true \
  -p intensity:=false"

# 2. LiDAR TF
start_proc "02_lidar_tf" \
"ros2 run tf2_ros static_transform_publisher \
  --x 0.20 \
  --y 0.00 \
  --z 0.10 \
  --roll 0 \
  --pitch 0 \
  --yaw 3.14159 \
  --frame-id base_link \
  --child-frame-id laser_frame_raw"

# 3. /scan_front filter
start_proc "03_scan_front_filter" \
"python \$NBYTICS_ROOT/tools/scan_front_filter.py \
  --ros-args \
  -p input_topic:=/scan \
  -p output_topic:=/scan_front \
  -p min_angle_deg:=-90.0 \
  -p max_angle_deg:=90.0 \
  -p lidar_yaw_deg:=180.0 \
  -p min_keep_range:=0.45 \
  -p max_keep_range:=6.0 \
  -p fixed_bins:=720"

# 4. Encoder Odom
start_proc "04_encoder_odom" \
"ros2 run nb_odom encoder_odom_node --ros-args \
  -p port:=/dev/arduino \
  -p baudrate:=115200 \
  -p ticks_per_rev:=3464.0 \
  -p wheel_radius:=0.075 \
  -p lx:=0.1575 \
  -p ly:=0.2125"

# 5. SLAM Toolbox
start_proc "05_slam_toolbox" \
"ros2 launch slam_toolbox online_async_launch.py \
  use_sim_time:=false \
  slam_params_file:=\$NBYTICS_ROOT/slam_config/mapper_params_online_async.yaml"

echo ""
echo "============================================================"
echo "[RUNNING] all mapping processes started."
echo ""
echo "Check topics:"
echo "  ros2 topic list | grep -E 'scan|odom|map'"
echo "  ros2 topic hz /scan"
echo "  ros2 topic hz /scan_front"
echo "  ros2 topic hz /odom"
echo "  ros2 topic echo /map --once"
echo ""
echo "Logs:"
echo "  tail -f $LOG_DIR/01_lidar.log"
echo "  tail -f $LOG_DIR/05_slam_toolbox.log"
echo ""
echo "Stop:"
echo "  Ctrl+C"
echo "============================================================"

# 스크립트가 바로 끝나지 않게 유지
# 어떤 프로세스 하나라도 죽으면 전체 정리
while true; do
  for i in "${!PIDS[@]}"; do
    pid="${PIDS[$i]}"
    name="${NAMES[$i]}"

    if ! kill -0 "$pid" 2>/dev/null; then
      echo ""
      echo "[ERROR] process exited: $name pid=$pid"
      echo "[ERROR] check log: $LOG_DIR/${name}.log"
      exit 1
    fi
  done

  sleep 2
done