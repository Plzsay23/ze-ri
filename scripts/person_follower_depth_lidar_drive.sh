#!/usr/bin/env bash

set -eo pipefail

ROOT="$HOME/ze-ri"
LOG_DIR="$ROOT/logs/person_follow_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$LOG_DIR"

source "$ROOT/source_zeri.sh"
export PYTHONPATH="$ROOT/.venv/lib/python3.12/site-packages:$PYTHONPATH"

# =========================
# User tunables
# =========================

ARDUINO_PORT="${ARDUINO_PORT:-/dev/arduino}"
LIDAR_PORT="${LIDAR_PORT:-/dev/lidar}"

# RealSense
# 카메라는 기본적으로 이 스크립트에서 실행하지 않습니다.
# 별도 터미널에서 scripts/realsense_pointcloud.sh 를 먼저 실행하십시오.
WAIT_CAMERA_TOPICS="${WAIT_CAMERA_TOPICS:-true}"
CAMERA_SERIAL="${CAMERA_SERIAL:-_944122071303}"
RGB_TOPIC="${RGB_TOPIC:-/camera/camera/color/image_raw}"
DEPTH_TOPIC="${DEPTH_TOPIC:-/camera/camera/aligned_depth_to_color/image_raw}"
CAMERA_INFO_TOPIC="${CAMERA_INFO_TOPIC:-/camera/camera/color/camera_info}"
POINTS_TOPIC="${POINTS_TOPIC:-/camera/camera/depth/color/points}"

# Follow / safety topics
CMD_RAW_TOPIC="${CMD_RAW_TOPIC:-/cmd_vel_raw}"
CMD_OUT_TOPIC="${CMD_OUT_TOPIC:-/cmd_vel}"
SCAN_TOPIC="${SCAN_TOPIC:-/scan_front}"
PERSON_STATE_TOPIC="${PERSON_STATE_TOPIC:-/zeri/person_follow/state}"
SAFETY_STATE_TOPIC="${SAFETY_STATE_TOPIC:-/zeri/safety_guard/state}"
MISSION_EVENT_TOPIC="${MISSION_EVENT_TOPIC:-/zeri/mission/event}"
ARRIVAL_EVENT_COOLDOWN_SEC="${ARRIVAL_EVENT_COOLDOWN_SEC:-30.0}"
ARRIVAL_EVENT_STABLE_SEC="${ARRIVAL_EVENT_STABLE_SEC:-1.0}"

# Optional visualization / mapping
START_SLAM="${START_SLAM:-true}"
SLAM_PARAMS_FILE="${SLAM_PARAMS_FILE:-$ROOT/configs/slam/mapper_params_online_async.yaml}"
START_OCTOMAP="${START_OCTOMAP:-false}"
START_RVIZ="${START_RVIZ:-false}"

# YOLO follow tuning
YOLO_MODEL="${YOLO_MODEL:-yolov8n.pt}"
YOLO_DEVICE="${YOLO_DEVICE:-cuda:0}"
USE_VAD_GATE="${USE_VAD_GATE:-false}"

TARGET_DISTANCE_M="${TARGET_DISTANCE_M:-1.00}"
STOP_DISTANCE_M="${STOP_DISTANCE_M:-0.75}"
RESUME_DISTANCE_M="${RESUME_DISTANCE_M:-0.95}"

FORWARD_SPEED="${FORWARD_SPEED:-0.12}"
MAX_FORWARD_SPEED="${MAX_FORWARD_SPEED:-0.16}"
TURN_KP="${TURN_KP:-0.35}"
MIN_TURN_SPEED="${MIN_TURN_SPEED:-0.00}"
MAX_TURN_SPEED="${MAX_TURN_SPEED:-0.32}"
CENTER_DEADBAND_NORM="${CENTER_DEADBAND_NORM:-0.18}"

# Safety tuning
FRONT_STOP_M="${FRONT_STOP_M:-0.55}"
FRONT_SLOW_M="${FRONT_SLOW_M:-0.95}"
FRONT_EMERGENCY_M="${FRONT_EMERGENCY_M:-0.38}"
STRAFE_SPEED="${STRAFE_SPEED:-0.10}"
STRAFE_CLEAR_M="${STRAFE_CLEAR_M:-0.75}"
SIDE_STOP_M="${SIDE_STOP_M:-0.45}"

# Base velocity limits
MAX_LINEAR_X="${MAX_LINEAR_X:-0.18}"
MAX_LINEAR_Y="${MAX_LINEAR_Y:-0.14}"
MAX_ANGULAR_Z="${MAX_ANGULAR_Z:-0.45}"

PIDS=()

say() {
  echo "[$(date +%H:%M:%S)] $*"
}

start_node() {
  local name="$1"
  shift

  say "START $name"
  echo "[CMD] $*" > "$LOG_DIR/${name}.log"

  "$@" >> "$LOG_DIR/${name}.log" 2>&1 &
  local pid=$!
  PIDS+=("$pid")

  echo "$pid" > "$LOG_DIR/${name}.pid"
  say "  pid=$pid log=$LOG_DIR/${name}.log"
}

stop_all() {
  echo
  say "STOP requested"

  # 바퀴 정지 명령 먼저 발행
  if command -v ros2 >/dev/null 2>&1; then
    timeout 2 ros2 topic pub --once "$CMD_OUT_TOPIC" geometry_msgs/msg/Twist \
      "{linear: {x: 0.0, y: 0.0, z: 0.0}, angular: {x: 0.0, y: 0.0, z: 0.0}}" \
      >/dev/null 2>&1 || true
  fi

  for pid in "${PIDS[@]}"; do
    if kill -0 "$pid" 2>/dev/null; then
      kill "$pid" 2>/dev/null || true
    fi
  done

  sleep 0.7

  for pid in "${PIDS[@]}"; do
    if kill -0 "$pid" 2>/dev/null; then
      kill -9 "$pid" 2>/dev/null || true
    fi
  done

  say "stopped"
  say "logs: $LOG_DIR"
}

trap stop_all EXIT INT TERM

wait_topic() {
  local topic="$1"
  local timeout_sec="${2:-15}"

  say "WAIT topic $topic"

  local start
  start=$(date +%s)

  while true; do
    if ros2 topic list 2>/dev/null | grep -qx "$topic"; then
      say "  OK $topic"
      return 0
    fi

    local now
    now=$(date +%s)
    if [ $((now - start)) -ge "$timeout_sec" ]; then
      say "  WARN topic not found within ${timeout_sec}s: $topic"
      return 1
    fi

    sleep 0.5
  done
}

wait_node_bin() {
  local path="$1"
  if [ ! -x "$path" ]; then
    say "ERROR executable not found or not executable: $path"
    exit 1
  fi
}

say "============================================================"
say "Ze-Ri person follow + LiDAR/Depth safety drive"
say "LOG_DIR=$LOG_DIR"
say "ARDUINO_PORT=$ARDUINO_PORT"
say "LIDAR_PORT=$LIDAR_PORT"
say "WAIT_CAMERA_TOPICS=$WAIT_CAMERA_TOPICS"
say "RGB_TOPIC=$RGB_TOPIC"
say "DEPTH_TOPIC=$DEPTH_TOPIC"
say "POINTS_TOPIC=$POINTS_TOPIC"
say "START_SLAM=$START_SLAM"
say "SLAM_PARAMS_FILE=$SLAM_PARAMS_FILE"
say "START_OCTOMAP=$START_OCTOMAP"
say "START_RVIZ=$START_RVIZ"
say "============================================================"

# =========================
# Preflight
# =========================

wait_node_bin "$ROOT/ros2_ws/install/zeri_base/bin/base_key_odom_serial_node"
wait_node_bin "$ROOT/ros2_ws/install/zeri_camera/bin/camera_person_follow_node"

if [ ! -f "$ROOT/tools/depth_lidar_safety_guard.py" ]; then
  say "ERROR missing: $ROOT/tools/depth_lidar_safety_guard.py"
  say "먼저 depth_lidar_safety_guard.py 파일을 만들어야 합니다."
  exit 1
fi

if [ ! -f "$ROOT/tools/person_arrival_event_bridge.py" ]; then
  say "ERROR missing: $ROOT/tools/person_arrival_event_bridge.py"
  say "먼저 person_arrival_event_bridge.py 파일을 만들어야 합니다."
  exit 1
fi

# =========================
# 1. Camera topic precheck
#    기본값: 이 스크립트에서는 RealSense를 실행하지 않음
# =========================

say "Camera is managed by a separate terminal. This script will not start or reconfigure RealSense."
say "  expected RGB:         $RGB_TOPIC"
say "  expected Depth:       $DEPTH_TOPIC"
say "  expected CameraInfo:  $CAMERA_INFO_TOPIC"
say "  expected PointCloud:  $POINTS_TOPIC"

if [ "$WAIT_CAMERA_TOPICS" = "true" ]; then
  wait_topic "$RGB_TOPIC" 30 || true
  wait_topic "$DEPTH_TOPIC" 30 || true
  wait_topic "$CAMERA_INFO_TOPIC" 10 || true
fi

say "Current camera topics:"
ros2 topic list | grep -E "camera|point|points" || true

# =========================
# 2. LiDAR
# =========================

start_node "02_lidar" \
  ros2 run ydlidar_ros2_driver ydlidar_ros2_driver_node --ros-args \
    -p port:="$LIDAR_PORT" \
    -p frame_id:=laser_frame_raw \
    -p baudrate:=115200 \
    -p lidar_type:=1 \
    -p device_type:=0 \
    -p isSingleChannel:=true \
    -p intensity:=false

wait_topic "/scan" 15 || true

# =========================
# 3. LiDAR TF
# =========================

start_node "03_lidar_tf" \
  ros2 run tf2_ros static_transform_publisher \
    --x 0.20 \
    --y 0.00 \
    --z 0.10 \
    --roll 0.0 \
    --pitch 0.0 \
    --yaw 3.14159 \
    --frame-id base_link \
    --child-frame-id laser_frame_raw

sleep 0.5

# =========================
# 4. scan front filter
# =========================

start_node "04_scan_front_filter" \
  ros2 run zeri_lidar scan_front_filter_node --ros-args \
    -p input_topic:=/scan \
    -p output_topic:="$SCAN_TOPIC" \
    -p min_angle_deg:=-90.0 \
    -p max_angle_deg:=90.0 \
    -p lidar_yaw_deg:=180.0 \
    -p min_keep_range:=0.45 \
    -p max_keep_range:=6.0 \
    -p fixed_bins:=720

wait_topic "$SCAN_TOPIC" 10 || true

# =========================
# 5. YOLO person follow
#    반드시 /cmd_vel_raw 로만 출력
# =========================

start_node "05_person_follow" \
  "$ROOT/ros2_ws/install/zeri_camera/bin/camera_person_follow_node" --ros-args \
    -p rgb_topic:="$RGB_TOPIC" \
    -p depth_topic:="$DEPTH_TOPIC" \
    -p camera_info_topic:="$CAMERA_INFO_TOPIC" \
    -p cmd_topic:="$CMD_RAW_TOPIC" \
    -p state_topic:="$PERSON_STATE_TOPIC" \
    -p model_path:="$YOLO_MODEL" \
    -p device:="$YOLO_DEVICE" \
    -p imgsz:=320 \
    -p conf_thres:=0.45 \
    -p infer_hz:=6.0 \
    -p use_vad_gate:="$USE_VAD_GATE" \
    -p target_distance_m:="$TARGET_DISTANCE_M" \
    -p stop_distance_m:="$STOP_DISTANCE_M" \
    -p resume_distance_m:="$RESUME_DISTANCE_M" \
    -p forward_speed:="$FORWARD_SPEED" \
    -p max_forward_speed:="$MAX_FORWARD_SPEED" \
    -p turn_kp:="$TURN_KP" \
    -p min_turn_speed:="$MIN_TURN_SPEED" \
    -p max_turn_speed:="$MAX_TURN_SPEED" \
    -p center_deadband_norm:="$CENTER_DEADBAND_NORM" \
    -p output_topic:="$CMD_RAW_TOPIC"

wait_topic "$CMD_RAW_TOPIC" 10 || true
wait_topic "$PERSON_STATE_TOPIC" 10 || true

# =========================
# 5b. Person arrival -> VLM mission event bridge
#     /zeri/person_follow/state -> /zeri/mission/event
# =========================

start_node "05b_person_arrival_event_bridge" \
  python3 "$ROOT/tools/person_arrival_event_bridge.py" --ros-args \
    -p person_state_topic:="$PERSON_STATE_TOPIC" \
    -p mission_event_topic:="$MISSION_EVENT_TOPIC" \
    -p stop_distance_m:="$STOP_DISTANCE_M" \
    -p resume_distance_m:="$RESUME_DISTANCE_M" \
    -p stable_sec:="$ARRIVAL_EVENT_STABLE_SEC" \
    -p cooldown_sec:="$ARRIVAL_EVENT_COOLDOWN_SEC" \
    -p selected_person_id:=person_follow_target \
    -p source:=person_follow_depth_lidar_drive

wait_topic "$MISSION_EVENT_TOPIC" 10 || true

# =========================
# 6. LiDAR + Depth safety guard
#    /cmd_vel_raw -> /cmd_vel
# =========================

start_node "06_depth_lidar_safety_guard" \
  python3 "$ROOT/tools/depth_lidar_safety_guard.py" --ros-args \
    -p cmd_raw_topic:="$CMD_RAW_TOPIC" \
    -p cmd_out_topic:="$CMD_OUT_TOPIC" \
    -p scan_topic:="$SCAN_TOPIC" \
    -p depth_topic:="$DEPTH_TOPIC" \
    -p person_state_topic:="$PERSON_STATE_TOPIC" \
    -p state_topic:="$SAFETY_STATE_TOPIC" \
    -p publish_hz:=20.0 \
    -p cmd_timeout_sec:=0.45 \
    -p raw_hold_sec:=0.25 \
    -p linear_accel_mps2:=0.25 \
    -p strafe_accel_mps2:=0.22 \
    -p angular_accel_rps2:=0.75 \
    -p max_linear_x:="$MAX_LINEAR_X" \
    -p max_linear_y:="$MAX_LINEAR_Y" \
    -p max_angular_z:="$MAX_ANGULAR_Z" \
    -p front_stop_m:="$FRONT_STOP_M" \
    -p front_slow_m:="$FRONT_SLOW_M" \
    -p front_emergency_m:="$FRONT_EMERGENCY_M" \
    -p enable_strafe_avoidance:=true \
    -p strafe_speed:="$STRAFE_SPEED" \
    -p strafe_clear_m:="$STRAFE_CLEAR_M" \
    -p side_stop_m:="$SIDE_STOP_M" \
    -p person_stop_distance_m:="$STOP_DISTANCE_M" \
    -p person_resume_distance_m:="$RESUME_DISTANCE_M" \
    -p person_emergency_distance_m:=0.45

wait_topic "$CMD_OUT_TOPIC" 10 || true

# =========================
# 7. Arduino base velocity node
#    Arduino에는 mobile_base_velocity_smooth.ino 필요
# =========================

start_node "07_base_velocity" \
  "$ROOT/ros2_ws/install/zeri_base/bin/base_key_odom_serial_node" --ros-args \
    -p port:="$ARDUINO_PORT" \
    -p baudrate:=115200 \
    -p cmd_topic:="$CMD_OUT_TOPIC" \
    -p odom_topic:=/odom \
    -p ticks_per_rev:=3464.0 \
    -p wheel_radius:=0.075 \
    -p lx:=0.1575 \
    -p ly:=0.2125 \
    -p send_hz:=20.0 \
    -p max_linear_x:=0.25 \
    -p max_linear_y:=0.22 \
    -p max_angular_z:=0.70 \
    -p enable_strafe:=true \
    -p arduino_pwm:=60 \
    -p set_pwm_on_start:=true \
    -p log_sent_command:=true \
    -p log_all_commands:=false \
    -p log_encoder_line:=false

# =========================
# 8. SLAM toolbox 2D map
#    /scan_front + /odom -> /map
# =========================

if [ "$START_SLAM" = "true" ]; then
  if [ ! -f "$SLAM_PARAMS_FILE" ]; then
    say "ERROR missing SLAM params file: $SLAM_PARAMS_FILE"
    exit 1
  fi

  start_node "08_slam_toolbox" \
    ros2 launch slam_toolbox online_async_launch.py \
      use_sim_time:=false \
      slam_params_file:="$SLAM_PARAMS_FILE"

  wait_topic "/map" 25 || true
fi

# =========================
# Optional OctoMap
# =========================

if [ "$START_OCTOMAP" = "true" ]; then
  start_node "09_octomap_camera_frame" \
    ros2 run octomap_server octomap_server_node --ros-args \
      -r cloud_in:="$POINTS_TOPIC" \
      -p frame_id:=camera_depth_optical_frame \
      -p resolution:=0.08 \
      -p sensor_model.max_range:=3.5 \
      -p pointcloud_min_z:=0.20 \
      -p pointcloud_max_z:=3.0 \
      -p occupancy_min_z:=0.20 \
      -p occupancy_max_z:=3.0
fi

# =========================
# Optional RViz
# =========================

if [ "$START_RVIZ" = "true" ]; then
  start_node "10_rviz" rviz2
fi

say "============================================================"
say "READY"
say ""
say "Monitor:"
say "  ros2 topic echo $SAFETY_STATE_TOPIC"
say "  ros2 topic echo $PERSON_STATE_TOPIC"
say "  ros2 topic echo $MISSION_EVENT_TOPIC"
say "  ros2 topic echo $CMD_RAW_TOPIC"
say "  ros2 topic echo $CMD_OUT_TOPIC"
say "  ros2 topic hz /map"
say ""
say "Logs:"
say "  $LOG_DIR"
say ""
say "Stop:"
say "  Ctrl+C"
say "============================================================"

while true; do
  sleep 1

  # 죽은 프로세스 감지
  for pid in "${PIDS[@]}"; do
    if ! kill -0 "$pid" 2>/dev/null; then
      say "WARN process died: pid=$pid"
    fi
  done
done

