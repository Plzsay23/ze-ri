#!/usr/bin/env bash

set -euo pipefail

ROOT="$HOME/ze-ri"
LOG_DIR="$ROOT/logs/person_follow_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$LOG_DIR"

set +u
source "$ROOT/source_zeri.sh"
set -u
export PYTHONPATH="$ROOT/.venv/lib/python3.12/site-packages:${PYTHONPATH:-}"

# =========================
# User tunables
# =========================

ARDUINO_PORT="${ARDUINO_PORT:-/dev/arduino}"
LIDAR_PORT="${LIDAR_PORT:-/dev/lidar}"

# RealSense / Ze-Ri public camera topics
# 기본 운용 정책: 카메라는 별도 터미널에서 scripts/camera_rgbd.sh 로 먼저 실행한다.
# 이 스크립트는 그 토픽을 받아 사람 추종/안전정지/미션 이벤트만 실행한다.
# 단독 테스트가 필요하면 START_CAMERA=true 로 camera_rgbd.sh까지 같이 띄울 수 있다.
START_CAMERA="${START_CAMERA:-false}"
WAIT_CAMERA_TOPICS="${WAIT_CAMERA_TOPICS:-true}"
REQUIRE_CAMERA_TOPICS="${REQUIRE_CAMERA_TOPICS:-true}"
REQUIRE_POINTCLOUD_TOPIC="${REQUIRE_POINTCLOUD_TOPIC:-false}"
CAMERA_SERIAL="${CAMERA_SERIAL:-944122071303}"
CAMERA_SCRIPT="${CAMERA_SCRIPT:-$ROOT/scripts/camera_rgbd.sh}"
CAMERA_NODE="${CAMERA_NODE:-/camera/camera}"

# camera_rgbd.sh / dashboard / VLM bridge 기준 공용 토픽
RGB_TOPIC="${RGB_TOPIC:-/zeri/vlm/input_rgb}"
DEPTH_TOPIC="${DEPTH_TOPIC:-/zeri/vlm/input_depth}"
POINTS_TOPIC="${POINTS_TOPIC:-/zeri/vlm/pointcloud}"

# camera_person_follow_node는 CameraInfo가 필요할 수 있다.
# camera_rgbd.sh는 내부 RealSense /camera/camera/* 토픽을 유지하므로 camera_info는 이쪽을 사용한다.
CAMERA_INFO_TOPIC="${CAMERA_INFO_TOPIC:-/camera/camera/color/camera_info}"

# Follow / safety topics
CMD_RAW_TOPIC="${CMD_RAW_TOPIC:-/cmd_vel_raw}"
CMD_OUT_TOPIC="${CMD_OUT_TOPIC:-/cmd_vel}"
SCAN_TOPIC="${SCAN_TOPIC:-/scan_front}"
PERSON_STATE_TOPIC="${PERSON_STATE_TOPIC:-/zeri/person_follow/state}"
SAFETY_STATE_TOPIC="${SAFETY_STATE_TOPIC:-/zeri/safety_guard/state}"
MISSION_EVENT_TOPIC="${MISSION_EVENT_TOPIC:-/zeri/mission/event}"
MISSION_EVENT_STABLE_SEC="${MISSION_EVENT_STABLE_SEC:-1.0}"
MISSION_EVENT_COOLDOWN_SEC="${MISSION_EVENT_COOLDOWN_SEC:-30.0}"

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

# Safety behavior for this launcher itself.
# true: child process가 죽으면 전체를 멈춰 반쯤 살아있는 주행 상태를 방지한다.
DIE_ON_PROCESS_EXIT="${DIE_ON_PROCESS_EXIT:-true}"

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

wait_topic_or_exit() {
  local topic="$1"
  local timeout_sec="${2:-15}"
  local required="${3:-true}"

  if wait_topic "$topic" "$timeout_sec"; then
    return 0
  fi

  if [ "$required" = "true" ]; then
    say "ERROR required topic not available: $topic"
    exit 1
  fi

  return 1
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
say "START_CAMERA=$START_CAMERA"
say "CAMERA_SCRIPT=$CAMERA_SCRIPT"
say "WAIT_CAMERA_TOPICS=$WAIT_CAMERA_TOPICS"
say "REQUIRE_CAMERA_TOPICS=$REQUIRE_CAMERA_TOPICS"
say "RGB_TOPIC=$RGB_TOPIC"
say "DEPTH_TOPIC=$DEPTH_TOPIC"
say "POINTS_TOPIC=$POINTS_TOPIC"
say "CAMERA_INFO_TOPIC=$CAMERA_INFO_TOPIC"
say "MISSION_EVENT_TOPIC=$MISSION_EVENT_TOPIC"
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

# =========================
# 1. Camera topic precheck
#    기본값: 이 스크립트에서는 RealSense를 실행하지 않음
# =========================

if [ "$START_CAMERA" = "true" ]; then
  if [ ! -x "$CAMERA_SCRIPT" ]; then
    say "ERROR camera script not found or not executable: $CAMERA_SCRIPT"
    exit 1
  fi

  SERIAL="$CAMERA_SERIAL" \
  RGB_TOPIC="$RGB_TOPIC" \
  DEPTH_TOPIC="$DEPTH_TOPIC" \
  POINTCLOUD_TOPIC="$POINTS_TOPIC" \
  start_node "01_camera_rgbd_pointcloud" \
    bash "$CAMERA_SCRIPT" "$CAMERA_SERIAL"
else
  say "SKIP RealSense start"
  say "  camera must be started separately:"
  say "  SERIAL=$CAMERA_SERIAL RGB_TOPIC=$RGB_TOPIC DEPTH_TOPIC=$DEPTH_TOPIC POINTCLOUD_TOPIC=$POINTS_TOPIC bash $CAMERA_SCRIPT"
fi

if [ "$WAIT_CAMERA_TOPICS" = "true" ]; then
  wait_topic_or_exit "$RGB_TOPIC" 30 "$REQUIRE_CAMERA_TOPICS"
  wait_topic_or_exit "$DEPTH_TOPIC" 30 "$REQUIRE_CAMERA_TOPICS"
  wait_topic_or_exit "$CAMERA_INFO_TOPIC" 10 "$REQUIRE_CAMERA_TOPICS"
fi

# PointCloud는 대시보드 3D 뷰용입니다.
# camera_rgbd.sh가 /zeri/vlm/pointcloud로 relay한다.
# raw RealSense node만 떠 있고 public pointcloud가 없으면 wrapper param을 보정한 뒤 한 번 더 기다린다.
if ! ros2 topic list 2>/dev/null | grep -qx "$POINTS_TOPIC"; then
  if ros2 node list 2>/dev/null | grep -qx "$CAMERA_NODE"; then
    say "PointCloud topic not found. Trying to enable pointcloud on existing camera node: $CAMERA_NODE"
    ros2 param set "$CAMERA_NODE" pointcloud__neon_.stream_filter 2 || true
    ros2 param set "$CAMERA_NODE" pointcloud__neon_.stream_index_filter 0 || true
    ros2 param set "$CAMERA_NODE" pointcloud__neon_.allow_no_texture_points false || true
    ros2 param set "$CAMERA_NODE" pointcloud__neon_.enable true || true
    sleep 1
  else
    say "WARN camera node not found. PointCloud topic unavailable: $POINTS_TOPIC"
  fi
fi

wait_topic_or_exit "$POINTS_TOPIC" 5 "$REQUIRE_POINTCLOUD_TOPIC" || true

say "Current camera topics:"
ros2 topic list | grep -E "zeri/vlm/input|zeri/vlm/pointcloud|camera_info|camera|point|points" || true

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

# =========================
# 5b. Person arrival -> VLM mission event bridge
#     /zeri/person_follow/state -> /zeri/mission/event
# =========================

if [ ! -f "$ROOT/tools/person_arrival_event_bridge.py" ]; then
  say "ERROR missing: $ROOT/tools/person_arrival_event_bridge.py"
  exit 1
fi

start_node "05b_person_arrival_event_bridge" \
  python3 "$ROOT/tools/person_arrival_event_bridge.py" --ros-args \
    -p person_state_topic:="$PERSON_STATE_TOPIC" \
    -p mission_event_topic:="$MISSION_EVENT_TOPIC" \
    -p stop_distance_m:="$STOP_DISTANCE_M" \
    -p stable_sec:="$MISSION_EVENT_STABLE_SEC" \
    -p cooldown_sec:="$MISSION_EVENT_COOLDOWN_SEC" \
    -p source:=person_follow_depth_lidar_drive

wait_topic "$MISSION_EVENT_TOPIC" 5 || true


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
say "  ros2 topic echo $CMD_RAW_TOPIC"
say "  ros2 topic echo $CMD_OUT_TOPIC"
say "  ros2 topic echo $MISSION_EVENT_TOPIC"
say "  ros2 topic hz $RGB_TOPIC"
say "  ros2 topic hz $DEPTH_TOPIC"
say "  ros2 topic hz $POINTS_TOPIC"
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
      if [ "$DIE_ON_PROCESS_EXIT" = "true" ]; then
        say "ERROR stopping stack because a child process exited"
        exit 1
      fi
    fi
  done
done

