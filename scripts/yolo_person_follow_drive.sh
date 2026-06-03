#!/usr/bin/env bash

set -e

source "$HOME/ze-ri/source_zeri.sh"
export PYTHONPATH="$HOME/ze-ri/.venv/lib/python3.12/site-packages:$PYTHONPATH"

LOG_DIR="$HOME/ze-ri/logs/yolo_person_follow_drive_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$LOG_DIR"
SLAM_PARAMS_FILE="$LOG_DIR/mapper_params_online_async.yaml"

LIDAR_PORT="${LIDAR_PORT:-/dev/lidar}"
ARDUINO_PORT="${ARDUINO_PORT:-/dev/arduino}"

LIDAR_TF_X="${LIDAR_TF_X:-0.20}"
LIDAR_TF_Y="${LIDAR_TF_Y:-0.00}"
LIDAR_TF_Z="${LIDAR_TF_Z:-0.10}"
LIDAR_TF_ROLL="${LIDAR_TF_ROLL:-0.0}"
LIDAR_TF_PITCH="${LIDAR_TF_PITCH:-0.0}"
LIDAR_TF_YAW="${LIDAR_TF_YAW:-3.14159}"
USE_SCAN_FRONT_FILTER="${USE_SCAN_FRONT_FILTER:-true}"
SCAN_FILTER_LIDAR_YAW_DEG="${SCAN_FILTER_LIDAR_YAW_DEG:-180.0}"
SCAN_FRONT_MIN_ANGLE_DEG="${SCAN_FRONT_MIN_ANGLE_DEG:--60.0}"
SCAN_FRONT_MAX_ANGLE_DEG="${SCAN_FRONT_MAX_ANGLE_DEG:-60.0}"
SLAM_SCAN_TOPIC="${SLAM_SCAN_TOPIC:-/scan_front}"
SAFETY_SCAN_TOPIC="${SAFETY_SCAN_TOPIC:-/scan_front}"
if [ "$USE_SCAN_FRONT_FILTER" != "true" ]; then
  SLAM_SCAN_TOPIC="/scan"
  SAFETY_SCAN_TOPIC="/scan"
fi

MODEL_PATH="${MODEL_PATH:-yolov8n.pt}"
YOLO_DEVICE="${YOLO_DEVICE:-cuda:0}"
YOLO_IMGSZ="${YOLO_IMGSZ:-320}"
YOLO_CONF="${YOLO_CONF:-0.45}"
YOLO_HZ="${YOLO_HZ:-6.0}"

RGB_TOPIC="${RGB_TOPIC:-/zeri/vlm/input_rgb}"
DEPTH_TOPIC="${DEPTH_TOPIC:-/zeri/vlm/input_depth}"
CAMERA_INFO_TOPIC="${CAMERA_INFO_TOPIC:-/zeri/camera/color/camera_info}"
CAMERA_FRAME_ID="${CAMERA_FRAME_ID:-top_camera_color_optical_frame}"

# Camera pose relative to base_link. Tune these to the real mount.
CAMERA_X="${CAMERA_X:-0.08}"
CAMERA_Y="${CAMERA_Y:-0.00}"
CAMERA_Z="${CAMERA_Z:-0.51}"
CAMERA_ROLL="${CAMERA_ROLL:--1.5708}"
CAMERA_PITCH="${CAMERA_PITCH:-0.0}"
CAMERA_YAW="${CAMERA_YAW:--1.5708}"

USE_VAD_GATE="${USE_VAD_GATE:-true}"
MARKER_USE_VAD_GATE="${MARKER_USE_VAD_GATE:-false}"
START_PERSON_FOLLOW="${START_PERSON_FOLLOW:-true}"
BASE_DRIVER="${BASE_DRIVER:-key}"
PERSON_MARKERS_FILE="${PERSON_MARKERS_FILE:-$HOME/ze-ri/data/person_markers.json}"
MAX_PERSON_MARKERS="${MAX_PERSON_MARKERS:-50}"
UPDATE_PERSON_MARKERS="${UPDATE_PERSON_MARKERS:-true}"
PERSON_MARKER_UPDATE_ALPHA="${PERSON_MARKER_UPDATE_ALPHA:-0.35}"
PERSON_MARKER_UPDATE_JUMP_M="${PERSON_MARKER_UPDATE_JUMP_M:-1.20}"
START_DEPTH_POINTCLOUD="${START_DEPTH_POINTCLOUD:-true}"
POINTCLOUD_TOPIC="${POINTCLOUD_TOPIC:-/zeri/vlm/points}"
POINTCLOUD_HZ="${POINTCLOUD_HZ:-5.0}"
POINTCLOUD_STRIDE="${POINTCLOUD_STRIDE:-4}"
BASE_KEY_HZ="${BASE_KEY_HZ:-20.0}"
BASE_TURN_MIX_GAIN="${BASE_TURN_MIX_GAIN:-1.25}"
BASE_TURN_MIX_MIN_DUTY="${BASE_TURN_MIX_MIN_DUTY:-0.22}"
BASE_TURN_MIX_MAX_DUTY="${BASE_TURN_MIX_MAX_DUTY:-0.55}"
BASE_SEND_HZ="${BASE_SEND_HZ:-20.0}"
BASE_MAX_LINEAR_X="${BASE_MAX_LINEAR_X:-0.25}"
BASE_MAX_LINEAR_Y="${BASE_MAX_LINEAR_Y:-0.25}"
BASE_MAX_ANGULAR_Z="${BASE_MAX_ANGULAR_Z:-0.70}"
BASE_ARDUINO_PWM="${BASE_ARDUINO_PWM:-60}"
VAD_POLL_HZ="${VAD_POLL_HZ:-40.0}"
VOICE_HOLD_SEC="${VOICE_HOLD_SEC:-3.0}"

TARGET_DISTANCE_M="${TARGET_DISTANCE_M:-1.00}"
DISTANCE_DEADBAND_M="${DISTANCE_DEADBAND_M:-0.18}"
TOO_CLOSE_M="${TOO_CLOSE_M:-0.55}"

CENTER_DEADBAND_NORM="${CENTER_DEADBAND_NORM:-0.12}"
FORWARD_MAX_ERR_NORM="${FORWARD_MAX_ERR_NORM:-0.65}"

FORWARD_SPEED="${FORWARD_SPEED:-0.16}"
TURN_KP="${TURN_KP:-0.70}"
MIN_TURN_SPEED="${MIN_TURN_SPEED:-0.30}"
MAX_TURN_SPEED="${MAX_TURN_SPEED:-0.45}"
INVERT_TURN="${INVERT_TURN:-false}"

LIDAR_STOP_DISTANCE="${LIDAR_STOP_DISTANCE:-0.45}"
LIDAR_CLEAR_DISTANCE="${LIDAR_CLEAR_DISTANCE:-0.65}"

START_RVIZ="false"
if [ "${1:-}" = "--rviz" ]; then
  START_RVIZ="true"
fi

PIDS=()

cleanup() {
  echo
  echo "[Ze-Ri YOLO Person Follow] stopping..."

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

  echo "[Ze-Ri YOLO Person Follow] stopped"
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
    tail -100 "$LOG_DIR/$name.log" || true
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

echo "[Ze-Ri YOLO Person Follow Drive]"
echo "  Camera publisher is NOT started here."
echo "  Start camera separately:"
echo "    ~/ze-ri/scripts/camera_rgbd.sh"
echo
echo "  MODEL_PATH=$MODEL_PATH"
echo "  YOLO_DEVICE=$YOLO_DEVICE"
echo "  YOLO_IMGSZ=$YOLO_IMGSZ"
echo "  LIDAR_TF_YAW=$LIDAR_TF_YAW"
echo "  USE_SCAN_FRONT_FILTER=$USE_SCAN_FRONT_FILTER"
echo "  SCAN_FILTER_LIDAR_YAW_DEG=$SCAN_FILTER_LIDAR_YAW_DEG"
echo "  SCAN_FRONT_MIN_ANGLE_DEG=$SCAN_FRONT_MIN_ANGLE_DEG"
echo "  SCAN_FRONT_MAX_ANGLE_DEG=$SCAN_FRONT_MAX_ANGLE_DEG"
echo "  SLAM_SCAN_TOPIC=$SLAM_SCAN_TOPIC"
echo "  SAFETY_SCAN_TOPIC=$SAFETY_SCAN_TOPIC"
echo "  RGB_TOPIC=$RGB_TOPIC"
echo "  DEPTH_TOPIC=$DEPTH_TOPIC"
echo "  CAMERA_INFO_TOPIC=$CAMERA_INFO_TOPIC"
echo "  CAMERA_FRAME_ID=$CAMERA_FRAME_ID"
echo "  USE_VAD_GATE=$USE_VAD_GATE"
echo "  MARKER_USE_VAD_GATE=$MARKER_USE_VAD_GATE"
echo "  START_PERSON_FOLLOW=$START_PERSON_FOLLOW"
echo "  BASE_DRIVER=$BASE_DRIVER"
echo "  PERSON_MARKERS_FILE=$PERSON_MARKERS_FILE"
echo "  MAX_PERSON_MARKERS=$MAX_PERSON_MARKERS"
echo "  UPDATE_PERSON_MARKERS=$UPDATE_PERSON_MARKERS"
echo "  PERSON_MARKER_UPDATE_ALPHA=$PERSON_MARKER_UPDATE_ALPHA"
echo "  START_DEPTH_POINTCLOUD=$START_DEPTH_POINTCLOUD"
echo "  POINTCLOUD_TOPIC=$POINTCLOUD_TOPIC"
echo "  BASE_KEY_HZ=$BASE_KEY_HZ"
echo "  TARGET_DISTANCE_M=$TARGET_DISTANCE_M"
echo "  CENTER_DEADBAND_NORM=$CENTER_DEADBAND_NORM"
echo "  LOG_DIR=$LOG_DIR"
echo "  RVIZ=$START_RVIZ"
echo

require_file "$HOME/ze-ri/ros2_ws/install/zeri_voice/bin/respeaker_vad_doa_node"
require_file "$HOME/ze-ri/ros2_ws/install/zeri_voice/bin/voice_stop_guard_node"
require_file "$HOME/ze-ri/ros2_ws/install/zeri_camera/bin/camera_person_follow_node"
require_file "$HOME/ze-ri/ros2_ws/install/zeri_camera/bin/person_map_marker_node"

sed "s|scan_topic: .*|scan_topic: $SLAM_SCAN_TOPIC|" \
  "$HOME/ze-ri/configs/slam/mapper_params_online_async.yaml" > "$SLAM_PARAMS_FILE"

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
    --x "$LIDAR_TF_X" \
    --y "$LIDAR_TF_Y" \
    --z "$LIDAR_TF_Z" \
    --roll "$LIDAR_TF_ROLL" \
    --pitch "$LIDAR_TF_PITCH" \
    --yaw "$LIDAR_TF_YAW" \
    --frame-id base_link \
    --child-frame-id laser_frame_raw

if [ "$USE_SCAN_FRONT_FILTER" = "true" ]; then
  start_node "03_scan_front_filter" \
    ros2 run zeri_lidar scan_front_filter_node --ros-args \
      -p input_topic:=/scan \
      -p output_topic:="$SAFETY_SCAN_TOPIC" \
      -p output_frame:=base_link \
      -p min_angle_deg:="$SCAN_FRONT_MIN_ANGLE_DEG" \
      -p max_angle_deg:="$SCAN_FRONT_MAX_ANGLE_DEG" \
      -p lidar_yaw_deg:="$SCAN_FILTER_LIDAR_YAW_DEG" \
      -p min_keep_range:=0.45 \
      -p max_keep_range:=6.0 \
      -p fixed_bins:=720
else
  echo "[SKIP] 03_scan_front_filter"
  echo "       using raw /scan for safety guard"
fi

start_node "04_voice_stop_guard" \
  "$HOME/ze-ri/ros2_ws/install/zeri_voice/bin/voice_stop_guard_node" --ros-args \
    -p input_cmd_topic:=/cmd_vel_raw \
    -p output_cmd_topic:=/cmd_vel \
    -p scan_topic:="$SAFETY_SCAN_TOPIC" \
    -p stop_distance:="$LIDAR_STOP_DISTANCE" \
    -p clear_distance:="$LIDAR_CLEAR_DISTANCE"

if [ "$BASE_DRIVER" = "velocity" ]; then
  start_node "05_base_velocity_odom" \
    env PYTHONPATH="$HOME/ze-ri/ros2_ws/install/zeri_base/lib/python3.12/site-packages:$PYTHONPATH" \
    "$HOME/ze-ri/.venv/bin/python" -m zeri_base.base_velocity_odom_serial_node --ros-args \
      -p port:="$ARDUINO_PORT" \
      -p baudrate:=115200 \
      -p cmd_topic:=/cmd_vel \
      -p odom_topic:=/odom \
      -p ticks_per_rev:=3464.0 \
      -p wheel_radius:=0.075 \
      -p lx:=0.1575 \
      -p ly:=0.2125 \
      -p send_hz:="$BASE_SEND_HZ" \
      -p max_linear_x:="$BASE_MAX_LINEAR_X" \
      -p max_linear_y:="$BASE_MAX_LINEAR_Y" \
      -p max_angular_z:="$BASE_MAX_ANGULAR_Z" \
      -p enable_strafe:=false \
      -p arduino_pwm:="$BASE_ARDUINO_PWM" \
      -p set_pwm_on_start:=true \
      -p log_sent_command:=true \
      -p log_all_commands:=false \
      -p log_encoder_line:=false
else
  start_node "05_base_key_odom" \
    "$HOME/ze-ri/ros2_ws/install/zeri_base/bin/base_key_odom_serial_node" --ros-args \
      -p port:="$ARDUINO_PORT" \
      -p baudrate:=115200 \
      -p cmd_topic:=/cmd_vel \
      -p odom_topic:=/odom \
      -p ticks_per_rev:=3464.0 \
      -p wheel_radius:=0.075 \
      -p lx:=0.1575 \
      -p ly:=0.2125 \
      -p key_hz:="$BASE_KEY_HZ" \
      -p enable_strafe:=false \
      -p mixed_forward_turn:=true \
      -p turn_mix_gain:="$BASE_TURN_MIX_GAIN" \
      -p turn_mix_min_duty:="$BASE_TURN_MIX_MIN_DUTY" \
      -p turn_mix_max_duty:="$BASE_TURN_MIX_MAX_DUTY" \
      -p angular_ref:=0.45 \
      -p log_sent_key:=true \
      -p log_all_keys:=false \
      -p log_encoder_line:=false
fi

start_node "06_slam_toolbox" \
  ros2 launch slam_toolbox online_async_launch.py \
    use_sim_time:=false \
    slam_params_file:="$SLAM_PARAMS_FILE"

start_node "07_respeaker_vad" \
  "$HOME/ze-ri/ros2_ws/install/zeri_voice/bin/respeaker_vad_doa_node" --ros-args \
    -p poll_hz:="$VAD_POLL_HZ" \
    -p vad_topic:=/zeri/audio/vad \
    -p doa_topic:=/zeri/audio/doa_deg \
    -p state_topic:=/zeri/audio/state \
    -p log_active_only:=true

start_node "08_camera_tf" \
  ros2 run tf2_ros static_transform_publisher \
    --x "$CAMERA_X" \
    --y "$CAMERA_Y" \
    --z "$CAMERA_Z" \
    --roll "$CAMERA_ROLL" \
    --pitch "$CAMERA_PITCH" \
    --yaw "$CAMERA_YAW" \
    --frame-id base_link \
    --child-frame-id "$CAMERA_FRAME_ID"

if [ "$START_PERSON_FOLLOW" = "true" ]; then
  start_node "09_camera_person_follow" \
    "$HOME/ze-ri/ros2_ws/install/zeri_camera/bin/camera_person_follow_node" --ros-args \
      -p rgb_topic:="$RGB_TOPIC" \
      -p depth_topic:="$DEPTH_TOPIC" \
      -p vad_topic:=/zeri/audio/vad \
      -p cmd_topic:=/cmd_vel_raw \
      -p state_topic:=/zeri/person_follow/state \
      -p model_path:="$MODEL_PATH" \
      -p device:="$YOLO_DEVICE" \
      -p imgsz:="$YOLO_IMGSZ" \
      -p conf_thres:="$YOLO_CONF" \
      -p infer_hz:="$YOLO_HZ" \
      -p use_vad_gate:="$USE_VAD_GATE" \
      -p voice_hold_sec:="$VOICE_HOLD_SEC" \
      -p target_distance_m:="$TARGET_DISTANCE_M" \
      -p distance_deadband_m:="$DISTANCE_DEADBAND_M" \
      -p too_close_m:="$TOO_CLOSE_M" \
      -p center_deadband_norm:="$CENTER_DEADBAND_NORM" \
      -p forward_max_err_norm:="$FORWARD_MAX_ERR_NORM" \
      -p forward_speed:="$FORWARD_SPEED" \
      -p turn_kp:="$TURN_KP" \
      -p min_turn_speed:="$MIN_TURN_SPEED" \
      -p max_turn_speed:="$MAX_TURN_SPEED" \
      -p invert_turn:="$INVERT_TURN"
else
  echo "[SKIP] 09_camera_person_follow"
  echo "       Nav2 or another planner may publish /cmd_vel_raw"
fi

start_node "10_person_map_marker" \
  "$HOME/ze-ri/ros2_ws/install/zeri_camera/bin/person_map_marker_node" --ros-args \
    -p rgb_topic:="$RGB_TOPIC" \
    -p depth_topic:="$DEPTH_TOPIC" \
    -p camera_info_topic:="$CAMERA_INFO_TOPIC" \
    -p vad_topic:=/zeri/audio/vad \
    -p marker_topic:=/zeri/person_markers \
    -p target_frame:=map \
    -p model_path:="$MODEL_PATH" \
    -p device:="$YOLO_DEVICE" \
    -p imgsz:="$YOLO_IMGSZ" \
    -p conf_thres:="$YOLO_CONF" \
    -p infer_hz:=4.0 \
    -p use_vad_gate:="$MARKER_USE_VAD_GATE" \
    -p voice_hold_sec:="$VOICE_HOLD_SEC" \
    -p storage_path:="$PERSON_MARKERS_FILE" \
    -p max_markers:="$MAX_PERSON_MARKERS" \
    -p update_existing_markers:="$UPDATE_PERSON_MARKERS" \
    -p marker_update_alpha:="$PERSON_MARKER_UPDATE_ALPHA" \
    -p max_marker_update_jump_m:="$PERSON_MARKER_UPDATE_JUMP_M" \
    -p publish_body_marker:=true

if [ "$START_DEPTH_POINTCLOUD" = "true" ]; then
  start_node "11_rgbd_pointcloud" \
    env PYTHONPATH="$HOME/ze-ri/ros2_ws/install/zeri_camera/lib/python3.12/site-packages:$PYTHONPATH" \
    "$HOME/ze-ri/.venv/bin/python" -m zeri_camera.rgbd_pointcloud_node --ros-args \
      -p rgb_topic:="$RGB_TOPIC" \
      -p depth_topic:="$DEPTH_TOPIC" \
      -p camera_info_topic:="$CAMERA_INFO_TOPIC" \
      -p points_topic:="$POINTCLOUD_TOPIC" \
      -p publish_hz:="$POINTCLOUD_HZ" \
      -p stride:="$POINTCLOUD_STRIDE"
else
  echo "[SKIP] 11_rgbd_pointcloud"
fi

if [ "$START_RVIZ" = "true" ]; then
  start_node "12_rviz2" rviz2
fi

echo
echo "[READY] YOLO person-follow drive is running."
echo
echo "Check:"
echo "  ros2 topic echo /zeri/person_follow/state"
echo "  ros2 topic echo /zeri/person_markers"
echo "  ros2 topic hz $POINTCLOUD_TOPIC"
echo "  ros2 topic echo /zeri/voice_stop_guard/state"
echo "  ros2 topic echo /cmd_vel_raw"
echo "  ros2 topic echo /cmd_vel"
echo
echo "Important:"
echo "  NO_PERSON -> stop, no rotation"
echo "  FORWARD_STEER_PERSON -> vx + wz together"
echo
echo "Logs:"
echo "  $LOG_DIR"
echo
echo "Press Ctrl+C to stop."

while true; do
  sleep 1
done
