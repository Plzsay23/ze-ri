#!/usr/bin/env bash

set -e

ROOT="$HOME/ze-ri"
LOG_DIR="$ROOT/logs/yolo_person_nav_drive_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$LOG_DIR"

source "$ROOT/source_zeri.sh"
export PYTHONPATH="$ROOT/.venv/lib/python3.12/site-packages:$PYTHONPATH"

NAV2_PARAMS_FILE="${NAV2_PARAMS_FILE:-$ROOT/configs/nav2/zeri_nav2_params.yaml}"
PERSON_MARKERS_FILE="${PERSON_MARKERS_FILE:-$ROOT/data/person_markers.json}"
CLEAR_PERSON_MARKERS_ON_START="${CLEAR_PERSON_MARKERS_ON_START:-true}"
MAX_PERSON_MARKERS="${MAX_PERSON_MARKERS:-50}"
START_BEHAVIOR_ENGINE="${START_BEHAVIOR_ENGINE:-true}"
BEHAVIOR_AUTO_APPROACH="${BEHAVIOR_AUTO_APPROACH:-true}"
BEHAVIOR_APPROACH_DISTANCE_M="${BEHAVIOR_APPROACH_DISTANCE_M:-1.0}"
BEHAVIOR_PERSON_FRESH_TIMEOUT_SEC="${BEHAVIOR_PERSON_FRESH_TIMEOUT_SEC:-5.0}"
BEHAVIOR_ARRIVAL_MODE="${BEHAVIOR_ARRIVAL_MODE:-vlm_wait}"
BEHAVIOR_ARRIVAL_DWELL_SEC="${BEHAVIOR_ARRIVAL_DWELL_SEC:-5.0}"
BEHAVIOR_MISSION_EVENT_TOPIC="${BEHAVIOR_MISSION_EVENT_TOPIC:-/zeri/mission/event}"
BEHAVIOR_ENABLE_SEARCH="${BEHAVIOR_ENABLE_SEARCH:-true}"
BEHAVIOR_SEARCH_WAYPOINTS="${BEHAVIOR_SEARCH_WAYPOINTS:-}"
BEHAVIOR_SEARCH_STEP_M="${BEHAVIOR_SEARCH_STEP_M:-0.5}"
BEHAVIOR_SEARCH_PAUSE_SEC="${BEHAVIOR_SEARCH_PAUSE_SEC:-2.0}"
START_RVIZ="false"
TEST_MODE="false"

for arg in "$@"; do
  case "$arg" in
    --rviz)
      START_RVIZ="true"
      ;;
    --test)
      TEST_MODE="true"
      BEHAVIOR_ARRIVAL_MODE="test_dwell"
      BEHAVIOR_ARRIVAL_DWELL_SEC="${BEHAVIOR_ARRIVAL_DWELL_SEC:-5.0}"
      ;;
    -h|--help)
      echo "Usage: $0 [--rviz] [--test]"
      echo "Default: auto approach, wait for VLM after arrival"
      echo "Test: --test waits 5 sec and resumes automatically"
      exit 0
      ;;
    *)
      echo "[ERROR] unknown argument: $arg"
      echo "Usage: $0 [--rviz] [--test]"
      exit 1
      ;;
  esac
done

PIDS=()

cleanup() {
  echo
  echo "[Ze-Ri YOLO Person Nav] stopping..."

  ros2 topic pub --once /cmd_vel_raw geometry_msgs/msg/Twist \
    "{linear: {x: 0.0, y: 0.0, z: 0.0}, angular: {x: 0.0, y: 0.0, z: 0.0}}" >/dev/null 2>&1 || true

  ros2 topic pub --once /cmd_vel geometry_msgs/msg/Twist \
    "{linear: {x: 0.0, y: 0.0, z: 0.0}, angular: {x: 0.0, y: 0.0, z: 0.0}}" >/dev/null 2>&1 || true

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

  echo "[Ze-Ri YOLO Person Nav] stopped"
  echo "logs: $LOG_DIR"
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

  sleep 1.0

  if ! kill -0 "$pid" 2>/dev/null; then
    echo "[ERROR] $name exited early. Check log:"
    echo "        $LOG_DIR/$name.log"
    tail -120 "$LOG_DIR/$name.log" || true
    exit 1
  fi
}

wait_topic() {
  local topic="$1"
  local timeout_sec="${2:-30}"
  local start
  start=$(date +%s)

  echo "[WAIT] topic $topic"

  while true; do
    if ros2 topic list 2>/dev/null | grep -qx "$topic"; then
      echo "       OK $topic"
      return 0
    fi

    local now
    now=$(date +%s)
    if [ $((now - start)) -ge "$timeout_sec" ]; then
      echo "       WARN missing after ${timeout_sec}s: $topic"
      return 1
    fi

    sleep 0.5
  done
}

require_file() {
  local path="$1"
  if [ ! -f "$path" ]; then
    echo "[ERROR] missing file: $path"
    exit 1
  fi
}

require_pkg() {
  local pkg="$1"
  if ! ros2 pkg prefix "$pkg" >/dev/null 2>&1; then
    echo "[ERROR] ROS package not found: $pkg"
    echo "        Nav2 is not installed or not sourced in this environment."
    echo "        On ROS Jazzy, install nav2_bringup/nav2 packages, then rerun:"
    echo "        ~/ze-ri/scripts/yolo_person_nav_drive.sh --rviz"
    exit 1
  fi
}

echo "[Ze-Ri YOLO Person Nav Drive]"
echo "  This is a new Nav2 wrapper. It does not modify yolo_person_follow_drive.sh."
echo "  Start camera first:"
echo "    ~/ze-ri/scripts/camera_rgbd.sh"
echo
echo "  NAV2_PARAMS_FILE=$NAV2_PARAMS_FILE"
echo "  PERSON_MARKERS_FILE=$PERSON_MARKERS_FILE"
echo "  CLEAR_PERSON_MARKERS_ON_START=$CLEAR_PERSON_MARKERS_ON_START"
echo "  MAX_PERSON_MARKERS=$MAX_PERSON_MARKERS"
echo "  START_BEHAVIOR_ENGINE=$START_BEHAVIOR_ENGINE"
echo "  BEHAVIOR_AUTO_APPROACH=$BEHAVIOR_AUTO_APPROACH"
echo "  BEHAVIOR_APPROACH_DISTANCE_M=$BEHAVIOR_APPROACH_DISTANCE_M"
echo "  BEHAVIOR_PERSON_FRESH_TIMEOUT_SEC=$BEHAVIOR_PERSON_FRESH_TIMEOUT_SEC"
echo "  BEHAVIOR_ARRIVAL_MODE=$BEHAVIOR_ARRIVAL_MODE"
echo "  BEHAVIOR_ARRIVAL_DWELL_SEC=$BEHAVIOR_ARRIVAL_DWELL_SEC"
echo "  BEHAVIOR_MISSION_EVENT_TOPIC=$BEHAVIOR_MISSION_EVENT_TOPIC"
echo "  BEHAVIOR_ENABLE_SEARCH=$BEHAVIOR_ENABLE_SEARCH"
echo "  BEHAVIOR_SEARCH_WAYPOINTS=$BEHAVIOR_SEARCH_WAYPOINTS"
echo "  BEHAVIOR_SEARCH_STEP_M=$BEHAVIOR_SEARCH_STEP_M"
echo "  BEHAVIOR_SEARCH_PAUSE_SEC=$BEHAVIOR_SEARCH_PAUSE_SEC"
echo "  RVIZ=$START_RVIZ"
echo "  TEST_MODE=$TEST_MODE"
echo
echo "Important:"
echo "  Nav2 command velocity is configured to publish /cmd_vel_raw."
echo "  LiDAR/depth safety still gates /cmd_vel_raw -> /cmd_vel."
echo "  Do not use YOLO follow voice command and Nav2 goal driving at the same time yet."
echo "  Behavior engine auto approach is enabled by default. Use cancel if needed."
echo

require_file "$ROOT/scripts/yolo_person_follow_drive.sh"
require_file "$NAV2_PARAMS_FILE"
require_pkg nav2_bringup

if [ "$CLEAR_PERSON_MARKERS_ON_START" = "true" ]; then
  mkdir -p "$(dirname "$PERSON_MARKERS_FILE")"
  rm -f "$PERSON_MARKERS_FILE"
  echo "[CLEAR] removed previous person markers: $PERSON_MARKERS_FILE"
fi

wait_topic /zeri/vlm/input_rgb 30 || true
wait_topic /zeri/vlm/input_depth 30 || true
wait_topic /zeri/camera/color/camera_info 10 || true

start_node "01_yolo_person_mapping_stack" \
  env \
    START_PERSON_FOLLOW=false \
    PERSON_MARKERS_FILE="$PERSON_MARKERS_FILE" \
    MAX_PERSON_MARKERS="$MAX_PERSON_MARKERS" \
    "$ROOT/scripts/yolo_person_follow_drive.sh"

wait_topic /map 45 || {
  echo "[ERROR] /map is missing. slam_toolbox did not publish a map."
  echo "        Check:"
  echo "          ROS_DOMAIN_ID=${ROS_DOMAIN_ID:-unset} ros2 lifecycle get /slam_toolbox"
  echo "          ROS_DOMAIN_ID=${ROS_DOMAIN_ID:-unset} ros2 topic hz /scan_front"
  exit 1
}
wait_topic /odom 20 || true
wait_topic /scan_front 20 || true
wait_topic /zeri/person_markers 20 || true

start_node "02_nav2" \
  ros2 launch nav2_bringup navigation_launch.py \
    use_sim_time:=false \
    params_file:="$NAV2_PARAMS_FILE" \
    autostart:=true

if [ "$START_BEHAVIOR_ENGINE" = "true" ]; then
  BEHAVIOR_ROS_ARGS=(
    -p marker_topic:=/zeri/person_markers
    -p command_topic:=/zeri/behavior/command
    -p state_topic:=/zeri/behavior/state
    -p vlm_event_topic:=/zeri/behavior/vlm_event
    -p mission_event_topic:="$BEHAVIOR_MISSION_EVENT_TOPIC"
    -p nav_action_name:=/navigate_to_pose
    -p target_frame:=map
    -p base_frame:=base_link
    -p approach_distance_m:="$BEHAVIOR_APPROACH_DISTANCE_M"
    -p person_fresh_timeout_sec:="$BEHAVIOR_PERSON_FRESH_TIMEOUT_SEC"
    -p auto_approach:="$BEHAVIOR_AUTO_APPROACH"
    -p arrival_mode:="$BEHAVIOR_ARRIVAL_MODE"
    -p arrival_dwell_sec:="$BEHAVIOR_ARRIVAL_DWELL_SEC"
    -p enable_search:="$BEHAVIOR_ENABLE_SEARCH"
    -p search_step_m:="$BEHAVIOR_SEARCH_STEP_M"
    -p search_pause_sec:="$BEHAVIOR_SEARCH_PAUSE_SEC"
  )

  if [ -n "$BEHAVIOR_SEARCH_WAYPOINTS" ]; then
    BEHAVIOR_ROS_ARGS+=(-p search_waypoints:="$BEHAVIOR_SEARCH_WAYPOINTS")
  fi

  start_node "03_behavior_engine" \
    env PYTHONPATH="$ROOT/ros2_ws/src/zeri_bringup:$ROOT/ros2_ws/install/zeri_bringup/lib/python3.12/site-packages:$PYTHONPATH" \
    "$ROOT/.venv/bin/python" -m zeri_bringup.disaster_behavior_engine_node --ros-args \
      "${BEHAVIOR_ROS_ARGS[@]}"
else
  echo "[SKIP] 03_behavior_engine"
fi

if [ "$START_RVIZ" = "true" ]; then
  start_node "04_rviz2" rviz2
fi

echo
echo "[READY] YOLO person detection + map markers + Nav2 are running."
echo
echo "Check:"
echo "  ros2 topic echo /zeri/person_markers --once"
echo "  ros2 action list | grep navigate"
echo "  ros2 topic echo /cmd_vel_raw"
echo "  ros2 topic echo /cmd_vel"
echo "  ros2 topic echo /zeri/behavior/state"
echo "  ros2 topic echo /zeri/behavior/vlm_event"
echo "  ros2 topic echo /zeri/mission/event"
echo
echo "Behavior commands:"
echo "  ros2 topic pub --once /zeri/behavior/command std_msgs/msg/String \"{data: 'go_nearest'}\""
echo "  ros2 topic pub --once /zeri/behavior/command std_msgs/msg/String \"{data: 'go_person 1'}\""
echo "  ros2 topic pub --once /zeri/behavior/command std_msgs/msg/String \"{data: 'auto_on'}\""
echo "  ros2 topic pub --once /zeri/behavior/command std_msgs/msg/String \"{data: 'cancel'}\""
echo "  ros2 topic pub --once /zeri/behavior/command std_msgs/msg/String \"{data: 'resume'}\""
echo "  ros2 topic pub --once /zeri/behavior/command std_msgs/msg/String \"{data: 'reset_visited'}\""
echo
echo "In RViz:"
echo "  1. Set Fixed Frame to map"
echo "  2. Add Map: /map"
echo "  3. Add MarkerArray: /zeri/person_markers"
echo "  4. Add Nav2 panels/displays if available"
echo "  5. Use 2D Goal Pose for a small nearby test goal"
echo
echo "Logs:"
echo "  $LOG_DIR"
echo
echo "Press Ctrl+C to stop."

while true; do
  sleep 1
done
