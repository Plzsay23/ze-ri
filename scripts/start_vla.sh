#!/usr/bin/env bash
set -Eeuo pipefail

# Ze-Ri two-arm VLA launcher
# Starts:
#   1) left policy server  on 8081
#   2) right policy server on 8082
#   3) left VLA client    (/dev/follower_left  + /dev/cam_left)
#   4) right VLA client   (/dev/follower_right + /dev/cam_right)
#   5) VLA router         (/zeri/vla/task_request -> left/right command)
#
# Default model for both arms: plzsay/pen_and_cup
# Camera key for both VLA clients: top

ZERI_ROOT="${ZERI_ROOT:-$HOME/ze-ri}"
SOURCE_FILE="${SOURCE_FILE:-$ZERI_ROOT/source_zeri_vlm.sh}"

if [[ ! -f "$SOURCE_FILE" ]]; then
  echo "[ERROR] source file not found: $SOURCE_FILE" >&2
  exit 1
fi

# Some ROS setup scripts reference unset variables. Avoid set -u failure during source.
set +u
source "$SOURCE_FILE"
set -u

cd "$ZERI_ROOT"

CONFIG_DIR="$ZERI_ROOT/config/vla"
LOG_DIR="$ZERI_ROOT/logs/vla_two_arm_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$CONFIG_DIR" "$LOG_DIR" "$ZERI_ROOT/policies"

# -----------------------------------------------------------------------------
# User-editable defaults
# -----------------------------------------------------------------------------
HOST="${HOST:-127.0.0.1}"
LEFT_SERVER_PORT="${LEFT_SERVER_PORT:-8081}"
RIGHT_SERVER_PORT="${RIGHT_SERVER_PORT:-8082}"

LEFT_MODEL="${LEFT_MODEL:-plzsay/pen_and_cup}"
RIGHT_MODEL="${RIGHT_MODEL:-plzsay/pen_and_cup}"
LEFT_POLICY_ID="${LEFT_POLICY_ID:-pen_and_cup}"
RIGHT_POLICY_ID="${RIGHT_POLICY_ID:-pen_and_cup}"

LEFT_ARM_PORT="${LEFT_ARM_PORT:-/dev/follower_left}"
RIGHT_ARM_PORT="${RIGHT_ARM_PORT:-/dev/follower_right}"
LEFT_ROBOT_ID="${LEFT_ROBOT_ID:-follower_left}"
RIGHT_ROBOT_ID="${RIGHT_ROBOT_ID:-follower_right}"

LEFT_CAM="${LEFT_CAM:-/dev/cam_left}"
RIGHT_CAM="${RIGHT_CAM:-/dev/cam_right}"
CAM_WIDTH="${CAM_WIDTH:-640}"
CAM_HEIGHT="${CAM_HEIGHT:-480}"
WRIST_FPS="${WRIST_FPS:-25}"

RUN_DURATION_SEC="${RUN_DURATION_SEC:-20.0}"
TIMEOUT_SEC="${TIMEOUT_SEC:-60.0}"
HOME_RETURN_SECONDS="${HOME_RETURN_SECONDS:-2.0}"
HOME_RETURN_FPS="${HOME_RETURN_FPS:-50}"

LEFT_HOME_RAW="${LEFT_HOME_RAW:-$ZERI_ROOT/policies/so101_start_raw_ticks_left.json}"
RIGHT_HOME_RAW="${RIGHT_HOME_RAW:-$ZERI_ROOT/policies/so101_start_raw_ticks_right.json}"

LEFT_MANIFEST="$CONFIG_DIR/left_vla_policy_manifest.json"
RIGHT_MANIFEST="$CONFIG_DIR/right_vla_policy_manifest.json"
ROUTE_MANIFEST="$CONFIG_DIR/vla_route_manifest.json"

CLIENT_MODULE="${CLIENT_MODULE:-lerobot.async_inference.robot_client_ros_multi_gate_raw_home}"
ROUTER_SCRIPT="${ROUTER_SCRIPT:-$ZERI_ROOT/src/lerobot/vlm_agent/vla_task_router_multi_node.py}"

# -----------------------------------------------------------------------------
# Sanity checks
# -----------------------------------------------------------------------------
CLIENT_FILE="$ZERI_ROOT/src/lerobot/async_inference/robot_client_ros_multi_gate_raw_home.py"
if [[ ! -f "$CLIENT_FILE" ]]; then
  echo "[ERROR] missing client file: $CLIENT_FILE" >&2
  echo "        Copy robot_client_ros_multi_gate_raw_home.py into src/lerobot/async_inference/ first." >&2
  exit 1
fi

if [[ ! -f "$ROUTER_SCRIPT" ]]; then
  echo "[ERROR] missing router file: $ROUTER_SCRIPT" >&2
  exit 1
fi

if [[ ! -e "$LEFT_ARM_PORT" ]]; then
  echo "[WARN] left arm port not found now: $LEFT_ARM_PORT" >&2
fi
if [[ ! -e "$RIGHT_ARM_PORT" ]]; then
  echo "[WARN] right arm port not found now: $RIGHT_ARM_PORT" >&2
fi
if [[ ! -e "$LEFT_CAM" ]]; then
  echo "[WARN] left camera not found now: $LEFT_CAM" >&2
fi
if [[ ! -e "$RIGHT_CAM" ]]; then
  echo "[WARN] right camera not found now: $RIGHT_CAM" >&2
fi
if [[ ! -f "$LEFT_HOME_RAW" ]]; then
  echo "[WARN] left raw home file not found: $LEFT_HOME_RAW" >&2
fi
if [[ ! -f "$RIGHT_HOME_RAW" ]]; then
  echo "[WARN] right raw home file not found: $RIGHT_HOME_RAW" >&2
fi

# -----------------------------------------------------------------------------
# Generate manifests for current test setup
# -----------------------------------------------------------------------------
cat > "$LEFT_MANIFEST" <<JSON
{
  "default_policy_id": "$LEFT_POLICY_ID",
  "policies": [
    {
      "id": "$LEFT_POLICY_ID",
      "policy_type": "act",
      "pretrained_name_or_path": "$LEFT_MODEL",
      "description": "Left-arm ACT policy. Camera key: top."
    }
  ]
}
JSON

cat > "$RIGHT_MANIFEST" <<JSON
{
  "default_policy_id": "$RIGHT_POLICY_ID",
  "policies": [
    {
      "id": "$RIGHT_POLICY_ID",
      "policy_type": "act",
      "pretrained_name_or_path": "$RIGHT_MODEL",
      "description": "Right-arm ACT policy. Camera key: top."
    }
  ]
}
JSON

cat > "$ROUTE_MANIFEST" <<JSON
{
  "routes": {
    "pen_and_cup_left": {
      "arm": "left",
      "policy_id": "$LEFT_POLICY_ID",
      "task": "Pick up the pen and place it in the cup.",
      "duration_sec": $RUN_DURATION_SEC,
      "timeout_sec": $TIMEOUT_SEC
    },
    "pen_and_cup_right": {
      "arm": "right",
      "policy_id": "$RIGHT_POLICY_ID",
      "task": "Pick up the pen and place it in the cup.",
      "duration_sec": $RUN_DURATION_SEC,
      "timeout_sec": $TIMEOUT_SEC
    },
    "oxygen_mask_delivery": {
      "arm": "left",
      "policy_id": "$LEFT_POLICY_ID",
      "task": "Pick up the pen and place it in the cup.",
      "duration_sec": $RUN_DURATION_SEC,
      "timeout_sec": $TIMEOUT_SEC
    },
    "radio_delivery": {
      "arm": "right",
      "policy_id": "$RIGHT_POLICY_ID",
      "task": "Pick up the pen and place it in the cup.",
      "duration_sec": $RUN_DURATION_SEC,
      "timeout_sec": $TIMEOUT_SEC
    }
  }
}
JSON

PIDS=()
NAMES=()

cleanup() {
  local code=$?
  trap - INT TERM EXIT
  echo
  echo "[INFO] stopping VLA processes..."
  for pid in "${PIDS[@]:-}"; do
    if kill -0 "$pid" 2>/dev/null; then
      kill "$pid" 2>/dev/null || true
    fi
  done
  sleep 1
  for pid in "${PIDS[@]:-}"; do
    if kill -0 "$pid" 2>/dev/null; then
      kill -9 "$pid" 2>/dev/null || true
    fi
  done
  echo "[INFO] logs saved in: $LOG_DIR"
  exit "$code"
}
trap cleanup INT TERM EXIT

start_bg() {
  local name="$1"
  shift
  local log="$LOG_DIR/${name}.log"
  echo "[START] $name"
  echo "        log: $log"
  ( "$@" ) > "$log" 2>&1 &
  local pid=$!
  PIDS+=("$pid")
  NAMES+=("$name")
  echo "        pid: $pid"
}

# -----------------------------------------------------------------------------
# Start processes
# -----------------------------------------------------------------------------
start_bg left_policy_server \
  env LEROBOT_POLICY_MANIFEST="$LEFT_MANIFEST" \
  python -m lerobot.async_inference.policy_server \
    --host="$HOST" \
    --port="$LEFT_SERVER_PORT"

sleep 2

start_bg right_policy_server \
  env LEROBOT_POLICY_MANIFEST="$RIGHT_MANIFEST" \
  python -m lerobot.async_inference.policy_server \
    --host="$HOST" \
    --port="$RIGHT_SERVER_PORT"

sleep 4

start_bg left_vla_client \
  env LEROBOT_POLICY_MANIFEST="$LEFT_MANIFEST" \
      LEROBOT_HOME_RETURN_SECONDS="$HOME_RETURN_SECONDS" \
      LEROBOT_HOME_RETURN_FPS="$HOME_RETURN_FPS" \
  python -m "$CLIENT_MODULE" \
    --robot.type=so101_follower \
    --robot.port="$LEFT_ARM_PORT" \
    --robot.id="$LEFT_ROBOT_ID" \
    --robot.cameras="{top: {type: opencv, index_or_path: $LEFT_CAM, width: $CAM_WIDTH, height: $CAM_HEIGHT, fps: $WRIST_FPS}}" \
    --task=idle \
    --server_address="$HOST:$LEFT_SERVER_PORT" \
    --policy_type=act \
    --pretrained_name_or_path="$LEFT_MODEL" \
    --policy_device=cuda \
    --actions_per_chunk=100 \
    --chunk_size_threshold=0.1 \
    --aggregate_fn_name=weighted_average \
    --debug_visualize_queue_size=True \
    --zeri_client_name=left \
    --zeri_command_topic=/zeri/vla/left/command \
    --zeri_stop_topic=/zeri/vla/left/stop \
    --zeri_status_topic=/zeri/vla/left/status \
    --zeri_run_duration_sec="$RUN_DURATION_SEC" \
    --zeri_timeout_sec="$TIMEOUT_SEC" \
    --zeri_auto_home_on_done=true \
    --zeri_home_raw_ticks_json="$LEFT_HOME_RAW"

sleep 2

start_bg right_vla_client \
  env LEROBOT_POLICY_MANIFEST="$RIGHT_MANIFEST" \
      LEROBOT_HOME_RETURN_SECONDS="$HOME_RETURN_SECONDS" \
      LEROBOT_HOME_RETURN_FPS="$HOME_RETURN_FPS" \
  python -m "$CLIENT_MODULE" \
    --robot.type=so101_follower \
    --robot.port="$RIGHT_ARM_PORT" \
    --robot.id="$RIGHT_ROBOT_ID" \
    --robot.cameras="{top: {type: opencv, index_or_path: $RIGHT_CAM, width: $CAM_WIDTH, height: $CAM_HEIGHT, fps: $WRIST_FPS}}" \
    --task=idle \
    --server_address="$HOST:$RIGHT_SERVER_PORT" \
    --policy_type=act \
    --pretrained_name_or_path="$RIGHT_MODEL" \
    --policy_device=cuda \
    --actions_per_chunk=100 \
    --chunk_size_threshold=0.1 \
    --aggregate_fn_name=weighted_average \
    --debug_visualize_queue_size=True \
    --zeri_client_name=right \
    --zeri_command_topic=/zeri/vla/right/command \
    --zeri_stop_topic=/zeri/vla/right/stop \
    --zeri_status_topic=/zeri/vla/right/status \
    --zeri_run_duration_sec="$RUN_DURATION_SEC" \
    --zeri_timeout_sec="$TIMEOUT_SEC" \
    --zeri_auto_home_on_done=true \
    --zeri_home_raw_ticks_json="$RIGHT_HOME_RAW"

sleep 2

start_bg vla_router \
  python "$ROUTER_SCRIPT" \
    --ros-args \
    -p route_manifest_path:="$ROUTE_MANIFEST" \
    -p task_request_topic:=/zeri/vla/task_request \
    -p left_command_topic:=/zeri/vla/left/command \
    -p right_command_topic:=/zeri/vla/right/command \
    -p left_status_topic:=/zeri/vla/left/status \
    -p right_status_topic:=/zeri/vla/right/status \
    -p vla_status_topic:=/zeri/vla/status

sleep 2

echo
echo "[READY] two-arm VLA stack started."
echo "[INFO] logs: $LOG_DIR"
echo
echo "Test left:"
echo "  ros2 topic pub --once /zeri/vla/task_request std_msgs/msg/String \"{data: '{\\\"selected_task\\\":\\\"pen_and_cup_left\\\",\\\"instruction\\\":\\\"Pick up the pen and place it in the cup.\\\",\\\"task_duration_sec\\\":20.0,\\\"timeout_sec\\\":60.0}'}\""
echo
echo "Test right:"
echo "  ros2 topic pub --once /zeri/vla/task_request std_msgs/msg/String \"{data: '{\\\"selected_task\\\":\\\"pen_and_cup_right\\\",\\\"instruction\\\":\\\"Pick up the pen and place it in the cup.\\\",\\\"task_duration_sec\\\":20.0,\\\"timeout_sec\\\":60.0}'}\""
echo
echo "Monitor:"
echo "  ros2 topic echo /zeri/vla/status"
echo
echo "Press Ctrl+C to stop all processes."

# Keep launcher alive. If any child exits, stop the whole stack to avoid half-running robot state.
while true; do
  sleep 2
  for i in "${!PIDS[@]}"; do
    pid="${PIDS[$i]}"
    name="${NAMES[$i]}"
    if ! kill -0 "$pid" 2>/dev/null; then
      echo "[ERROR] process exited: $name pid=$pid"
      echo "[ERROR] check log: $LOG_DIR/${name}.log"
      exit 1
    fi
  done
done
