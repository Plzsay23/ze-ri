#!/usr/bin/env bash
set -Eeuo pipefail

# Ze-Ri demo launcher: camera + autonomy + VLA + VLM + TTS + dashboard
# STT is intentionally NOT started by this script.

ROOT="${ZERI_ROOT:-$HOME/ze-ri}"
SOURCE_FILE="${SOURCE_FILE:-$ROOT/source_zeri.sh}"
LOG_DIR="${LOG_DIR:-$ROOT/logs/zeri_demo_all_$(date +%Y%m%d_%H%M%S)}"
mkdir -p "$LOG_DIR"

# -------------------------
# User tunables
# -------------------------
START_CAMERA="${START_CAMERA:-true}"
START_TTS="${START_TTS:-true}"
START_VLA="${START_VLA:-true}"
START_AUTONOMY="${START_AUTONOMY:-true}"
START_VLM="${START_VLM:-true}"
START_DASHBOARD="${START_DASHBOARD:-true}"
# TTS
# Speed is fixed inside tts_edge_node.py. Do not control it from this launcher.
TTS_VOICE="${TTS_VOICE:-ko-KR-SunHiNeural}"

# Dashboard
DASHBOARD_HOST="${DASHBOARD_HOST:-0.0.0.0}"
DASHBOARD_PORT="${DASHBOARD_PORT:-8080}"
POINTCLOUD_MAX_POINTS="${POINTCLOUD_MAX_POINTS:-30000}"

# VLM / VLA topics
RGB_TOPIC="${RGB_TOPIC:-/zeri/vlm/input_rgb}"
DEPTH_TOPIC="${DEPTH_TOPIC:-/zeri/vlm/input_depth}"
POINTCLOUD_TOPIC="${POINTCLOUD_TOPIC:-/zeri/vlm/pointcloud}"
STT_TOPIC="${STT_TOPIC:-/stt/text}"
MISSION_EVENT_TOPIC="${MISSION_EVENT_TOPIC:-/zeri/mission/event}"
LEFT_HANDOFF_IMAGE_TOPIC="${LEFT_HANDOFF_IMAGE_TOPIC:-/zeri/vla/left/handoff_image}"
RIGHT_HANDOFF_IMAGE_TOPIC="${RIGHT_HANDOFF_IMAGE_TOPIC:-/zeri/vla/right/handoff_image}"

# Script paths
CAMERA_SCRIPT="${CAMERA_SCRIPT:-$ROOT/scripts/camera_rgbd_pointcloud.sh}"
if [[ ! -x "$CAMERA_SCRIPT" && -x "$ROOT/scripts/camera_rgbd.sh" ]]; then
  CAMERA_SCRIPT="$ROOT/scripts/camera_rgbd.sh"
fi
AUTONOMY_SCRIPT="${AUTONOMY_SCRIPT:-$ROOT/scripts/person_follow_depth_lidar_drive.sh}"
VLA_SCRIPT="${VLA_SCRIPT:-$ROOT/scripts/start_vla_yolo_handoff.sh}"
TTS_SCRIPT="${TTS_SCRIPT:-$ROOT/src/lerobot/vlm_agent/tts_edge_node.py}"
VLM_SCRIPT="${VLM_SCRIPT:-$ROOT/src/lerobot/vlm_agent/vlm_stt_bridge_node.py}"
if [[ -x "$ROOT/dashboard/dashboard_server.py" ]]; then
  DASHBOARD_SCRIPT="${DASHBOARD_SCRIPT:-$ROOT/dashboard/dashboard_server.py}"
elif [[ -x "$ROOT/dashboard_server.py" ]]; then
  DASHBOARD_SCRIPT="${DASHBOARD_SCRIPT:-$ROOT/dashboard_server.py}"
elif [[ -f "$ROOT/dashboard/dashboard_server.py" ]]; then
  DASHBOARD_SCRIPT="${DASHBOARD_SCRIPT:-$ROOT/dashboard/dashboard_server.py}"
else
  DASHBOARD_SCRIPT="${DASHBOARD_SCRIPT:-$ROOT/dashboard_server.py}"
fi

PIDS=()
NAMES=()
say() {
  echo "[$(date +%H:%M:%S)] $*"
}

require_file() {
  local path="$1"
  local label="$2"
  if [[ ! -e "$path" ]]; then
    say "ERROR missing $label: $path"
    exit 1
  fi
}

start_bg() {
  local name="$1"
  shift
  local log="$LOG_DIR/${name}.log"

  say "START $name"
  printf '[CMD] ' > "$log"
  printf '%q ' "$@" >> "$log"
  printf '\n\n' >> "$log"

  ( cd "$ROOT" && "$@" ) >> "$log" 2>&1 &
  local pid=$!
  PIDS+=("$pid")
  NAMES+=("$name")
  echo "$pid" > "$LOG_DIR/${name}.pid"
  say "  pid=$pid log=$log"
}

wait_topic() {
  local topic="$1"
  local timeout_sec="${2:-20}"
  local start now
  start=$(date +%s)
  say "WAIT topic $topic"

  while true; do
    if ros2 topic list 2>/dev/null | grep -qx "$topic"; then
      say "  OK $topic"
      return 0
    fi

    now=$(date +%s)
    if (( now - start >= timeout_sec )); then
      say "  WARN topic not found within ${timeout_sec}s: $topic"
      return 1
    fi
    sleep 0.5
  done
}

print_summary() {
  echo
  say "============================================================"
  say "Ze-Ri demo launcher"
  say "ROOT=$ROOT"
  say "LOG_DIR=$LOG_DIR"
  say "STT is NOT started by this launcher."
  say ""
  say "START_CAMERA=$START_CAMERA"
  say "START_TTS=$START_TTS voice=$TTS_VOICE"
  say "START_VLA=$START_VLA"
  say "START_AUTONOMY=$START_AUTONOMY"
  say "START_VLM=$START_VLM"
  say "START_DASHBOARD=$START_DASHBOARD http://<ROBOT_IP>:$DASHBOARD_PORT"
  say ""
  say "RGB_TOPIC=$RGB_TOPIC"
  say "DEPTH_TOPIC=$DEPTH_TOPIC"
  say "POINTCLOUD_TOPIC=$POINTCLOUD_TOPIC"
  say "MISSION_EVENT_TOPIC=$MISSION_EVENT_TOPIC"
  say "============================================================"
  echo
}

cleanup() {
  local code=$?
  trap - EXIT INT TERM

  echo
  say "STOP requested. Stopping launched processes..."

  # Stop base velocity first for safety.
  if command -v ros2 >/dev/null 2>&1; then
    timeout 2 ros2 topic pub --once /cmd_vel geometry_msgs/msg/Twist \
      "{linear: {x: 0.0, y: 0.0, z: 0.0}, angular: {x: 0.0, y: 0.0, z: 0.0}}" \
      >/dev/null 2>&1 || true
  fi

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


  say "logs: $LOG_DIR"
  exit "$code"
}

trap cleanup EXIT INT TERM

if [[ ! -d "$ROOT" ]]; then
  echo "[ERROR] ROOT not found: $ROOT" >&2
  exit 1
fi
if [[ ! -f "$SOURCE_FILE" ]]; then
  echo "[ERROR] source file not found: $SOURCE_FILE" >&2
  exit 1
fi

cd "$ROOT"
set +u
source "$SOURCE_FILE"
set -u

export PYTHONPATH="$ROOT/.venv/lib/python3.12/site-packages:${PYTHONPATH:-}"

print_summary

[[ "$START_CAMERA" == "true" ]] && require_file "$CAMERA_SCRIPT" "camera script"
[[ "$START_TTS" == "true" ]] && require_file "$TTS_SCRIPT" "TTS script"
[[ "$START_VLA" == "true" ]] && require_file "$VLA_SCRIPT" "VLA launcher"
[[ "$START_AUTONOMY" == "true" ]] && require_file "$AUTONOMY_SCRIPT" "autonomy launcher"
[[ "$START_VLM" == "true" ]] && require_file "$VLM_SCRIPT" "VLM bridge"
[[ "$START_DASHBOARD" == "true" ]] && require_file "$DASHBOARD_SCRIPT" "dashboard server"
if [[ "$START_CAMERA" == "true" ]]; then
  start_bg "01_camera_rgbd_pointcloud" bash "$CAMERA_SCRIPT"
  wait_topic "$RGB_TOPIC" 30 || true
  wait_topic "$DEPTH_TOPIC" 30 || true
  wait_topic "$POINTCLOUD_TOPIC" 30 || true
fi

if [[ "$START_TTS" == "true" ]]; then
  start_bg "02_tts_edge" \
    python3 "$TTS_SCRIPT" \
      --ros-args \
      -p voice:="$TTS_VOICE"
fi

if [[ "$START_VLA" == "true" ]]; then
  start_bg "03_vla_yolo_handoff" \
    bash "$VLA_SCRIPT" --restart-vla
  # VLA stack startup is intentionally long; wait for status topics but do not fail hard.
  wait_topic "/zeri/vla/status" 35 || true
  wait_topic "/zeri/vla/left/status" 35 || true
  wait_topic "/zeri/vla/right/status" 35 || true
fi

if [[ "$START_AUTONOMY" == "true" ]]; then
  start_bg "04_person_follow_depth_lidar_drive" \
    bash "$AUTONOMY_SCRIPT"
  wait_topic "$MISSION_EVENT_TOPIC" 15 || true
  wait_topic "/cmd_vel" 20 || true
fi

if [[ "$START_VLM" == "true" ]]; then
  start_bg "05_vlm_stt_bridge" \
    python "$VLM_SCRIPT" \
      --ros-args \
      -p rgb_topic:="$RGB_TOPIC" \
      -p depth_topic:="$DEPTH_TOPIC" \
      -p stt_topic:="$STT_TOPIC" \
      -p mission_event_topic:="$MISSION_EVENT_TOPIC" \
      -p enable_mission_events:=true \
      -p enable_vla:=true \
      -p stt_gate_mode:=all \
      -p left_wrist_snapshot_topic:="$LEFT_HANDOFF_IMAGE_TOPIC" \
      -p right_wrist_snapshot_topic:="$RIGHT_HANDOFF_IMAGE_TOPIC"
  wait_topic "/zeri/vlm/inference_status" 15 || true
fi

if [[ "$START_DASHBOARD" == "true" ]]; then
  start_bg "06_dashboard" \
    python3 "$DASHBOARD_SCRIPT" \
      --host "$DASHBOARD_HOST" \
      --port "$DASHBOARD_PORT" \
      --image-qos best_effort \
      --rgb-topic "$RGB_TOPIC" \
      --depth-topic "$DEPTH_TOPIC" \
      --pointcloud-topic "$POINTCLOUD_TOPIC" \
      --pointcloud-max-points "$POINTCLOUD_MAX_POINTS"
fi

say "============================================================"
say "READY. STT is still OFF."
say "Dashboard: http://<ROBOT_IP>:$DASHBOARD_PORT"
say "Logs: $LOG_DIR"
say ""
say "Useful checks:"
say "  ros2 topic echo /zeri/mission/event"
say "  ros2 topic echo /zeri/vla/status"
say "  ros2 topic echo /zeri/vlm/inference_status"
say "  ros2 topic hz $RGB_TOPIC"
say "  ros2 topic hz $POINTCLOUD_TOPIC"
say ""
say "Press Ctrl+C here to stop all processes launched by this script."
say "============================================================"

while true; do
  sleep 2
  for i in "${!PIDS[@]}"; do
    pid="${PIDS[$i]}"
    name="${NAMES[$i]}"
    if ! kill -0 "$pid" 2>/dev/null; then
      say "ERROR process exited: $name pid=$pid"
      say "check log: $LOG_DIR/${name}.log"
      exit 1
    fi
  done
done
