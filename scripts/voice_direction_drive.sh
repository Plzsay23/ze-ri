#!/usr/bin/env bash
set -Eeuo pipefail

ZERI_ROOT="${ZERI_ROOT:-$HOME/ze-ri}"
SOURCE_FILE="${SOURCE_FILE:-$ZERI_ROOT/source_zeri.sh}"

LOG_DIR="${LOG_DIR:-$ZERI_ROOT/logs/voice_direction_drive_$(date +%Y%m%d_%H%M%S)}"
mkdir -p "$LOG_DIR"

FRONT_ANGLE_DEG="${FRONT_ANGLE_DEG:-245.0}"
INVERT_TURN="${INVERT_TURN:-false}"
TARGET_HOLD_SEC="${TARGET_HOLD_SEC:-4.0}"
FORWARD_SPEED="${FORWARD_SPEED:-0.16}"
ALLOW_ARC_MOTION="${ALLOW_ARC_MOTION:-false}"
CMD_TOPIC="${CMD_TOPIC:-/cmd_vel_raw}"

PIDS=()
NAMES=()

cleanup() {
  local code=$?
  trap - INT TERM EXIT
  echo
  echo "[Ze-Ri] stopping voice direction nodes..."
  for pid in "${PIDS[@]:-}"; do
    if kill -0 "$pid" 2>/dev/null; then
      kill "$pid" 2>/dev/null || true
    fi
  done
  sleep 0.5
  for pid in "${PIDS[@]:-}"; do
    if kill -0 "$pid" 2>/dev/null; then
      kill -9 "$pid" 2>/dev/null || true
    fi
  done
  echo "[Ze-Ri] logs saved in: $LOG_DIR"
  exit "$code"
}
trap cleanup INT TERM EXIT

source_common() {
  if [[ ! -f "$SOURCE_FILE" ]]; then
    echo "[ERROR] source file not found: $SOURCE_FILE" >&2
    exit 1
  fi

  set +u
  source "$SOURCE_FILE"
  set -u
}

start_node() {
  local name="$1"
  shift
  local log="$LOG_DIR/${name}.log"

  echo "[START] $name"
  echo "        log: $log"
  "$@" > "$log" 2>&1 &
  local pid=$!
  PIDS+=("$pid")
  NAMES+=("$name")
  echo "        pid: $pid"
  sleep 0.8
}

cd "$ZERI_ROOT"
source_common

echo "[Ze-Ri Voice Direction Drive]"
echo "  FRONT_ANGLE_DEG=$FRONT_ANGLE_DEG"
echo "  INVERT_TURN=$INVERT_TURN"
echo "  TARGET_HOLD_SEC=$TARGET_HOLD_SEC"
echo "  FORWARD_SPEED=$FORWARD_SPEED"
echo "  ALLOW_ARC_MOTION=$ALLOW_ARC_MOTION"
echo "  CMD_TOPIC=$CMD_TOPIC"
echo "  LOG_DIR=$LOG_DIR"
echo
echo "[INFO] Start scripts/slam_depth_drive.sh in another terminal first."
echo

start_node "01_respeaker_vad_doa" \
  ros2 run zeri_voice respeaker_vad_doa_node

start_node "02_voice_follow_cmd" \
  ros2 run zeri_voice voice_follow_cmd_node --ros-args \
    -p cmd_topic:="$CMD_TOPIC" \
    -p front_angle_deg:="$FRONT_ANGLE_DEG" \
    -p invert_turn:="$INVERT_TURN" \
    -p target_hold_sec:="$TARGET_HOLD_SEC" \
    -p forward_speed:="$FORWARD_SPEED" \
    -p allow_arc_motion:="$ALLOW_ARC_MOTION"

echo
echo "[READY] Voice direction drive is publishing to $CMD_TOPIC."
echo "Monitor:"
echo "  ros2 topic echo /zeri/audio/state"
echo "  ros2 topic echo /zeri/audio/follow_state"
echo "  ros2 topic echo $CMD_TOPIC"
echo
echo "Press Ctrl+C to stop voice direction nodes."

while true; do
  sleep 1
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
