#!/usr/bin/env bash
set -Eeuo pipefail

# Ze-Ri STT + TTS + VLM launcher
# Starts:
#   1) TTS node
#   2) STT launch
#   3) VLM-STT bridge
#
# This script does NOT start camera or VLA stack.
# Start those separately when testing real handoff:
#   SERIAL=_944122071303 bash scripts/realsense_pointcloud.sh
#   bash scripts/start_vla.sh

ZERI_ROOT="${ZERI_ROOT:-$HOME/ze-ri}"
SOURCE_FILE="${SOURCE_FILE:-$ZERI_ROOT/source_zeri.sh}"
APP_VENV="${APP_VENV:-$ZERI_ROOT/.venv}"
STT_VENV="${STT_VENV:-$ZERI_ROOT/.venv_stt}"
APP_PYTHON="${APP_PYTHON:-$APP_VENV/bin/python}"
STT_PYTHON="${STT_PYTHON:-$STT_VENV/bin/python}"
STT_NODE_FILE="${STT_NODE_FILE:-$ZERI_ROOT/ros2_ws/src/nb_voice_stt/nb_voice_stt/stt_node.py}"

LOG_DIR="${LOG_DIR:-$ZERI_ROOT/logs/stt_tts_vlm_$(date +%Y%m%d_%H%M%S)}"
mkdir -p "$LOG_DIR"

# -----------------------------
# STT defaults
# -----------------------------
STT_MODEL_DIR="${STT_MODEL_DIR:-$ZERI_ROOT/models/sensevoice_ko}"
STT_AUDIO_DEVICE="${STT_AUDIO_DEVICE:--1}"
STT_CHANNELS="${STT_CHANNELS:-6}"
STT_USE_CHANNEL_INDEX="${STT_USE_CHANNEL_INDEX:-0}"
STT_SAMPLE_RATE="${STT_SAMPLE_RATE:-16000}"
STT_DEVICE="${STT_DEVICE:-cpu}"

# -----------------------------
# VLM defaults
# -----------------------------
RGB_TOPIC="${RGB_TOPIC:-/camera/camera/color/image_raw}"
DEPTH_TOPIC="${DEPTH_TOPIC:-/camera/camera/aligned_depth_to_color/image_raw}"
STT_TOPIC="${STT_TOPIC:-/stt/text}"
ROBOT_SPEECH_TOPIC="${ROBOT_SPEECH_TOPIC:-/zeri/vlm/robot_speech}"
TTS_STATUS_TOPIC="${TTS_STATUS_TOPIC:-/zeri/tts/status}"
MISSION_EVENT_TOPIC="${MISSION_EVENT_TOPIC:-/zeri/mission/event}"
BEHAVIOR_COMMAND_TOPIC="${BEHAVIOR_COMMAND_TOPIC:-/zeri/behavior/command}"
COMPLETE_BEHAVIOR_ON_VLA_SUCCESS="${COMPLETE_BEHAVIOR_ON_VLA_SUCCESS:-true}"

VLA_TIMEOUT_SEC="${VLA_TIMEOUT_SEC:-60.0}"
VLA_TASK_DURATION_SEC="${VLA_TASK_DURATION_SEC:-20.0}"
HANDOFF_VERIFY_TIMEOUT_SEC="${HANDOFF_VERIFY_TIMEOUT_SEC:-10.0}"
HOME_RETURN_TIMEOUT_SEC="${HOME_RETURN_TIMEOUT_SEC:-5.0}"
TTS_MAX_WAIT_SEC="${TTS_MAX_WAIT_SEC:-20.0}"
STT_BLOCK_AFTER_TTS_SEC="${STT_BLOCK_AFTER_TTS_SEC:-0.8}"
STT_GATE_MODE="${STT_GATE_MODE:-all}"
USE_VAD_GATE="${USE_VAD_GATE:-false}"
ENABLE_VLA="${ENABLE_VLA:-true}"
ENABLE_MISSION_EVENTS="${ENABLE_MISSION_EVENTS:-true}"

LEFT_WRIST_SNAPSHOT_TOPIC="${LEFT_WRIST_SNAPSHOT_TOPIC:-/zeri/vla/left/wrist_snapshot}"
RIGHT_WRIST_SNAPSHOT_TOPIC="${RIGHT_WRIST_SNAPSHOT_TOPIC:-/zeri/vla/right/wrist_snapshot}"
ARM_HOME_REQUEST_TOPIC="${ARM_HOME_REQUEST_TOPIC:-/zeri/arm/home_request}"

PIDS=()
NAMES=()

cleanup() {
  local code=$?
  trap - INT TERM EXIT
  echo
  echo "[INFO] stopping STT/TTS/VLM processes..."
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

source_common() {
  if [[ ! -f "$SOURCE_FILE" ]]; then
    echo "[ERROR] source file not found: $SOURCE_FILE" >&2
    exit 1
  fi

  # Some ROS setup scripts reference unset variables. Avoid set -u failure during source.
  set +u
  source "$SOURCE_FILE"
  set -u
}

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

cd "$ZERI_ROOT"
source_common

if [[ ! -x "$APP_PYTHON" ]]; then
  echo "[ERROR] app python not found or not executable: $APP_PYTHON" >&2
  exit 1
fi
if [[ ! -x "$STT_PYTHON" ]]; then
  echo "[ERROR] STT python not found or not executable: $STT_PYTHON" >&2
  exit 1
fi
if [[ ! -f "$STT_NODE_FILE" ]]; then
  echo "[ERROR] STT node file not found: $STT_NODE_FILE" >&2
  exit 1
fi
if [[ ! -d "$STT_MODEL_DIR" ]]; then
  echo "[WARN] STT model dir not found now: $STT_MODEL_DIR" >&2
fi

# Ensure C-Media USB speaker is the default output if available.
if command -v zeri_set_speaker >/dev/null 2>&1; then
  zeri_set_speaker "${SPEAKER_VOLUME:-70}" || true
elif command -v pactl >/dev/null 2>&1; then
  SINK="alsa_output.usb-C-Media_Electronics_Inc._USB_Audio_Device-00.analog-stereo"
  if pactl list short sinks | grep -q "$SINK"; then
    pactl set-default-sink "$SINK" >/dev/null 2>&1 || true
    pactl set-sink-mute "$SINK" 0 >/dev/null 2>&1 || true
    pactl set-sink-volume "$SINK" "${SPEAKER_VOLUME:-70}%" >/dev/null 2>&1 || true
  fi
fi

start_bg tts_edge_node \
  bash -lc "cd '$ZERI_ROOT' && \
    set +u && source '$SOURCE_FILE' && source '$APP_VENV/bin/activate' && set -u && \
    exec '$APP_PYTHON' src/lerobot/vlm_agent/tts_edge_node.py"

sleep 1

start_bg stt_node \
  bash -lc "cd '$ZERI_ROOT' && \
    set +u && \
    source '$SOURCE_FILE' && \
    if declare -F deactivate >/dev/null 2>&1; then deactivate || true; fi && \
    source '$STT_VENV/bin/activate' && \
    set -u && \
    export PYTHONPATH='$ZERI_ROOT/ros2_ws/src/nb_voice_stt':\"\${PYTHONPATH:-}\" && \
    exec '$STT_PYTHON' '$STT_NODE_FILE' --ros-args \
      -r __node:=sensevoice_stt_node \
      -p model_dir:='$STT_MODEL_DIR' \
      -p audio_device:='$STT_AUDIO_DEVICE' \
      -p channels:='$STT_CHANNELS' \
      -p use_channel_index:='$STT_USE_CHANNEL_INDEX' \
      -p sample_rate:='$STT_SAMPLE_RATE' \
      -p device:='$STT_DEVICE'"

sleep 2

start_bg vlm_stt_bridge_node \
  bash -lc "cd '$ZERI_ROOT' && \
    set +u && source '$SOURCE_FILE' && source '$APP_VENV/bin/activate' && set -u && \
    exec '$APP_PYTHON' src/lerobot/vlm_agent/vlm_stt_bridge_node.py --ros-args \
    -p rgb_topic:='$RGB_TOPIC' \
    -p depth_topic:='$DEPTH_TOPIC' \
    -p stt_topic:='$STT_TOPIC' \
    -p robot_speech_topic:='$ROBOT_SPEECH_TOPIC' \
    -p tts_status_topic:='$TTS_STATUS_TOPIC' \
    -p mission_event_topic:='$MISSION_EVENT_TOPIC' \
    -p behavior_command_topic:='$BEHAVIOR_COMMAND_TOPIC' \
    -p stt_gate_mode:='$STT_GATE_MODE' \
    -p use_vad_gate:='$USE_VAD_GATE' \
    -p enable_vla:='$ENABLE_VLA' \
    -p enable_mission_events:='$ENABLE_MISSION_EVENTS' \
    -p complete_behavior_on_vla_success:='$COMPLETE_BEHAVIOR_ON_VLA_SUCCESS' \
    -p vla_timeout_sec:='$VLA_TIMEOUT_SEC' \
    -p vla_default_task_duration_sec:='$VLA_TASK_DURATION_SEC' \
    -p left_wrist_snapshot_topic:='$LEFT_WRIST_SNAPSHOT_TOPIC' \
    -p right_wrist_snapshot_topic:='$RIGHT_WRIST_SNAPSHOT_TOPIC' \
    -p arm_home_request_topic:='$ARM_HOME_REQUEST_TOPIC' \
    -p handoff_verify_timeout_sec:='$HANDOFF_VERIFY_TIMEOUT_SEC' \
    -p home_return_timeout_sec:='$HOME_RETURN_TIMEOUT_SEC' \
    -p tts_max_wait_sec:='$TTS_MAX_WAIT_SEC' \
    -p stt_block_after_tts_sec:='$STT_BLOCK_AFTER_TTS_SEC'"

sleep 2

echo
echo "[READY] STT + TTS + VLM started."
echo "[INFO] logs: $LOG_DIR"
echo
echo "Monitor:"
echo "  ros2 topic echo /stt/text"
echo "  ros2 topic echo /zeri/vlm/decision"
echo "  ros2 topic echo /zeri/mission/event"
echo "  ros2 topic echo /zeri/behavior/command"
echo "  ros2 topic echo /zeri/vla/status"
echo "  ros2 topic hz /zeri/vla/left/wrist_snapshot"
echo "  ros2 topic hz /zeri/vla/right/wrist_snapshot"
echo
echo "Press Ctrl+C to stop all three processes."

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
