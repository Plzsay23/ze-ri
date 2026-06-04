#!/usr/bin/env bash
set -Eeuo pipefail

ZERI_ROOT="${ZERI_ROOT:-$HOME/ze-ri}"

ARM="left"
MODEL_PATH="$ZERI_ROOT/models/hand_yolo.pt"
DEVICE="0"
CONF="0.5"
IMGSZ="640"
STABLE_FRAMES_REQUIRED="1"
STABLE_HAND_SEC_REQUIRED="0.0"

# Tunable timing params
RELEASE_DELAY_SEC="1.0"
HAND_LOST_GRACE_SEC="1.0"
MIN_RUN_SEC_BEFORE_RELEASE="8.0"

ALLOW_RELEASE_WHILE_RUNNING="true"
ROI="[0.0, 0.0, 1.0, 1.0]"
SNAPSHOT_HZ="5"
VLA_START_WAIT_SEC="3"

usage() {
  cat <<EOF
Usage:
  $0 [options]

Main timing options:
  --min-run-sec SEC              VLA 시작 후 release 허용까지 대기 시간. default: ${MIN_RUN_SEC_BEFORE_RELEASE}
  --release-delay-sec SEC        손을 처음 본 뒤 실제 release까지 대기 시간. default: ${RELEASE_DELAY_SEC}
  --hand-lost-grace-sec SEC      YOLO가 손을 잠깐 놓쳐도 timer 유지하는 시간. default: ${HAND_LOST_GRACE_SEC}

YOLO options:
  --conf VALUE                   YOLO confidence threshold. default: ${CONF}
  --imgsz SIZE                   YOLO image size. default: ${IMGSZ}
  --stable-frames N              필요 검출 frame 수. default: ${STABLE_FRAMES_REQUIRED}
  --stable-hand-sec SEC          연속 손 검출 시간 gate. delay 방식 쓸 때는 보통 0.0. default: ${STABLE_HAND_SEC_REQUIRED}
  --model PATH                   YOLO model path. default: ${MODEL_PATH}
  --device DEVICE                YOLO device. default: ${DEVICE}
  --roi JSON                     ROI normalized xyxy. default: ${ROI}

VLA options:
  --arm left|right               target arm. default: ${ARM}
  --snapshot-hz HZ               /handoff_image publish rate. default: ${SNAPSHOT_HZ}
  --vla-start-wait-sec SEC       VLA stack 실행 후 YOLO 시작 전 대기. default: ${VLA_START_WAIT_SEC}

Examples:
  $0
  $0 --min-run-sec 8.5 --release-delay-sec 1.0 --hand-lost-grace-sec 1.0
  $0 --conf 0.45 --min-run-sec 8.0 --release-delay-sec 1.5
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --min-run-sec|--min_run_sec|--min-run-sec-before-release)
      MIN_RUN_SEC_BEFORE_RELEASE="$2"; shift 2 ;;
    --release-delay-sec|--release_delay_sec)
      RELEASE_DELAY_SEC="$2"; shift 2 ;;
    --hand-lost-grace-sec|--hand_lost_grace_sec)
      HAND_LOST_GRACE_SEC="$2"; shift 2 ;;

    --conf)
      CONF="$2"; shift 2 ;;
    --imgsz)
      IMGSZ="$2"; shift 2 ;;
    --stable-frames|--stable_frames_required)
      STABLE_FRAMES_REQUIRED="$2"; shift 2 ;;
    --stable-hand-sec|--stable_hand_sec_required)
      STABLE_HAND_SEC_REQUIRED="$2"; shift 2 ;;
    --model|--model-path)
      MODEL_PATH="$2"; shift 2 ;;
    --device)
      DEVICE="$2"; shift 2 ;;
    --roi)
      ROI="$2"; shift 2 ;;

    --arm)
      ARM="$2"; shift 2 ;;
    --snapshot-hz)
      SNAPSHOT_HZ="$2"; shift 2 ;;
    --vla-start-wait-sec)
      VLA_START_WAIT_SEC="$2"; shift 2 ;;

    -h|--help)
      usage; exit 0 ;;
    *)
      echo "[ERR] unknown option: $1" >&2
      usage
      exit 2 ;;
  esac
done

if [[ "$ARM" != "left" && "$ARM" != "right" ]]; then
  echo "[ERR] --arm must be left or right" >&2
  exit 2
fi

if [[ ! -f "$MODEL_PATH" ]]; then
  echo "[ERR] YOLO model not found: $MODEL_PATH" >&2
  exit 1
fi

cd "$ZERI_ROOT"
source "$ZERI_ROOT/source_zeri.sh"

# Existing VLA handoff settings
export LEFT_GRIPPER_OPEN_ACTION_VALUE="${LEFT_GRIPPER_OPEN_ACTION_VALUE:-100}"
export HANDOFF_DETECTOR_TYPE="${HANDOFF_DETECTOR_TYPE:-none}"
export HANDOFF_REQUIRE_REFERENCE="${HANDOFF_REQUIRE_REFERENCE:-false}"
export HANDOFF_REQUIRE_GRIPPER_CLOSED="${HANDOFF_REQUIRE_GRIPPER_CLOSED:-false}"

# Required for YOLO sidecar image stream
export ZERI_PUBLISH_SNAPSHOT_ONLY_AFTER_HANDOFF="${ZERI_PUBLISH_SNAPSHOT_ONLY_AFTER_HANDOFF:-false}"
export ZERI_HANDOFF_SNAPSHOT_HZ="$SNAPSHOT_HZ"

if [[ "$ARM" == "left" ]]; then
  export ZERI_HANDOFF_CAMERA_KEY="${ZERI_HANDOFF_CAMERA_KEY:-cam_left}"
else
  export ZERI_HANDOFF_CAMERA_KEY="${ZERI_HANDOFF_CAMERA_KEY:-cam_right}"
fi

echo "===================================================="
echo " Ze-Ri VLA + YOLO Auto Handoff"
echo "===================================================="
echo "ARM=$ARM"
echo "MODEL_PATH=$MODEL_PATH"
echo "CONF=$CONF"
echo "IMGSZ=$IMGSZ"
echo "STABLE_FRAMES_REQUIRED=$STABLE_FRAMES_REQUIRED"
echo "STABLE_HAND_SEC_REQUIRED=$STABLE_HAND_SEC_REQUIRED"
echo "RELEASE_DELAY_SEC=$RELEASE_DELAY_SEC"
echo "HAND_LOST_GRACE_SEC=$HAND_LOST_GRACE_SEC"
echo "MIN_RUN_SEC_BEFORE_RELEASE=$MIN_RUN_SEC_BEFORE_RELEASE"
echo "SNAPSHOT_HZ=$SNAPSHOT_HZ"
echo "ZERI_HANDOFF_CAMERA_KEY=$ZERI_HANDOFF_CAMERA_KEY"
echo "===================================================="

VLA_PID=""
cleanup() {
  echo
  echo "[INFO] shutting down wrapper..."

  pkill -f "yolo_handoff_auto_release_node" 2>/dev/null || true

  if [[ -n "${VLA_PID}" ]] && kill -0 "$VLA_PID" 2>/dev/null; then
    kill "$VLA_PID" 2>/dev/null || true
  fi
}
trap cleanup INT TERM EXIT

echo "[INFO] starting VLA stack..."
"$ZERI_ROOT/scripts/start_vla_handoff.sh" &
VLA_PID=$!

echo "[INFO] waiting ${VLA_START_WAIT_SEC}s before starting YOLO sidecar..."
sleep "$VLA_START_WAIT_SEC"

echo "[INFO] starting YOLO auto-release sidecar..."
python -u "$ZERI_ROOT/src/lerobot/vlm_agent/yolo_handoff_auto_release_node.py" \
  --ros-args \
  -p arm:="$ARM" \
  -p model_path:="$MODEL_PATH" \
  -p device:="$DEVICE" \
  -p conf:="$CONF" \
  -p imgsz:="$IMGSZ" \
  -p stable_frames_required:="$STABLE_FRAMES_REQUIRED" \
  -p stable_hand_sec_required:="$STABLE_HAND_SEC_REQUIRED" \
  -p release_delay_sec:="$RELEASE_DELAY_SEC" \
  -p hand_lost_grace_sec:="$HAND_LOST_GRACE_SEC" \
  -p min_run_sec_before_release:="$MIN_RUN_SEC_BEFORE_RELEASE" \
  -p allow_release_while_running:="$ALLOW_RELEASE_WHILE_RUNNING" \
  -p roi:="$ROI"
