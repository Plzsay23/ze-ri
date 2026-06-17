#!/usr/bin/env bash
set -Eeuo pipefail

ZERI_ROOT="${ZERI_ROOT:-$HOME/ze-ri}"

# 기본값: 양팔 모두 YOLO handoff 감시
ARM="both"

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
VLA_START_WAIT_SEC="14"

KEEP_VLA_ON_YOLO_EXIT="false"
SIDECAR_ONLY="false"
RESTART_VLA="false"

usage() {
  cat <<EOF
Usage:
  $0 [options]

Main options:
  --arm left|right|both           YOLO handoff 감시 arm. default: ${ARM}
                                  both이면 left/right YOLO sidecar를 둘 다 실행한다.
  --sidecar-only                  기존 VLA stack은 건드리지 않고 YOLO sidecar만 실행한다.
                                  이미 left/right client가 떠 있어야 한다.
  --restart-vla                   기존 VLA/client/server/router/YOLO를 정리한 뒤 새 VLA stack을 시작한다.

Main timing options:
  --min-run-sec SEC               VLA 시작 후 release 허용까지 대기 시간. default: ${MIN_RUN_SEC_BEFORE_RELEASE}
  --release-delay-sec SEC         손을 처음 본 뒤 실제 release까지 대기 시간. default: ${RELEASE_DELAY_SEC}
  --hand-lost-grace-sec SEC       YOLO가 손을 잠깐 놓쳐도 timer 유지하는 시간. default: ${HAND_LOST_GRACE_SEC}

YOLO options:
  --conf VALUE                    YOLO confidence threshold. default: ${CONF}
  --imgsz SIZE                    YOLO image size. default: ${IMGSZ}
  --stable-frames N               필요 검출 frame 수. default: ${STABLE_FRAMES_REQUIRED}
  --stable-hand-sec SEC           연속 손 검출 시간 gate. delay 방식 쓸 때는 보통 0.0. default: ${STABLE_HAND_SEC_REQUIRED}
  --model PATH                    YOLO model path. default: ${MODEL_PATH}
  --device DEVICE                 YOLO device. default: ${DEVICE}
  --roi JSON                      ROI normalized xyxy. default: ${ROI}

VLA options:
  --snapshot-hz HZ                /handoff_image publish rate. default: ${SNAPSHOT_HZ}
  --vla-start-wait-sec SEC        VLA stack 실행 후 YOLO 시작 전 대기. default: ${VLA_START_WAIT_SEC}
  --keep-vla-on-yolo-exit         YOLO sidecar 종료 시 이 wrapper가 시작한 VLA stack은 죽이지 않는다.

Examples:
  # 완전 새로 시작. 기존 중복 client까지 정리한다.
  $0 --restart-vla

  # VLA stack이 이미 정상 실행 중이면 YOLO만 붙인다.
  $0 --sidecar-only

  $0 --restart-vla --arm both --min-run-sec 8.5 --release-delay-sec 1.0
  $0 --sidecar-only --arm left --conf 0.5
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --arm)
      ARM="$2"; shift 2 ;;

    --sidecar-only)
      SIDECAR_ONLY="true"; shift 1 ;;
    --restart-vla)
      RESTART_VLA="true"; shift 1 ;;

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

    --snapshot-hz)
      SNAPSHOT_HZ="$2"; shift 2 ;;
    --vla-start-wait-sec)
      VLA_START_WAIT_SEC="$2"; shift 2 ;;
    --keep-vla-on-yolo-exit)
      KEEP_VLA_ON_YOLO_EXIT="true"; shift 1 ;;

    -h|--help)
      usage; exit 0 ;;
    *)
      echo "[ERR] unknown option: $1" >&2
      usage
      exit 2 ;;
  esac
done

if [[ "$ARM" != "left" && "$ARM" != "right" && "$ARM" != "both" ]]; then
  echo "[ERR] --arm must be left, right, or both" >&2
  exit 2
fi

if [[ "$SIDECAR_ONLY" == "true" && "$RESTART_VLA" == "true" ]]; then
  echo "[ERR] --sidecar-only and --restart-vla cannot be used together." >&2
  exit 2
fi

if [[ ! -f "$MODEL_PATH" ]]; then
  echo "[ERR] YOLO model not found: $MODEL_PATH" >&2
  exit 1
fi

cd "$ZERI_ROOT"

# source_zeri.sh 내부에서 set -u와 충돌하는 경우 방지
set +u
source "$ZERI_ROOT/source_zeri.sh"
set -u

# Existing VLA handoff settings
export LEFT_GRIPPER_OPEN_ACTION_VALUE="${LEFT_GRIPPER_OPEN_ACTION_VALUE:-100}"
export RIGHT_GRIPPER_OPEN_ACTION_VALUE="${RIGHT_GRIPPER_OPEN_ACTION_VALUE:-100}"

export HANDOFF_DETECTOR_TYPE="${HANDOFF_DETECTOR_TYPE:-none}"
export HANDOFF_REQUIRE_REFERENCE="${HANDOFF_REQUIRE_REFERENCE:-false}"
export HANDOFF_REQUIRE_GRIPPER_CLOSED="${HANDOFF_REQUIRE_GRIPPER_CLOSED:-false}"

# Required for YOLO sidecar image stream
export ZERI_PUBLISH_SNAPSHOT_ONLY_AFTER_HANDOFF="${ZERI_PUBLISH_SNAPSHOT_ONLY_AFTER_HANDOFF:-false}"
export ZERI_HANDOFF_SNAPSHOT_HZ="$SNAPSHOT_HZ"

# start_vla_handoff.sh 쪽에서 이 값을 사용하더라도 한쪽으로 강제하지 않도록 기본값만 둔다.
# 실제 handoff_image가 left/right 둘 다 나오면 YOLO sidecar는 arm별 topic을 직접 구독한다.
export ZERI_HANDOFF_CAMERA_KEY="${ZERI_HANDOFF_CAMERA_KEY:-cam_left}"

VLA_PID=""
YOLO_LEFT_PID=""
YOLO_RIGHT_PID=""
STARTED_VLA="false"

count_left_clients() {
  # pgrep returns 1 when there are no matches. With set -e + pipefail, that would
  # abort the wrapper before we can print a useful error. Force a successful pipe.
  { pgrep -af "robot_client_ros_multi_gate.*--zeri_client_name=left" 2>/dev/null || true; } | wc -l
}

count_right_clients() {
  { pgrep -af "robot_client_ros_multi_gate.*--zeri_client_name=right" 2>/dev/null || true; } | wc -l
}

vla_stack_running() {
  pgrep -f "robot_client_ros_multi_gate.*--zeri_client_name=left" >/dev/null 2>&1 \
    || pgrep -f "robot_client_ros_multi_gate.*--zeri_client_name=right" >/dev/null 2>&1 \
    || pgrep -f "vla_task_router_multi_node" >/dev/null 2>&1 \
    || pgrep -f "vla_handoff_supervisor_node" >/dev/null 2>&1 \
    || pgrep -f "lerobot.async_inference.policy_server" >/dev/null 2>&1 \
    || pgrep -f "start_vla_handoff.sh" >/dev/null 2>&1
}

kill_pids_by_pattern() {
  local pattern="$1"
  local label="$2"
  local pids=""

  # Exclude this wrapper process and its immediate shell from accidental self-kill.
  pids="$(pgrep -f "$pattern" 2>/dev/null | awk -v self="$$" -v bashpid="${BASHPID:-$$}" '$1 != self && $1 != bashpid {print $1}' | sort -u | xargs || true)"

  if [[ -z "$pids" ]]; then
    return 0
  fi

  echo "[KILL] $label: $pids"
  kill -TERM $pids 2>/dev/null || true
}

kill_pids_by_pattern_force() {
  local pattern="$1"
  local label="$2"
  local pids=""

  pids="$(pgrep -f "$pattern" 2>/dev/null | awk -v self="$$" -v bashpid="${BASHPID:-$$}" '$1 != self && $1 != bashpid {print $1}' | sort -u | xargs || true)"

  if [[ -z "$pids" ]]; then
    return 0
  fi

  echo "[KILL -9] $label: $pids"
  kill -KILL $pids 2>/dev/null || true
}

kill_existing_vla_stack() {
  echo "[INFO] killing existing VLA/YOLO stack before restart..."

  # First, stop children before their launchers. Some clients can hang in camera/motor cleanup,
  # so TERM is followed by KILL if they remain alive.
  kill_pids_by_pattern "yolo_handoff_auto_release_node" "YOLO sidecar"
  kill_pids_by_pattern "robot_client_ros_multi_gate_raw_home_handoff" "VLA robot client handoff"
  kill_pids_by_pattern "robot_client_ros_multi_gate" "VLA robot client"
  kill_pids_by_pattern "vla_task_router_multi_node" "VLA router"
  kill_pids_by_pattern "vla_handoff_supervisor_node" "VLA handoff supervisor"
  kill_pids_by_pattern "lerobot.async_inference.policy_server" "policy server"
  kill_pids_by_pattern "start_vla_handoff.sh" "VLA launcher"

  sleep 3

  # Force-kill stale processes that ignored TERM or got stuck during hardware disconnect.
  kill_pids_by_pattern_force "yolo_handoff_auto_release_node" "YOLO sidecar"
  kill_pids_by_pattern_force "robot_client_ros_multi_gate_raw_home_handoff" "VLA robot client handoff"
  kill_pids_by_pattern_force "robot_client_ros_multi_gate" "VLA robot client"
  kill_pids_by_pattern_force "vla_task_router_multi_node" "VLA router"
  kill_pids_by_pattern_force "vla_handoff_supervisor_node" "VLA handoff supervisor"
  kill_pids_by_pattern_force "lerobot.async_inference.policy_server" "policy server"
  kill_pids_by_pattern_force "start_vla_handoff.sh" "VLA launcher"

  sleep 1

  if vla_stack_running; then
    echo "[ERR] stale VLA stack still alive after TERM/KILL." >&2
    print_vla_process_summary >&2
    echo "[HINT] Check if another user/session/supervisor is restarting it, or kill listed PIDs manually." >&2
    exit 1
  fi
}

print_vla_process_summary() {
  echo "==== VLA process summary ===="
  echo "left client count:  $(count_left_clients)"
  echo "right client count: $(count_right_clients)"
  pgrep -af "robot_client_ros_multi_gate.*--zeri_client_name=(left|right)" 2>/dev/null || true
  ss -ltnp 2>/dev/null | grep -E ":8081|:8082" || true
  echo "============================="
}


wait_for_required_clients() {
  local deadline now left_count right_count
  deadline=$(( $(date +%s) + ${VLA_START_WAIT_SEC%.*} ))

  echo "[INFO] waiting up to ${VLA_START_WAIT_SEC}s for required VLA clients..."

  while true; do
    left_count="$(count_left_clients | tr -d ' ')"
    right_count="$(count_right_clients | tr -d ' ')"

    echo "[WAIT] left clients=${left_count}, right clients=${right_count}"

    if [[ "$ARM" == "left" && "$left_count" -eq 1 ]]; then
      return 0
    fi
    if [[ "$ARM" == "right" && "$right_count" -eq 1 ]]; then
      return 0
    fi
    if [[ "$ARM" == "both" && "$left_count" -eq 1 && "$right_count" -eq 1 ]]; then
      return 0
    fi

    now=$(date +%s)
    if [[ "$now" -ge "$deadline" ]]; then
      echo "[ERR] timed out waiting for VLA clients." >&2
      print_vla_process_summary >&2
      return 1
    fi

    sleep 1
  done
}

validate_required_clients() {
  local left_count right_count bad
  left_count="$(count_left_clients)"
  right_count="$(count_right_clients)"
  bad="false"

  if [[ "$ARM" == "left" || "$ARM" == "both" ]]; then
    if [[ "$left_count" -ne 1 ]]; then
      echo "[ERR] expected exactly 1 left VLA client, got $left_count" >&2
      bad="true"
    fi
  fi

  if [[ "$ARM" == "right" || "$ARM" == "both" ]]; then
    if [[ "$right_count" -ne 1 ]]; then
      echo "[ERR] expected exactly 1 right VLA client, got $right_count" >&2
      bad="true"
    fi
  fi

  if [[ "$bad" == "true" ]]; then
    print_vla_process_summary >&2
    exit 1
  fi
}

cleanup() {
  echo
  echo "[INFO] shutting down wrapper..."

  if [[ -n "${YOLO_LEFT_PID:-}" ]] && kill -0 "$YOLO_LEFT_PID" 2>/dev/null; then
    kill "$YOLO_LEFT_PID" 2>/dev/null || true
  fi

  if [[ -n "${YOLO_RIGHT_PID:-}" ]] && kill -0 "$YOLO_RIGHT_PID" 2>/dev/null; then
    kill "$YOLO_RIGHT_PID" 2>/dev/null || true
  fi

  # 이 wrapper가 직접 시작한 VLA만 종료한다. sidecar-only 모드에서 기존 VLA를 죽이지 않는다.
  if [[ "$STARTED_VLA" == "true" && "$KEEP_VLA_ON_YOLO_EXIT" != "true" ]]; then
    if [[ -n "${VLA_PID:-}" ]] && kill -0 "$VLA_PID" 2>/dev/null; then
      kill "$VLA_PID" 2>/dev/null || true
    fi
  fi
}

trap cleanup INT TERM EXIT

start_yolo_sidecar() {
  local arm="$1"

  echo "[INFO] starting YOLO auto-release sidecar: arm=$arm"

  python -u "$ZERI_ROOT/src/lerobot/vlm_agent/yolo_handoff_auto_release_node.py" \
    --ros-args \
    -p arm:="$arm" \
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
    -p roi:="$ROI" &
}

echo "===================================================="
echo " Ze-Ri VLA + YOLO Auto Handoff"
echo "===================================================="
echo "ARM=$ARM"
echo "MODEL_PATH=$MODEL_PATH"
echo "DEVICE=$DEVICE"
echo "CONF=$CONF"
echo "IMGSZ=$IMGSZ"
echo "STABLE_FRAMES_REQUIRED=$STABLE_FRAMES_REQUIRED"
echo "STABLE_HAND_SEC_REQUIRED=$STABLE_HAND_SEC_REQUIRED"
echo "RELEASE_DELAY_SEC=$RELEASE_DELAY_SEC"
echo "HAND_LOST_GRACE_SEC=$HAND_LOST_GRACE_SEC"
echo "MIN_RUN_SEC_BEFORE_RELEASE=$MIN_RUN_SEC_BEFORE_RELEASE"
echo "ALLOW_RELEASE_WHILE_RUNNING=$ALLOW_RELEASE_WHILE_RUNNING"
echo "ROI=$ROI"
echo "SNAPSHOT_HZ=$SNAPSHOT_HZ"
echo "VLA_START_WAIT_SEC=$VLA_START_WAIT_SEC"
echo "KEEP_VLA_ON_YOLO_EXIT=$KEEP_VLA_ON_YOLO_EXIT"
echo "SIDECAR_ONLY=$SIDECAR_ONLY"
echo "RESTART_VLA=$RESTART_VLA"
echo "ZERI_PUBLISH_SNAPSHOT_ONLY_AFTER_HANDOFF=$ZERI_PUBLISH_SNAPSHOT_ONLY_AFTER_HANDOFF"
echo "ZERI_HANDOFF_SNAPSHOT_HZ=$ZERI_HANDOFF_SNAPSHOT_HZ"
echo "ZERI_HANDOFF_CAMERA_KEY=$ZERI_HANDOFF_CAMERA_KEY"
echo "===================================================="

if [[ "$RESTART_VLA" == "true" ]]; then
  kill_existing_vla_stack
fi

if [[ "$SIDECAR_ONLY" == "true" ]]; then
  echo "[INFO] sidecar-only mode: not starting VLA stack."
  if ! vla_stack_running; then
    echo "[ERR] --sidecar-only requested, but no VLA stack appears to be running." >&2
    print_vla_process_summary >&2
    exit 1
  fi
  validate_required_clients
else
  if vla_stack_running; then
    echo "[ERR] existing VLA stack detected. Starting another stack would duplicate arm clients." >&2
    echo "      Use one of:" >&2
    echo "        $0 --restart-vla" >&2
    echo "        $0 --sidecar-only" >&2
    print_vla_process_summary >&2
    exit 1
  fi

  echo "[INFO] starting VLA stack via scripts/start_vla_handoff.sh ..."
  "$ZERI_ROOT/scripts/start_vla_handoff.sh" &
  VLA_PID=$!
  STARTED_VLA="true"

  wait_for_required_clients
  validate_required_clients
fi

case "$ARM" in
  left)
    start_yolo_sidecar left
    YOLO_LEFT_PID=$!
    wait "$YOLO_LEFT_PID"
    ;;
  right)
    start_yolo_sidecar right
    YOLO_RIGHT_PID=$!
    wait "$YOLO_RIGHT_PID"
    ;;
  both)
    start_yolo_sidecar left
    YOLO_LEFT_PID=$!

    start_yolo_sidecar right
    YOLO_RIGHT_PID=$!

    wait "$YOLO_LEFT_PID" "$YOLO_RIGHT_PID"
    ;;
esac
