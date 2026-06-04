#!/usr/bin/env bash
set -euo pipefail

ZERI_ROOT="${ZERI_ROOT:-$HOME/ze-ri}"
cd "$ZERI_ROOT"

# 중요: source_zeri.sh 쓰지 않음.
# source_zeri.sh는 메인 .venv를 activate하기 때문.
source /opt/ros/jazzy/setup.bash
source "$ZERI_ROOT/.venv-handoff/bin/activate"

python "$ZERI_ROOT/src/lerobot/vlm_agent/mediapipe_handoff_auto_release_node.py" \
  --ros-args \
  -p arm:=left \
  -p stable_frames_required:=2 \
  -p roi:="[0.0, 0.0, 1.0, 1.0]"
