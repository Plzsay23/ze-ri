# Zero-Risk

Zero-Risk는 재난 상황 초동 조치를 위한 VLM 기반 지능형 모바일 매니퓰레이터입니다. RGB-D 카메라, 2D LiDAR, ReSpeaker Mic Array, SenseVoice STT, Edge-TTS, Qwen3-VL, SO-101 양팔 로봇팔, 4륜 메카넘 베이스를 ROS2 Jazzy 토픽으로 연결합니다.

## 주요 기능

- RGB-D 카메라와 LiDAR 기반 사람 접근 및 안전 정지
- ReSpeaker VAD/DOA와 SenseVoice 기반 음성 입력
- Edge-TTS 기반 구조 안내 발화
- Qwen3-VL 기반 구조 대상자 요청 판단
- 물, 산소마스크, 무전기 전달 VLA task routing
- SO-101 양팔 로봇팔 기반 물품 집기와 전달
- RGB, Depth, PointCloud, STT, TTS, VLM, VLA, 자율주행 상태 대시보드

## 환경

- Ubuntu 24.04
- ROS2 Jazzy
- Python 3.12
- NVIDIA Jetson Thor
- Intel RealSense D435i/D435if
- EAI X2 2D LiDAR
- ReSpeaker USB Mic Array v3.0
- SO-101 robot arm 2대

## 빌드

```bash
cd ~/ze-ri
source /opt/ros/jazzy/setup.bash
colcon build --base-paths ros2_ws/src --symlink-install
source ~/ze-ri/source_zeri.sh
```

## 수동 실행

각 명령은 별도 터미널에서 실행합니다.

```bash
cd ~/ze-ri
source ~/ze-ri/source_zeri.sh
```

카메라:

```bash
bash ~/ze-ri/scripts/camera_rgbd.sh
```

SLAM, LiDAR safety, base odometry:

```bash
bash ~/ze-ri/scripts/slam_depth_drive.sh
```

음성 방향 주행은 SLAM/base stack을 먼저 켠 뒤 실행합니다:

```bash
bash ~/ze-ri/scripts/voice_direction_drive.sh
```

STT, TTS, VLM:

```bash
bash ~/ze-ri/scripts/start_hri.sh
```

양팔 VLA:

```bash
bash ~/ze-ri/scripts/start_vla.sh
```

대시보드:

```bash
python3 ~/ze-ri/dashboard/dashboard_server.py
```

## 주요 토픽

- `/zeri/vlm/input_rgb`
- `/zeri/vlm/input_depth`
- `/stt/text`
- `/zeri/vlm/robot_speech`
- `/zeri/vlm/decision`
- `/zeri/vla/task_request`
- `/zeri/vla/status`
- `/cmd_vel_raw`
- `/cmd_vel`
- `/odom`
- `/scan_front`
- `/map`

## 스모크 테스트

```bash
python3 -m py_compile \
  ros2_ws/src/nb_voice_stt/nb_voice_stt/*.py \
  ros2_ws/src/zeri_base/zeri_base/*.py \
  ros2_ws/src/zeri_bringup/zeri_bringup/*.py \
  ros2_ws/src/zeri_camera/zeri_camera/*.py \
  ros2_ws/src/zeri_lidar/zeri_lidar/*.py \
  ros2_ws/src/zeri_voice/zeri_voice/*.py \
  src/lerobot/vlm_agent/*.py

python3 - <<'PY'
from pathlib import Path
import xml.etree.ElementTree as ET
for path in Path("ros2_ws/src").glob("*/package.xml"):
    ET.parse(path)
print("package.xml ok")
PY

bash -n source_zeri.sh source_zeri_vlm.sh scripts/*.sh
```
