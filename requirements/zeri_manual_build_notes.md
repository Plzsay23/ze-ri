# Ze-Ri manual build notes

## Target environment

```text
Jetson AGX Thor
Ubuntu 24.04
ROS2 Jazzy
Python 3.12
CUDA 13.0
torch 2.11.0+cu130
GPU capability: sm_110
Project venv: ~/ze-ri/.venv
```

---

## pyrealsense2

Status: built and installed manually into `~/ze-ri/.venv`.

Verified:

```text
pyrealsense2 import ok
version: 2.58.0
context ok
device_count: 0
```

`device_count: 0` is normal when RealSense is not connected.

Build source:

```text
IntelRealSense/librealsense
branch: development
commit: 6301abeb2
```

CMake configure:

```bash
source ~/ze-ri/source_zeri_vlm.sh

cd ~/build/librealsense

cmake -S . -B build_zeri \
  -DCMAKE_BUILD_TYPE=Release \
  -DFORCE_RSUSB_BACKEND=ON \
  -DBUILD_PYTHON_BINDINGS=ON \
  -DBUILD_WITH_CUDA=ON \
  -DBUILD_EXAMPLES=OFF \
  -DBUILD_GRAPHICAL_EXAMPLES=OFF \
  -DBUILD_TOOLS=OFF \
  -DCMAKE_CUDA_ARCHITECTURES=110 \
  -DPYTHON_EXECUTABLE="$HOME/ze-ri/.venv/bin/python" \
  -DPython_EXECUTABLE="$HOME/ze-ri/.venv/bin/python" \
  -DCMAKE_INSTALL_PREFIX="$PWD/install_zeri"
```

Build:

```bash
cmake --build build_zeri -j2
cmake --build build_zeri --target pyrealsense2 -j2
cmake --install build_zeri
```

Installed Python module:

```text
~/ze-ri/.venv/lib/python3.12/site-packages/pyrealsense2/pyrealsense2.cpython-312-aarch64-linux-gnu.so
```

Bundled `librealsense2.so` into:

```text
~/ze-ri/.venv/lib/python3.12/site-packages/pyrealsense2/
```

RPATH patch:

```bash
patchelf --set-rpath '$ORIGIN:/usr/local/cuda/lib64' \
  ~/ze-ri/.venv/lib/python3.12/site-packages/pyrealsense2/pyrealsense2.cpython-312-aarch64-linux-gnu.so
```

Expected `ldd` result:

```text
librealsense2.so.2.58 => ~/ze-ri/.venv/lib/python3.12/site-packages/pyrealsense2/librealsense2.so.2.58
libusb-1.0.so.0 => /lib/aarch64-linux-gnu/libusb-1.0.so.0
```

---

## FlashAttention

GROOT was dropped from the main training path.

XVLA is the selected policy path.

FlashAttention was built during debugging but should not be treated as part of the minimal XVLA runtime unless a model path explicitly requires it.

Built artifacts:

```text
FlashAttention-3:
  wheel: flash_attn_3-3.0.0-cp39-abi3-linux_aarch64.whl
  target: sm_110
  import: flash_attn_interface
  status: tested

FlashAttention 2:
  wheel: flash_attn-2.8.4-cp312-cp312-linux_aarch64.whl
  target: sm_110
  import: flash_attn
  status: tested
```

FlashAttention build notes:

```text
CUDA_HOME=/usr/local/cuda
CUDA version=13.0
GPU arch=sm_110
MAX_JOBS was adjusted based on memory pressure.
```

Minimal XVLA requirements should not require FlashAttention by default.

---

## SLAM / LiDAR notes

Device aliases:

```text
/dev/lidar   -> YDLiDAR
/dev/arduino -> Arduino wheel controller
```

YDLiDAR config was changed from:

```text
port: /dev/ydlidar
```

to:

```text
port: /dev/lidar
```

in:

```text
NBYtics/src/ydlidar_ros2_driver/params/ydlidar.yaml
NBYtics/install/ydlidar_ros2_driver/share/ydlidar_ros2_driver/params/ydlidar.yaml
```

Working SLAM launch:

```bash
source ~/ze-ri/source_zeri_vlm.sh

ros2 launch ydlidar_ros2_driver ydlidar_launch.py
ros2 launch rf2o_laser_odometry rf2o_laser_odometry.launch.py

ros2 launch slam_toolbox online_async_launch.py \
  slam_params_file:=$NBYTICS_ROOT/slam_config/mapper_params_online_async.yaml \
  use_sim_time:=false
```

Important: `slam_toolbox` uses `slam_params_file`, not `params_file`.

Verified topics:

```text
/scan
/odom_rf2o
/tf
/tf_static
/map
```

---

## STT notes

STT runs from the unified `~/ze-ri/.venv`.

Verified imports:

```text
sounddevice
soundfile
pyserial
onnxruntime
sentencepiece
kaldi_native_fbank
```

Working STT launch:

```bash
source ~/ze-ri/source_zeri_vlm.sh

ros2 launch nb_voice_stt stt.launch.py \
  model_dir:=$NBYTICS_ROOT/models/sensevoice_ko \
  audio_device:=-1 \
  channels:=6 \
  use_channel_index:=0 \
  sample_rate:=16000 \
  device:=cpu
```

---

## Do not commit

Do not commit these generated/build/runtime paths:

```text
.venv/
build/
install/
log/
target/
env_backup/
__pycache__/
*.pyc
*.bak
NBYtics/build/
NBYtics/install/
NBYtics/log/
```
