#!/usr/bin/env bash

set -e

source "$HOME/ze-ri/source_zeri.sh"

SERIAL="${SERIAL:-_944122071303}"
WIDTH="${WIDTH:-640}"
HEIGHT="${HEIGHT:-480}"
FPS="${FPS:-30}"

CAMERA_NODE="${CAMERA_NODE:-/camera/camera}"

PIDS=()

cleanup() {
  echo
  echo "[RealSense PointCloud] stopping..."

  for pid in "${PIDS[@]}"; do
    if kill -0 "$pid" 2>/dev/null; then
      kill "$pid" 2>/dev/null || true
    fi
  done

  sleep 0.5

  for pid in "${PIDS[@]}"; do
    if kill -0 "$pid" 2>/dev/null; then
      kill -9 "$pid" 2>/dev/null || true
    fi
  done

  echo "[RealSense PointCloud] stopped"
}

trap cleanup EXIT INT TERM

echo "[Ze-Ri RealSense PointCloud]"
echo "  SERIAL=$SERIAL"
echo "  PROFILE=${WIDTH}x${HEIGHT}x${FPS}"
echo "  CAMERA_NODE=$CAMERA_NODE"
echo

ros2 launch realsense2_camera rs_launch.py \
  serial_no:="$SERIAL" \
  enable_color:=true \
  enable_depth:=true \
  enable_sync:=true \
  align_depth.enable:=true \
  pointcloud__neon_.enable:=true \
  pointcloud__neon_.stream_filter:=2 \
  pointcloud__neon_.stream_index_filter:=0 \
  pointcloud__neon_.allow_no_texture_points:=false \
  rgb_camera.profile:="${WIDTH}x${HEIGHT}x${FPS}" \
  depth_module.profile:="${WIDTH}x${HEIGHT}x${FPS}" &

PIDS+=("$!")

echo "[WAIT] waiting for camera node: $CAMERA_NODE"

for i in $(seq 1 30); do
  if ros2 node list | grep -qx "$CAMERA_NODE"; then
    echo "[OK] camera node detected"
    break
  fi

  if [ "$i" -eq 30 ]; then
    echo "[ERROR] camera node not found: $CAMERA_NODE"
    ros2 node list
    exit 1
  fi

  sleep 0.5
done

echo "[SET] enabling pointcloud params"

# wrapper 버전에 따라 launch arg가 반영 안 되는 경우가 있어서 런타임 param set을 한 번 더 강제한다.
ros2 param set "$CAMERA_NODE" pointcloud__neon_.stream_filter 2 || true
ros2 param set "$CAMERA_NODE" pointcloud__neon_.stream_index_filter 0 || true
ros2 param set "$CAMERA_NODE" pointcloud__neon_.allow_no_texture_points false || true
ros2 param set "$CAMERA_NODE" pointcloud__neon_.enable true || true

sleep 1.0

echo
echo "[CHECK] pointcloud params"
ros2 param get "$CAMERA_NODE" pointcloud__neon_.enable || true
ros2 param get "$CAMERA_NODE" pointcloud__neon_.allow_no_texture_points || true

echo
echo "[CHECK] point topics"
ros2 topic list | grep -E "point|points" || true

echo
echo "[READY] RealSense RGB-D + PointCloud is running."
echo
echo "Useful topics:"
echo "  ros2 topic list | grep points"
echo "  ros2 topic list | grep -E 'color/image_raw|aligned_depth|camera_info|points'"
echo
echo "Press Ctrl+C to stop."

wait
