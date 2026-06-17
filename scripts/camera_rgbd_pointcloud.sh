#!/usr/bin/env bash
set -Eeuo pipefail

# Ze-Ri RealSense RGB-D + PointCloud launcher
# Publishes dashboard/VLM-facing topics:
#   /zeri/vlm/input_rgb     sensor_msgs/Image
#   /zeri/vlm/input_depth   sensor_msgs/Image
#   /zeri/vlm/pointcloud    sensor_msgs/PointCloud2
#
# Internal RealSense topics remain available under /camera/camera/*.

ZERI_ROOT="${ZERI_ROOT:-$HOME/ze-ri}"
SOURCE_FILE="${SOURCE_FILE:-$ZERI_ROOT/source_zeri.sh}"

if [[ ! -f "$SOURCE_FILE" ]]; then
  echo "[ERROR] source file not found: $SOURCE_FILE" >&2
  exit 1
fi

set +u
source "$SOURCE_FILE"
set -u

# pyrealsense2 / rclpy가 .venv에 있으므로 경로 강제 추가
export PYTHONPATH="$ZERI_ROOT/.venv/lib/python3.12/site-packages:${PYTHONPATH:-}"

RAW_SERIAL="${1:-${SERIAL:-944122071303}}"
if [[ "$RAW_SERIAL" == _* ]]; then
  RS_SERIAL="$RAW_SERIAL"
else
  RS_SERIAL="_${RAW_SERIAL}"
fi

WIDTH="${WIDTH:-640}"
HEIGHT="${HEIGHT:-480}"
FPS="${FPS:-30}"

CAMERA_NODE="${CAMERA_NODE:-/camera/camera}"

# Internal RealSense topics
RS_RGB_TOPIC="${RS_RGB_TOPIC:-/camera/camera/color/image_raw}"
RS_DEPTH_TOPIC="${RS_DEPTH_TOPIC:-/camera/camera/aligned_depth_to_color/image_raw}"
RS_POINTCLOUD_TOPIC="${RS_POINTCLOUD_TOPIC:-/camera/camera/depth/color/points}"

# Ze-Ri public topics used by VLM/dashboard
RGB_TOPIC="${RGB_TOPIC:-/zeri/vlm/input_rgb}"
DEPTH_TOPIC="${DEPTH_TOPIC:-/zeri/vlm/input_depth}"
POINTCLOUD_TOPIC="${POINTCLOUD_TOPIC:-/zeri/vlm/pointcloud}"

# Inline Python relay reads these from env.
export RS_RGB_TOPIC RS_DEPTH_TOPIC RS_POINTCLOUD_TOPIC RGB_TOPIC DEPTH_TOPIC POINTCLOUD_TOPIC

PIDS=()

cleanup() {
  echo
  echo "[Ze-Ri Camera RGBD+PointCloud] stopping..."

  for pid in "${PIDS[@]:-}"; do
    if kill -0 "$pid" 2>/dev/null; then
      kill "$pid" 2>/dev/null || true
    fi
  done

  sleep 0.8

  for pid in "${PIDS[@]:-}"; do
    if kill -0 "$pid" 2>/dev/null; then
      kill -9 "$pid" 2>/dev/null || true
    fi
  done

  echo "[Ze-Ri Camera RGBD+PointCloud] stopped"
}

trap cleanup EXIT INT TERM

echo "[Ze-Ri Camera RGBD+PointCloud]"
echo "  raw_serial=$RAW_SERIAL"
echo "  realsense_serial=$RS_SERIAL"
echo "  profile=${WIDTH}x${HEIGHT}x${FPS}"
echo "  camera_node=$CAMERA_NODE"
echo "  internal_rgb=$RS_RGB_TOPIC"
echo "  internal_depth=$RS_DEPTH_TOPIC"
echo "  internal_pointcloud=$RS_POINTCLOUD_TOPIC"
echo "  publish_rgb=$RGB_TOPIC"
echo "  publish_depth=$DEPTH_TOPIC"
echo "  publish_pointcloud=$POINTCLOUD_TOPIC"
echo

# RealSense ROS wrapper. PointCloud launch args follow the existing Ze-Ri script style.
ros2 launch realsense2_camera rs_launch.py \
  serial_no:="$RS_SERIAL" \
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
for i in $(seq 1 40); do
  if ros2 node list | grep -qx "$CAMERA_NODE"; then
    echo "[OK] camera node detected"
    break
  fi

  if [[ "$i" -eq 40 ]]; then
    echo "[ERROR] camera node not found: $CAMERA_NODE" >&2
    ros2 node list || true
    exit 1
  fi

  sleep 0.5
done

# wrapper 버전에 따라 launch arg가 반영 안 되는 경우가 있어서 런타임 param set도 한 번 더 강제한다.
echo "[SET] enabling pointcloud params"
ros2 param set "$CAMERA_NODE" pointcloud__neon_.stream_filter 2 || true
ros2 param set "$CAMERA_NODE" pointcloud__neon_.stream_index_filter 0 || true
ros2 param set "$CAMERA_NODE" pointcloud__neon_.allow_no_texture_points false || true
ros2 param set "$CAMERA_NODE" pointcloud__neon_.enable true || true

# Relay internal RealSense topics to Ze-Ri public topics.
python -u - <<'PY_RELAY' &
import os
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, HistoryPolicy, ReliabilityPolicy, DurabilityPolicy
from sensor_msgs.msg import Image, PointCloud2

RS_RGB_TOPIC = os.environ["RS_RGB_TOPIC"]
RS_DEPTH_TOPIC = os.environ["RS_DEPTH_TOPIC"]
RS_POINTCLOUD_TOPIC = os.environ["RS_POINTCLOUD_TOPIC"]
RGB_TOPIC = os.environ["RGB_TOPIC"]
DEPTH_TOPIC = os.environ["DEPTH_TOPIC"]
POINTCLOUD_TOPIC = os.environ["POINTCLOUD_TOPIC"]

class CameraRelay(Node):
    def __init__(self):
        super().__init__("zeri_camera_rgbd_pointcloud_relay")
        qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=5,
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
        )
        self.rgb_pub = self.create_publisher(Image, RGB_TOPIC, qos)
        self.depth_pub = self.create_publisher(Image, DEPTH_TOPIC, qos)
        self.pc_pub = self.create_publisher(PointCloud2, POINTCLOUD_TOPIC, qos)
        self.create_subscription(Image, RS_RGB_TOPIC, self.rgb_pub.publish, qos)
        self.create_subscription(Image, RS_DEPTH_TOPIC, self.depth_pub.publish, qos)
        self.create_subscription(PointCloud2, RS_POINTCLOUD_TOPIC, self.pc_pub.publish, qos)
        self.get_logger().info("Camera relay started")
        self.get_logger().info(f"  {RS_RGB_TOPIC} -> {RGB_TOPIC}")
        self.get_logger().info(f"  {RS_DEPTH_TOPIC} -> {DEPTH_TOPIC}")
        self.get_logger().info(f"  {RS_POINTCLOUD_TOPIC} -> {POINTCLOUD_TOPIC}")

rclpy.init()
node = CameraRelay()
try:
    rclpy.spin(node)
finally:
    node.destroy_node()
    if rclpy.ok():
        rclpy.shutdown()
PY_RELAY
PIDS+=("$!")

sleep 1.0

echo
echo "[CHECK] pointcloud params"
ros2 param get "$CAMERA_NODE" pointcloud__neon_.enable || true
ros2 param get "$CAMERA_NODE" pointcloud__neon_.allow_no_texture_points || true

echo
echo "[CHECK] Ze-Ri public camera topics"
ros2 topic list | grep -E "^${RGB_TOPIC}$|^${DEPTH_TOPIC}$|^${POINTCLOUD_TOPIC}$" || true

echo
echo "[READY] RealSense RGB-D + PointCloud relay is running."
echo "Useful checks:"
echo "  ros2 topic hz $RGB_TOPIC"
echo "  ros2 topic hz $DEPTH_TOPIC"
echo "  ros2 topic hz $POINTCLOUD_TOPIC"
echo "  ros2 topic info $POINTCLOUD_TOPIC -v"
echo
echo "Press Ctrl+C to stop."

wait
