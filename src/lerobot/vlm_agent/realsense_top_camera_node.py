#!/usr/bin/env python3
# src/lerobot/vlm_agent/realsense_top_camera_node.py

import time
from typing import Optional

import numpy as np
import pyrealsense2 as rs
import rclpy
from rclpy.node import Node
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import Image
from std_msgs.msg import String


def make_sensor_qos(depth: int = 5) -> QoSProfile:
    return QoSProfile(
        history=HistoryPolicy.KEEP_LAST,
        depth=depth,
        reliability=ReliabilityPolicy.BEST_EFFORT,
    )


def make_reliable_qos(depth: int = 10) -> QoSProfile:
    return QoSProfile(
        history=HistoryPolicy.KEEP_LAST,
        depth=depth,
        reliability=ReliabilityPolicy.RELIABLE,
    )


def safe_get_device_info(device: rs.device, info_key) -> str:
    try:
        if device.supports(info_key):
            return str(device.get_info(info_key))
    except Exception:
        pass
    return ""


def make_image_msg(
    array: np.ndarray,
    stamp,
    frame_id: str,
    encoding: str,
) -> Image:
    array = np.ascontiguousarray(array)

    msg = Image()
    msg.header.stamp = stamp
    msg.header.frame_id = frame_id
    msg.height = int(array.shape[0])
    msg.width = int(array.shape[1])
    msg.encoding = encoding
    msg.is_bigendian = False

    if array.ndim == 2:
        item_size = int(array.dtype.itemsize)
        msg.step = int(array.shape[1] * item_size)
    elif array.ndim == 3:
        item_size = int(array.dtype.itemsize)
        channels = int(array.shape[2])
        msg.step = int(array.shape[1] * channels * item_size)
    else:
        raise ValueError(f"Unsupported image array shape: {array.shape}")

    msg.data = array.tobytes()
    return msg


class RealSenseTopCameraNode(Node):
    """
    RealSense top camera publisher for Ze-Ri.

    Publish:
      /zeri/top/rgb/image_raw      sensor_msgs/Image, bgr8
      /zeri/top/depth/image_raw    sensor_msgs/Image, 16UC1
      /zeri/top/camera_status      std_msgs/String

    VLM node default input:
      rgb_topic   = /zeri/top/rgb/image_raw
      depth_topic = /zeri/top/depth/image_raw
    """

    def __init__(self) -> None:
        super().__init__("zeri_realsense_top_camera_node")

        self.declare_parameter("serial_number_or_name", "")

        self.declare_parameter("rgb_topic", "/zeri/top/rgb/image_raw")
        self.declare_parameter("depth_topic", "/zeri/top/depth/image_raw")
        self.declare_parameter("status_topic", "/zeri/top/camera_status")

        self.declare_parameter("color_frame_id", "zeri_top_color_frame")
        self.declare_parameter("depth_frame_id", "zeri_top_depth_frame")

        self.declare_parameter("width", 1280)
        self.declare_parameter("height", 720)
        self.declare_parameter("fps", 30)

        self.declare_parameter("enable_depth", True)
        self.declare_parameter("align_depth_to_color", True)

        self.declare_parameter("frame_timeout_ms", 2000)
        self.declare_parameter("reconnect_delay_sec", 2.0)
        self.declare_parameter("status_period_sec", 1.0)

        self.declare_parameter("qos_depth", 5)

        self.serial_number_or_name = str(
            self.get_parameter("serial_number_or_name").value
        ).strip()

        self.rgb_topic = str(self.get_parameter("rgb_topic").value)
        self.depth_topic = str(self.get_parameter("depth_topic").value)
        self.status_topic = str(self.get_parameter("status_topic").value)

        self.color_frame_id = str(self.get_parameter("color_frame_id").value)
        self.depth_frame_id = str(self.get_parameter("depth_frame_id").value)

        self.width = int(self.get_parameter("width").value)
        self.height = int(self.get_parameter("height").value)
        self.fps = int(self.get_parameter("fps").value)

        self.enable_depth = bool(self.get_parameter("enable_depth").value)
        self.align_depth_to_color = bool(
            self.get_parameter("align_depth_to_color").value
        )

        self.frame_timeout_ms = int(self.get_parameter("frame_timeout_ms").value)
        self.reconnect_delay_sec = float(
            self.get_parameter("reconnect_delay_sec").value
        )
        self.status_period_sec = float(
            self.get_parameter("status_period_sec").value
        )

        qos_depth = int(self.get_parameter("qos_depth").value)

        sensor_qos = make_sensor_qos(depth=qos_depth)
        reliable_qos = make_reliable_qos(depth=10)

        self.rgb_pub = self.create_publisher(Image, self.rgb_topic, sensor_qos)
        self.depth_pub = self.create_publisher(Image, self.depth_topic, sensor_qos)
        self.status_pub = self.create_publisher(String, self.status_topic, reliable_qos)

        self.pipeline: Optional[rs.pipeline] = None
        self.align: Optional[rs.align] = None

        self.selected_serial = ""
        self.selected_name = ""
        self.connected = False

        self.rgb_count = 0
        self.depth_count = 0
        self.last_reconnect_attempt = 0.0
        self.last_status_time = 0.0

        self.get_logger().info("Ze-Ri RealSense top camera settings:")
        self.get_logger().info(f"  serial_number_or_name: {self.serial_number_or_name or '<auto>'}")
        self.get_logger().info(f"  rgb_topic:             {self.rgb_topic}")
        self.get_logger().info(f"  depth_topic:           {self.depth_topic}")
        self.get_logger().info(f"  status_topic:          {self.status_topic}")
        self.get_logger().info(f"  resolution:            {self.width}x{self.height}@{self.fps}")
        self.get_logger().info(f"  enable_depth:          {self.enable_depth}")
        self.get_logger().info(f"  align_depth_to_color:  {self.align_depth_to_color}")

        self.connect_camera(force=True)

        timer_period = 1.0 / max(1, self.fps)
        self.timer = self.create_timer(timer_period, self.poll_once)

    def publish_status(self, status: str, force: bool = False) -> None:
        now = time.time()

        if not force and now - self.last_status_time < self.status_period_sec:
            return

        self.last_status_time = now

        payload = {
            "status": status,
            "connected": self.connected,
            "serial": self.selected_serial,
            "name": self.selected_name,
            "rgb_topic": self.rgb_topic,
            "depth_topic": self.depth_topic,
            "width": self.width,
            "height": self.height,
            "fps": self.fps,
            "rgb_count": self.rgb_count,
            "depth_count": self.depth_count,
            "enable_depth": self.enable_depth,
            "align_depth_to_color": self.align_depth_to_color,
            "stamp_sec": round(now, 3),
        }

        msg = String()
        msg.data = json_dumps(payload)
        self.status_pub.publish(msg)

        if force:
            self.get_logger().info(f"[CAMERA STATUS] {msg.data}")

    def find_matching_device(self) -> tuple[str, str]:
        ctx = rs.context()
        devices = list(ctx.query_devices())

        if not devices:
            raise RuntimeError("No RealSense device found.")

        hint = self.serial_number_or_name.strip().lower()

        candidates = []

        for dev in devices:
            serial = safe_get_device_info(dev, rs.camera_info.serial_number)
            name = safe_get_device_info(dev, rs.camera_info.name)
            product_id = safe_get_device_info(dev, rs.camera_info.product_id)
            firmware = safe_get_device_info(dev, rs.camera_info.firmware_version)

            candidates.append(
                {
                    "serial": serial,
                    "name": name,
                    "product_id": product_id,
                    "firmware": firmware,
                }
            )

        self.get_logger().info("Detected RealSense devices:")
        for c in candidates:
            self.get_logger().info(
                f"  serial={c['serial']} | name={c['name']} | "
                f"product_id={c['product_id']} | firmware={c['firmware']}"
            )

        if not hint:
            first = candidates[0]
            serial = first["serial"]
            name = first["name"]

            if not serial:
                raise RuntimeError("First RealSense device has no serial number.")

            return serial, name

        for c in candidates:
            serial = c["serial"]
            name = c["name"]
            product_id = c["product_id"]

            blob = f"{serial} {name} {product_id}".lower()

            if hint == serial.lower() or hint in blob:
                if not serial:
                    raise RuntimeError(
                        f"Matched device has no serial number: {c}"
                    )
                return serial, name

        raise RuntimeError(
            "No RealSense device matched "
            f"serial_number_or_name='{self.serial_number_or_name}'. "
            f"Detected={candidates}"
        )

    def stop_camera(self) -> None:
        if self.pipeline is not None:
            try:
                self.pipeline.stop()
            except Exception:
                pass

        self.pipeline = None
        self.align = None
        self.connected = False

    def connect_camera(self, force: bool = False) -> None:
        now = time.time()

        if not force and now - self.last_reconnect_attempt < self.reconnect_delay_sec:
            return

        self.last_reconnect_attempt = now

        self.stop_camera()

        try:
            serial, name = self.find_matching_device()

            config = rs.config()
            config.enable_device(serial)

            config.enable_stream(
                rs.stream.color,
                self.width,
                self.height,
                rs.format.bgr8,
                self.fps,
            )

            if self.enable_depth:
                config.enable_stream(
                    rs.stream.depth,
                    self.width,
                    self.height,
                    rs.format.z16,
                    self.fps,
                )

            pipeline = rs.pipeline()
            pipeline.start(config)

            self.pipeline = pipeline
            self.selected_serial = serial
            self.selected_name = name
            self.connected = True

            if self.enable_depth and self.align_depth_to_color:
                self.align = rs.align(rs.stream.color)
            else:
                self.align = None

            self.publish_status("connected", force=True)
            self.get_logger().info(
                f"RealSense connected: serial={serial}, name={name}"
            )

        except Exception as exc:
            self.stop_camera()
            self.publish_status(f"connect_error: {exc}", force=True)
            self.get_logger().error(f"Failed to connect RealSense: {exc}")

    def poll_once(self) -> None:
        if self.pipeline is None:
            self.connect_camera(force=False)
            return

        try:
            frames = self.pipeline.wait_for_frames(
                timeout_ms=self.frame_timeout_ms
            )

            if self.align is not None:
                frames = self.align.process(frames)

            stamp = self.get_clock().now().to_msg()

            color_frame = frames.get_color_frame()
            if color_frame is not None:
                color_bgr = np.asanyarray(color_frame.get_data())
                rgb_msg = make_image_msg(
                    array=color_bgr,
                    stamp=stamp,
                    frame_id=self.color_frame_id,
                    encoding="bgr8",
                )
                self.rgb_pub.publish(rgb_msg)
                self.rgb_count += 1
            else:
                self.get_logger().warn("No color frame in RealSense frameset.")

            if self.enable_depth:
                depth_frame = frames.get_depth_frame()
                if depth_frame is not None:
                    depth_u16 = np.asanyarray(depth_frame.get_data())
                    depth_msg = make_image_msg(
                        array=depth_u16,
                        stamp=stamp,
                        frame_id=self.depth_frame_id,
                        encoding="16UC1",
                    )
                    self.depth_pub.publish(depth_msg)
                    self.depth_count += 1
                else:
                    self.get_logger().warn("No depth frame in RealSense frameset.")

            self.publish_status("running", force=False)

        except Exception as exc:
            self.get_logger().error(f"RealSense polling error: {exc}")
            self.stop_camera()
            self.publish_status(f"poll_error: {exc}", force=True)

    def destroy_node(self) -> None:
        self.get_logger().info("Stopping RealSense top camera node.")

        try:
            self.publish_status("shutting_down", force=True)
        except Exception:
            pass

        self.stop_camera()
        super().destroy_node()


def json_dumps(payload: dict) -> str:
    import json

    return json.dumps(payload, ensure_ascii=False)


def main() -> None:
    rclpy.init()

    node: Optional[RealSenseTopCameraNode] = None

    try:
        node = RealSenseTopCameraNode()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if node is not None:
            node.destroy_node()

        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
