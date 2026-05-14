#!/usr/bin/env python3
# src/lerobot/vlm_agent/realsense_top_camera_node.py

import threading
from typing import Optional

import numpy as np
import pyrealsense2 as rs
import rclpy
from rclpy.node import Node
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import Image


def make_sensor_qos(depth: int = 5) -> QoSProfile:
    return QoSProfile(
        history=HistoryPolicy.KEEP_LAST,
        depth=depth,
        reliability=ReliabilityPolicy.BEST_EFFORT,
    )


def rgb_np_to_ros_image(
    rgb: np.ndarray,
    stamp,
    frame_id: str,
) -> Image:
    rgb = np.ascontiguousarray(rgb, dtype=np.uint8)

    msg = Image()
    msg.header.stamp = stamp
    msg.header.frame_id = frame_id
    msg.height = int(rgb.shape[0])
    msg.width = int(rgb.shape[1])
    msg.encoding = "rgb8"
    msg.is_bigendian = False
    msg.step = int(rgb.shape[1] * 3)
    msg.data = rgb.tobytes()
    return msg


def depth_np_to_ros_image(
    depth: np.ndarray,
    stamp,
    frame_id: str,
) -> Image:
    if depth.dtype != np.uint16:
        depth = depth.astype(np.uint16)

    depth = np.ascontiguousarray(depth)

    msg = Image()
    msg.header.stamp = stamp
    msg.header.frame_id = frame_id
    msg.height = int(depth.shape[0])
    msg.width = int(depth.shape[1])
    msg.encoding = "16UC1"
    msg.is_bigendian = False
    msg.step = int(depth.shape[1] * 2)
    msg.data = depth.tobytes()
    return msg


class RealSenseTopCameraNode(Node):
    def __init__(self) -> None:
        super().__init__("zeri_realsense_top_camera_node")

        self.declare_parameter("serial", "332322071907")
        self.declare_parameter("width", 640)
        self.declare_parameter("height", 480)
        self.declare_parameter("fps", 30)

        self.declare_parameter("rgb_topic", "/zeri/top/rgb/image_raw")
        self.declare_parameter("depth_topic", "/zeri/top/depth/image_raw")

        self.declare_parameter("rgb_frame_id", "zeri_top_rgb")
        self.declare_parameter("depth_frame_id", "zeri_top_depth")

        self.declare_parameter("publish_depth", True)
        self.declare_parameter("align_depth_to_color", True)

        self.serial = str(self.get_parameter("serial").value)
        self.width = int(self.get_parameter("width").value)
        self.height = int(self.get_parameter("height").value)
        self.fps = int(self.get_parameter("fps").value)

        self.rgb_topic = str(self.get_parameter("rgb_topic").value)
        self.depth_topic = str(self.get_parameter("depth_topic").value)

        self.rgb_frame_id = str(self.get_parameter("rgb_frame_id").value)
        self.depth_frame_id = str(self.get_parameter("depth_frame_id").value)

        self.publish_depth = bool(self.get_parameter("publish_depth").value)
        self.align_depth_to_color = bool(self.get_parameter("align_depth_to_color").value)

        sensor_qos = make_sensor_qos(depth=5)

        self.rgb_pub = self.create_publisher(Image, self.rgb_topic, sensor_qos)
        self.depth_pub = self.create_publisher(Image, self.depth_topic, sensor_qos)

        self.pipeline: Optional[rs.pipeline] = None
        self.align: Optional[rs.align] = None
        self.lock = threading.Lock()

        self.get_logger().info("Starting RealSense top camera node")
        self.get_logger().info(f"  serial: {self.serial}")
        self.get_logger().info(f"  resolution: {self.width}x{self.height}@{self.fps}")
        self.get_logger().info(f"  rgb_topic: {self.rgb_topic}")
        self.get_logger().info(f"  depth_topic: {self.depth_topic}")
        self.get_logger().info(f"  publish_depth: {self.publish_depth}")
        self.get_logger().info(f"  align_depth_to_color: {self.align_depth_to_color}")

        self.start_camera()

        timer_period = 1.0 / max(1, self.fps)
        self.timer = self.create_timer(timer_period, self.timer_callback)

    def start_camera(self) -> None:
        pipeline = rs.pipeline()
        config = rs.config()

        config.enable_device(self.serial)

        config.enable_stream(
            rs.stream.color,
            self.width,
            self.height,
            rs.format.bgr8,
            self.fps,
        )

        if self.publish_depth:
            config.enable_stream(
                rs.stream.depth,
                self.width,
                self.height,
                rs.format.z16,
                self.fps,
            )

        pipeline.start(config)

        if self.publish_depth and self.align_depth_to_color:
            self.align = rs.align(rs.stream.color)

        for _ in range(30):
            pipeline.wait_for_frames()

        self.pipeline = pipeline
        self.get_logger().info("RealSense top camera started.")

    def timer_callback(self) -> None:
        if self.pipeline is None:
            return

        try:
            with self.lock:
                frames = self.pipeline.wait_for_frames(timeout_ms=1000)

                if self.align is not None:
                    frames = self.align.process(frames)

                color_frame = frames.get_color_frame()
                if not color_frame:
                    self.get_logger().warn("No color frame.")
                    return

                bgr = np.asanyarray(color_frame.get_data())
                rgb = bgr[:, :, ::-1]

                stamp = self.get_clock().now().to_msg()

                rgb_msg = rgb_np_to_ros_image(
                    rgb=rgb,
                    stamp=stamp,
                    frame_id=self.rgb_frame_id,
                )
                self.rgb_pub.publish(rgb_msg)

                if self.publish_depth:
                    depth_frame = frames.get_depth_frame()
                    if depth_frame:
                        depth = np.asanyarray(depth_frame.get_data()).copy()
                        depth_msg = depth_np_to_ros_image(
                            depth=depth,
                            stamp=stamp,
                            frame_id=self.depth_frame_id,
                        )
                        self.depth_pub.publish(depth_msg)

        except Exception as exc:
            self.get_logger().error(f"RealSense publish error: {exc}")

    def destroy_node(self) -> None:
        self.get_logger().info("Stopping RealSense top camera node.")

        with self.lock:
            if self.pipeline is not None:
                self.pipeline.stop()
                self.pipeline = None

        super().destroy_node()


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