#!/usr/bin/env python3

import time
from typing import Optional

import numpy as np

import rclpy
from rclpy.node import Node
from rclpy.qos import (
    QoSProfile,
    ReliabilityPolicy,
    HistoryPolicy,
    DurabilityPolicy,
)

from sensor_msgs.msg import Image, CameraInfo


try:
    import pyrealsense2 as rs
except ImportError as exc:
    rs = None
    RS_IMPORT_ERROR = exc
else:
    RS_IMPORT_ERROR = None


class RealSenseRgbdNode(Node):
    """
    RealSense RGB + Depth publisher.

    Publish:
      /zeri/vlm/input_rgb      sensor_msgs/Image, rgb8
      /zeri/vlm/input_depth    sensor_msgs/Image, 16UC1, depth in millimeters
      /zeri/camera/color/camera_info
      /zeri/camera/depth/camera_info
    """

    def __init__(self) -> None:
        super().__init__("realsense_rgbd_node")

        self.declare_parameter("serial_number", "944122071303")

        self.declare_parameter("rgb_topic", "/zeri/vlm/input_rgb")
        self.declare_parameter("depth_topic", "/zeri/vlm/input_depth")
        self.declare_parameter("color_info_topic", "/zeri/camera/color/camera_info")
        self.declare_parameter("depth_info_topic", "/zeri/camera/depth/camera_info")

        self.declare_parameter("color_frame_id", "top_camera_color_optical_frame")
        self.declare_parameter("depth_frame_id", "top_camera_depth_optical_frame")

        self.declare_parameter("width", 640)
        self.declare_parameter("height", 480)
        self.declare_parameter("fps", 30)

        self.declare_parameter("align_depth_to_color", True)
        self.declare_parameter("publish_rate_limit_hz", 30.0)

        self.declare_parameter("enable_rgb", True)
        self.declare_parameter("enable_depth", True)

        self.declare_parameter("log_every_n_frames", 60)

        self.serial_number = str(self.get_parameter("serial_number").value)

        self.rgb_topic = str(self.get_parameter("rgb_topic").value)
        self.depth_topic = str(self.get_parameter("depth_topic").value)
        self.color_info_topic = str(self.get_parameter("color_info_topic").value)
        self.depth_info_topic = str(self.get_parameter("depth_info_topic").value)

        self.color_frame_id = str(self.get_parameter("color_frame_id").value)
        self.depth_frame_id = str(self.get_parameter("depth_frame_id").value)

        self.width = int(self.get_parameter("width").value)
        self.height = int(self.get_parameter("height").value)
        self.fps = int(self.get_parameter("fps").value)

        self.align_depth_to_color = bool(
            self.get_parameter("align_depth_to_color").value
        )
        self.publish_rate_limit_hz = float(
            self.get_parameter("publish_rate_limit_hz").value
        )

        self.enable_rgb = bool(self.get_parameter("enable_rgb").value)
        self.enable_depth = bool(self.get_parameter("enable_depth").value)

        self.log_every_n_frames = int(self.get_parameter("log_every_n_frames").value)

        if rs is None:
            raise RuntimeError(
                f"pyrealsense2 import failed: {RS_IMPORT_ERROR}. "
                "Install or activate the environment where pyrealsense2 works."
            )

        qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=5,
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
        )

        self.rgb_pub = self.create_publisher(Image, self.rgb_topic, qos)
        self.depth_pub = self.create_publisher(Image, self.depth_topic, qos)
        self.color_info_pub = self.create_publisher(CameraInfo, self.color_info_topic, qos)
        self.depth_info_pub = self.create_publisher(CameraInfo, self.depth_info_topic, qos)

        self.pipeline: Optional["rs.pipeline"] = None
        self.align = None

        self.color_intrinsics = None
        self.depth_intrinsics = None

        self.frame_count = 0
        self.last_publish_time = 0.0

        self._start_camera()

        # RealSense wait_for_frames가 blocking이므로 timer는 작게 둠
        self.timer = self.create_timer(0.001, self.on_timer)

        self.get_logger().info("realsense_rgbd_node started")
        self.get_logger().info(f"  serial_number={self.serial_number}")
        self.get_logger().info(f"  rgb_topic={self.rgb_topic}")
        self.get_logger().info(f"  depth_topic={self.depth_topic}")
        self.get_logger().info(f"  width={self.width}, height={self.height}, fps={self.fps}")
        self.get_logger().info(f"  align_depth_to_color={self.align_depth_to_color}")

    def _start_camera(self) -> None:
        self.pipeline = rs.pipeline()
        config = rs.config()

        if self.serial_number:
            config.enable_device(self.serial_number)

        if self.enable_depth:
            config.enable_stream(
                rs.stream.depth,
                self.width,
                self.height,
                rs.format.z16,
                self.fps,
            )

        if self.enable_rgb:
            config.enable_stream(
                rs.stream.color,
                self.width,
                self.height,
                rs.format.rgb8,
                self.fps,
            )

        profile = self.pipeline.start(config)

        if self.align_depth_to_color and self.enable_rgb and self.enable_depth:
            self.align = rs.align(rs.stream.color)
        else:
            self.align = None

        color_stream = profile.get_stream(rs.stream.color).as_video_stream_profile()
        depth_stream = profile.get_stream(rs.stream.depth).as_video_stream_profile()

        self.color_intrinsics = color_stream.get_intrinsics()
        self.depth_intrinsics = depth_stream.get_intrinsics()

        depth_sensor = profile.get_device().first_depth_sensor()
        depth_scale = depth_sensor.get_depth_scale()

        self.get_logger().info(f"RealSense opened: serial={self.serial_number}")
        self.get_logger().info(f"depth_scale={depth_scale}")
        self.get_logger().info(
            f"color intrinsics: fx={self.color_intrinsics.fx:.2f}, "
            f"fy={self.color_intrinsics.fy:.2f}, "
            f"ppx={self.color_intrinsics.ppx:.2f}, "
            f"ppy={self.color_intrinsics.ppy:.2f}"
        )
        self.get_logger().info(
            f"depth intrinsics: fx={self.depth_intrinsics.fx:.2f}, "
            f"fy={self.depth_intrinsics.fy:.2f}, "
            f"ppx={self.depth_intrinsics.ppx:.2f}, "
            f"ppy={self.depth_intrinsics.ppy:.2f}"
        )

    def on_timer(self) -> None:
        if self.pipeline is None:
            return

        now = time.monotonic()
        if self.publish_rate_limit_hz > 0.0:
            min_period = 1.0 / self.publish_rate_limit_hz
            if now - self.last_publish_time < min_period:
                return

        try:
            frames = self.pipeline.wait_for_frames(timeout_ms=1000)
        except Exception as exc:
            self.get_logger().warn(f"wait_for_frames failed: {exc}")
            return

        if self.align is not None:
            frames = self.align.process(frames)

        color_frame = frames.get_color_frame() if self.enable_rgb else None
        depth_frame = frames.get_depth_frame() if self.enable_depth else None

        if self.enable_rgb and not color_frame:
            self.get_logger().warn("no color frame")
            return

        if self.enable_depth and not depth_frame:
            self.get_logger().warn("no depth frame")
            return

        stamp = self.get_clock().now().to_msg()

        if color_frame:
            color_np = np.asanyarray(color_frame.get_data())
            rgb_msg = self._make_rgb_image_msg(color_np, stamp)
            self.rgb_pub.publish(rgb_msg)

            color_info = self._make_camera_info_msg(
                self.color_intrinsics,
                stamp,
                self.color_frame_id,
            )
            self.color_info_pub.publish(color_info)

        if depth_frame:
            depth_np = np.asanyarray(depth_frame.get_data())

            # RealSense z16는 uint16 mm 단위로 쓰는 게 ROS에서 가장 다루기 편함.
            depth_msg = self._make_depth_image_msg(depth_np, stamp)
            self.depth_pub.publish(depth_msg)

            info_intr = self.color_intrinsics if self.align is not None else self.depth_intrinsics
            info_frame = self.depth_frame_id
            if self.align is not None:
                # depth가 color frame에 align되었으므로 intrinsics는 color 기준
                info_frame = self.depth_frame_id

            depth_info = self._make_camera_info_msg(
                info_intr,
                stamp,
                info_frame,
            )
            self.depth_info_pub.publish(depth_info)

        self.frame_count += 1
        self.last_publish_time = now

        if self.log_every_n_frames > 0 and self.frame_count % self.log_every_n_frames == 0:
            self.get_logger().info(
                f"published frames={self.frame_count} "
                f"rgb={self.rgb_topic} depth={self.depth_topic}"
            )

    def _make_rgb_image_msg(self, arr: np.ndarray, stamp) -> Image:
        msg = Image()
        msg.header.stamp = stamp
        msg.header.frame_id = self.color_frame_id

        msg.height = int(arr.shape[0])
        msg.width = int(arr.shape[1])
        msg.encoding = "rgb8"
        msg.is_bigendian = False
        msg.step = int(msg.width * 3)
        msg.data = arr.tobytes()

        return msg

    def _make_depth_image_msg(self, arr: np.ndarray, stamp) -> Image:
        if arr.dtype != np.uint16:
            arr = arr.astype(np.uint16)

        msg = Image()
        msg.header.stamp = stamp
        msg.header.frame_id = self.depth_frame_id

        msg.height = int(arr.shape[0])
        msg.width = int(arr.shape[1])
        msg.encoding = "16UC1"
        msg.is_bigendian = False
        msg.step = int(msg.width * 2)
        msg.data = arr.tobytes()

        return msg

    def _make_camera_info_msg(self, intr, stamp, frame_id: str) -> CameraInfo:
        msg = CameraInfo()
        msg.header.stamp = stamp
        msg.header.frame_id = frame_id

        msg.width = int(intr.width)
        msg.height = int(intr.height)

        msg.distortion_model = "plumb_bob"

        # pyrealsense2 intrinsics coeffs: [k1, k2, p1, p2, k3]
        coeffs = list(intr.coeffs)
        if len(coeffs) < 5:
            coeffs += [0.0] * (5 - len(coeffs))
        msg.d = [float(x) for x in coeffs[:5]]

        fx = float(intr.fx)
        fy = float(intr.fy)
        cx = float(intr.ppx)
        cy = float(intr.ppy)

        msg.k = [
            fx, 0.0, cx,
            0.0, fy, cy,
            0.0, 0.0, 1.0,
        ]

        msg.r = [
            1.0, 0.0, 0.0,
            0.0, 1.0, 0.0,
            0.0, 0.0, 1.0,
        ]

        msg.p = [
            fx, 0.0, cx, 0.0,
            0.0, fy, cy, 0.0,
            0.0, 0.0, 1.0, 0.0,
        ]

        return msg

    def destroy_node(self) -> bool:
        try:
            if self.pipeline is not None:
                self.pipeline.stop()
        except Exception:
            pass

        return super().destroy_node()


def main(args=None) -> None:
    rclpy.init(args=args)

    node = RealSenseRgbdNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
