#!/usr/bin/env python3

import math
import time

import numpy as np

import rclpy
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy

from sensor_msgs.msg import Image, CameraInfo, PointCloud2, PointField
import sensor_msgs_py.point_cloud2 as pc2


class RgbdPointCloudNode(Node):
    def __init__(self):
        super().__init__("rgbd_pointcloud_node")

        self.declare_parameter("rgb_topic", "/zeri/vlm/input_rgb")
        self.declare_parameter("depth_topic", "/zeri/vlm/input_depth")
        self.declare_parameter("camera_info_topic", "/zeri/camera/color/camera_info")
        self.declare_parameter("points_topic", "/zeri/vlm/points")
        self.declare_parameter("publish_hz", 5.0)
        self.declare_parameter("stride", 4)
        self.declare_parameter("depth_min_m", 0.20)
        self.declare_parameter("depth_max_m", 5.0)

        self.rgb_topic = str(self.get_parameter("rgb_topic").value)
        self.depth_topic = str(self.get_parameter("depth_topic").value)
        self.camera_info_topic = str(self.get_parameter("camera_info_topic").value)
        self.points_topic = str(self.get_parameter("points_topic").value)
        self.publish_hz = float(self.get_parameter("publish_hz").value)
        self.stride = max(1, int(self.get_parameter("stride").value))
        self.depth_min_m = float(self.get_parameter("depth_min_m").value)
        self.depth_max_m = float(self.get_parameter("depth_max_m").value)

        self.last_rgb = None
        self.last_depth = None
        self.last_info = None
        self.last_rgb_stamp = 0.0
        self.last_depth_stamp = 0.0
        self.last_info_stamp = 0.0
        self.last_depth_header = None

        sensor_qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=5,
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
        )

        self.create_subscription(Image, self.rgb_topic, self.on_rgb, sensor_qos)
        self.create_subscription(Image, self.depth_topic, self.on_depth, sensor_qos)
        self.create_subscription(CameraInfo, self.camera_info_topic, self.on_info, sensor_qos)
        self.pub = self.create_publisher(PointCloud2, self.points_topic, 5)
        self.timer = self.create_timer(1.0 / max(self.publish_hz, 0.5), self.publish_cloud)

        self.get_logger().info(
            "rgbd pointcloud node started: "
            f"rgb={self.rgb_topic}, depth={self.depth_topic}, info={self.camera_info_topic}, "
            f"points={self.points_topic}, stride={self.stride}"
        )

    def on_rgb(self, msg):
        try:
            self.last_rgb = self.decode_rgb(msg)
            self.last_rgb_stamp = time.time()
        except Exception as exc:
            self.get_logger().warn(f"RGB decode failed: {exc}")

    def on_depth(self, msg):
        try:
            self.last_depth = self.decode_depth(msg)
            self.last_depth_stamp = time.time()
            self.last_depth_header = msg.header
        except Exception as exc:
            self.get_logger().warn(f"Depth decode failed: {exc}")

    def on_info(self, msg):
        self.last_info = msg
        self.last_info_stamp = time.time()

    def decode_rgb(self, msg):
        h = int(msg.height)
        w = int(msg.width)
        step = int(msg.step)
        enc = msg.encoding.lower()
        raw = np.frombuffer(msg.data, dtype=np.uint8).reshape(h, step)

        if enc == "rgb8":
            return raw[:, :w * 3].reshape(h, w, 3).copy()
        if enc == "bgr8":
            return raw[:, :w * 3].reshape(h, w, 3)[:, :, ::-1].copy()
        if enc in ("rgba8", "bgra8"):
            arr = raw[:, :w * 4].reshape(h, w, 4)[:, :, :3]
            if enc == "bgra8":
                arr = arr[:, :, ::-1]
            return arr.copy()

        raise RuntimeError(f"unsupported RGB encoding: {msg.encoding}")

    def decode_depth(self, msg):
        h = int(msg.height)
        w = int(msg.width)
        step = int(msg.step)
        enc = msg.encoding.lower()

        if enc in ("16uc1", "mono16"):
            raw = np.frombuffer(msg.data, dtype=np.uint16).reshape(h, step // 2)
            return raw[:, :w].astype(np.float32) * 0.001

        if enc == "32fc1":
            raw = np.frombuffer(msg.data, dtype=np.float32).reshape(h, step // 4)
            return raw[:, :w].astype(np.float32)

        raise RuntimeError(f"unsupported depth encoding: {msg.encoding}")

    def publish_cloud(self):
        now = time.time()
        if self.last_rgb is None or self.last_depth is None or self.last_info is None:
            return
        if now - self.last_rgb_stamp > 2.0 or now - self.last_depth_stamp > 2.0:
            return
        if now - self.last_info_stamp > 5.0:
            return

        rgb = self.last_rgb
        depth = self.last_depth
        info = self.last_info

        h = min(rgb.shape[0], depth.shape[0])
        w = min(rgb.shape[1], depth.shape[1])
        fx = float(info.k[0])
        fy = float(info.k[4])
        cx = float(info.k[2])
        cy = float(info.k[5])

        if fx <= 1e-6 or fy <= 1e-6:
            return

        points = []
        for v in range(0, h, self.stride):
            for u in range(0, w, self.stride):
                z = float(depth[v, u])
                if not math.isfinite(z) or z < self.depth_min_m or z > self.depth_max_m:
                    continue

                x = (float(u) - cx) * z / fx
                y = (float(v) - cy) * z / fy
                r, g, b = [int(c) for c in rgb[v, u]]
                rgb_u32 = (r << 16) | (g << 8) | b
                points.append((x, y, z, rgb_u32))

        if not points:
            return

        header = self.last_depth_header
        if header is None:
            return
        header.stamp = self.get_clock().now().to_msg()

        fields = [
            PointField(name="x", offset=0, datatype=PointField.FLOAT32, count=1),
            PointField(name="y", offset=4, datatype=PointField.FLOAT32, count=1),
            PointField(name="z", offset=8, datatype=PointField.FLOAT32, count=1),
            PointField(name="rgb", offset=12, datatype=PointField.UINT32, count=1),
        ]
        cloud = pc2.create_cloud(header, fields, points)
        self.pub.publish(cloud)


def main(args=None):
    rclpy.init(args=args)
    node = RgbdPointCloudNode()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
