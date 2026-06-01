#!/usr/bin/env python3

import math
from typing import List

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy
from sensor_msgs.msg import LaserScan


class ScanFrontFilterNode(Node):
    """
    /scan 전체 LiDAR 데이터를 받아서 로봇 전방 영역만 /scan_front로 발행한다.

    Input:
        /scan

    Output:
        /scan_front
    """

    def __init__(self) -> None:
        super().__init__("scan_front_filter_node")

        self.declare_parameter("input_topic", "/scan")
        self.declare_parameter("output_topic", "/scan_front")

        self.declare_parameter("min_angle_deg", -90.0)
        self.declare_parameter("max_angle_deg", 90.0)
        self.declare_parameter("lidar_yaw_deg", 180.0)

        self.declare_parameter("min_keep_range", 0.45)
        self.declare_parameter("max_keep_range", 6.0)
        self.declare_parameter("fixed_bins", 720)

        self.input_topic = str(self.get_parameter("input_topic").value)
        self.output_topic = str(self.get_parameter("output_topic").value)

        self.min_angle = math.radians(float(self.get_parameter("min_angle_deg").value))
        self.max_angle = math.radians(float(self.get_parameter("max_angle_deg").value))
        self.lidar_yaw = math.radians(float(self.get_parameter("lidar_yaw_deg").value))

        self.min_keep_range = float(self.get_parameter("min_keep_range").value)
        self.max_keep_range = float(self.get_parameter("max_keep_range").value)
        self.fixed_bins = int(self.get_parameter("fixed_bins").value)

        if self.fixed_bins < 2:
            raise ValueError("fixed_bins must be >= 2")

        if self.min_angle >= self.max_angle:
            raise ValueError("min_angle_deg must be smaller than max_angle_deg")

        scan_sub_qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
        )

        scan_pub_qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.VOLATILE,
        )

        self.pub = self.create_publisher(
            LaserScan,
            self.output_topic,
            scan_pub_qos,
        )

        self.sub = self.create_subscription(
            LaserScan,
            self.input_topic,
            self.on_scan,
            scan_sub_qos,
        )

        self.get_logger().info("scan_front_filter_node started")
        self.get_logger().info(f"  input_topic={self.input_topic}")
        self.get_logger().info(f"  output_topic={self.output_topic}")
        self.get_logger().info(f"  min_angle_deg={math.degrees(self.min_angle):.1f}")
        self.get_logger().info(f"  max_angle_deg={math.degrees(self.max_angle):.1f}")
        self.get_logger().info(f"  lidar_yaw_deg={math.degrees(self.lidar_yaw):.1f}")
        self.get_logger().info(f"  min_keep_range={self.min_keep_range}")
        self.get_logger().info(f"  max_keep_range={self.max_keep_range}")
        self.get_logger().info(f"  fixed_bins={self.fixed_bins}")

    def on_scan(self, msg: LaserScan) -> None:
        out = LaserScan()

        out.header = msg.header
        out.header.frame_id = msg.header.frame_id

        out.angle_min = self.min_angle
        out.angle_max = self.max_angle
        out.angle_increment = (self.max_angle - self.min_angle) / float(self.fixed_bins - 1)

        out.time_increment = msg.time_increment
        out.scan_time = msg.scan_time
        out.range_min = max(float(msg.range_min), self.min_keep_range)
        out.range_max = min(float(msg.range_max), self.max_keep_range)

        ranges: List[float] = [float("inf")] * self.fixed_bins
        intensities: List[float] = [0.0] * self.fixed_bins

        for i, raw_range in enumerate(msg.ranges):
            if not math.isfinite(raw_range):
                continue

            if raw_range < self.min_keep_range or raw_range > self.max_keep_range:
                continue

            raw_angle = msg.angle_min + float(i) * msg.angle_increment
            robot_angle = self._normalize_angle(raw_angle + self.lidar_yaw)

            if robot_angle < self.min_angle or robot_angle > self.max_angle:
                continue

            bin_index = int(round((robot_angle - self.min_angle) / out.angle_increment))

            if bin_index < 0 or bin_index >= self.fixed_bins:
                continue

            if raw_range < ranges[bin_index]:
                ranges[bin_index] = float(raw_range)

                if i < len(msg.intensities):
                    intensities[bin_index] = float(msg.intensities[i])

        out.ranges = ranges
        out.intensities = intensities

        self.pub.publish(out)

    @staticmethod
    def _normalize_angle(angle: float) -> float:
        while angle > math.pi:
            angle -= 2.0 * math.pi
        while angle < -math.pi:
            angle += 2.0 * math.pi
        return angle


def main(args=None) -> None:
    rclpy.init(args=args)

    node = ScanFrontFilterNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()