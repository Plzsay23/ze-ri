#!/usr/bin/env python3
import math

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy, QoSDurabilityPolicy


def norm_angle(a):
    return math.atan2(math.sin(a), math.cos(a))


class ScanFrontFilter(Node):
    def __init__(self):
        super().__init__('scan_front_filter')

        self.declare_parameter('input_topic', '/scan')
        self.declare_parameter('output_topic', '/scan_front')

        # 로봇 기준 전방 각도
        self.declare_parameter('min_angle_deg', -135.0)
        self.declare_parameter('max_angle_deg', 135.0)

        # LiDAR가 base_link 기준 몇 도 돌아가 있는지
        # 반대로 설치했으면 180
        self.declare_parameter('lidar_yaw_deg', 180.0)

        # 로봇팔/차체 제거
        self.declare_parameter('min_keep_range', 0.30)
        self.declare_parameter('max_keep_range', 6.0)

        # slam_toolbox 에러 방지용 고정 길이
        self.declare_parameter('fixed_bins', 720)

        self.input_topic = self.get_parameter('input_topic').value
        self.output_topic = self.get_parameter('output_topic').value

        self.keep_min_angle = math.radians(float(self.get_parameter('min_angle_deg').value))
        self.keep_max_angle = math.radians(float(self.get_parameter('max_angle_deg').value))
        self.lidar_yaw = math.radians(float(self.get_parameter('lidar_yaw_deg').value))

        self.min_keep_range = float(self.get_parameter('min_keep_range').value)
        self.max_keep_range = float(self.get_parameter('max_keep_range').value)
        self.fixed_bins = int(self.get_parameter('fixed_bins').value)

        self.out_angle_min = -math.pi
        self.out_angle_max = math.pi
        self.out_angle_increment = (self.out_angle_max - self.out_angle_min) / (self.fixed_bins - 1)

        sub_qos = QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=10,
            durability=QoSDurabilityPolicy.VOLATILE,
        )

        pub_qos = QoSProfile(
            reliability=QoSReliabilityPolicy.RELIABLE,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=10,
            durability=QoSDurabilityPolicy.VOLATILE,
        )

        self.sub = self.create_subscription(
            LaserScan,
            self.input_topic,
            self.cb,
            sub_qos,
        )

        self.pub = self.create_publisher(
            LaserScan,
            self.output_topic,
            pub_qos,
        )

        self.get_logger().info(
            f'Filtering {self.input_topic} -> {self.output_topic}, '
            f'robot-front angle [{math.degrees(self.keep_min_angle):.1f}, {math.degrees(self.keep_max_angle):.1f}] deg, '
            f'lidar_yaw={math.degrees(self.lidar_yaw):.1f} deg, '
            f'range [{self.min_keep_range:.2f}, {self.max_keep_range:.2f}] m, '
            f'fixed_bins={self.fixed_bins}'
        )

    def cb(self, msg: LaserScan):
        out = LaserScan()
        out.header = msg.header

        out.angle_min = self.out_angle_min
        out.angle_max = self.out_angle_max
        out.angle_increment = self.out_angle_increment
        out.time_increment = 0.0
        out.scan_time = msg.scan_time

        out.range_min = self.min_keep_range
        out.range_max = self.max_keep_range

        out.ranges = [float('inf')] * self.fixed_bins
        out.intensities = []

        for i, r in enumerate(msg.ranges):
            if not math.isfinite(r):
                continue

            if r < self.min_keep_range or r > self.max_keep_range:
                continue

            raw_angle = msg.angle_min + i * msg.angle_increment
            raw_angle = norm_angle(raw_angle)

            # base_link 기준 각도 = laser raw angle + lidar yaw
            robot_angle = norm_angle(raw_angle + self.lidar_yaw)

            # 로봇 기준 전방 영역만 남김
            if robot_angle < self.keep_min_angle or robot_angle > self.keep_max_angle:
                continue

            # LaserScan 자체는 laser_frame 기준 angle로 유지해야 TF와 맞음
            idx = int(round((raw_angle - self.out_angle_min) / self.out_angle_increment))

            if idx < 0 or idx >= self.fixed_bins:
                continue

            if not math.isfinite(out.ranges[idx]) or r < out.ranges[idx]:
                out.ranges[idx] = r

        self.pub.publish(out)


def main():
    rclpy.init()
    node = ScanFrontFilter()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
