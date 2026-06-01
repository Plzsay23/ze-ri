#!/usr/bin/env python3

import math
import time

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist


class CmdVelSafetyNode(Node):
    """
    Temporary safety relay node.

    Input:
        /cmd_vel_raw

    Output:
        /cmd_vel

    This node does not use LiDAR yet.
    It only:
      1. clamps velocity,
      2. republishes command,
      3. sends zero velocity when input command times out.

    Later, LiDAR obstacle stop logic will be added in zeri_lidar/lidar_guard_node.py.
    """

    def __init__(self) -> None:
        super().__init__("cmd_vel_safety_node")

        self.declare_parameter("input_topic", "/cmd_vel_raw")
        self.declare_parameter("output_topic", "/cmd_vel")
        self.declare_parameter("publish_rate_hz", 20.0)
        self.declare_parameter("command_timeout_sec", 0.5)

        self.declare_parameter("max_linear_x", 0.30)
        self.declare_parameter("max_linear_y", 0.30)
        self.declare_parameter("max_angular_z", 1.50)

        self.input_topic = str(self.get_parameter("input_topic").value)
        self.output_topic = str(self.get_parameter("output_topic").value)

        self.publish_rate_hz = float(self.get_parameter("publish_rate_hz").value)
        self.command_timeout_sec = float(
            self.get_parameter("command_timeout_sec").value
        )

        self.max_linear_x = float(self.get_parameter("max_linear_x").value)
        self.max_linear_y = float(self.get_parameter("max_linear_y").value)
        self.max_angular_z = float(self.get_parameter("max_angular_z").value)

        self.last_cmd_time = 0.0
        self.current_cmd = Twist()

        self.sub = self.create_subscription(
            Twist,
            self.input_topic,
            self.on_cmd_vel_raw,
            10,
        )

        self.pub = self.create_publisher(
            Twist,
            self.output_topic,
            10,
        )

        period = 1.0 / max(self.publish_rate_hz, 1.0)
        self.timer = self.create_timer(period, self.on_timer)

        self.get_logger().info("cmd_vel_safety_node started")
        self.get_logger().info(f"  input_topic={self.input_topic}")
        self.get_logger().info(f"  output_topic={self.output_topic}")
        self.get_logger().info(f"  publish_rate_hz={self.publish_rate_hz}")
        self.get_logger().info(f"  command_timeout_sec={self.command_timeout_sec}")

    def on_cmd_vel_raw(self, msg: Twist) -> None:
        cmd = Twist()

        cmd.linear.x = self._clamp_finite(
            msg.linear.x,
            -self.max_linear_x,
            self.max_linear_x,
        )
        cmd.linear.y = self._clamp_finite(
            msg.linear.y,
            -self.max_linear_y,
            self.max_linear_y,
        )
        cmd.angular.z = self._clamp_finite(
            msg.angular.z,
            -self.max_angular_z,
            self.max_angular_z,
        )

        self.current_cmd = cmd
        self.last_cmd_time = time.monotonic()

    def on_timer(self) -> None:
        now = time.monotonic()

        if now - self.last_cmd_time > self.command_timeout_sec:
            self.pub.publish(Twist())
            return

        self.pub.publish(self.current_cmd)

    @staticmethod
    def _clamp_finite(value: float, low: float, high: float) -> float:
        value = float(value)

        if not math.isfinite(value):
            return 0.0

        if value < low:
            return low

        if value > high:
            return high

        return value


def main(args=None) -> None:
    rclpy.init(args=args)

    node = CmdVelSafetyNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
