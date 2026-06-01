#!/usr/bin/env python3

import math
import time

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from geometry_msgs.msg import Twist
from sensor_msgs.msg import LaserScan
from std_msgs.msg import String, Bool, Float32


class VoiceStopGuardNode(Node):
    """
    Voice-follow 전용 safety guard.

    목적:
      - 회전 명령은 그대로 통과
      - 전진 명령은 전방 장애물이 너무 가까우면 정지
      - lateral avoidance(q/e)는 절대 만들지 않음

    Pipeline:
      /cmd_vel_raw
        -> voice_stop_guard_node
        -> /cmd_vel
        -> base_key_odom_serial_node
    """

    def __init__(self):
        super().__init__("voice_stop_guard_node")

        self.declare_parameter("input_cmd_topic", "/cmd_vel_raw")
        self.declare_parameter("output_cmd_topic", "/cmd_vel")
        self.declare_parameter("scan_topic", "/scan_front")

        self.declare_parameter("front_min_angle_deg", -25.0)
        self.declare_parameter("front_max_angle_deg", 25.0)

        self.declare_parameter("stop_distance", 0.65)
        self.declare_parameter("clear_distance", 0.85)

        self.declare_parameter("cmd_timeout_sec", 0.5)
        self.declare_parameter("scan_timeout_sec", 1.0)

        self.declare_parameter("state_topic", "/zeri/voice_stop_guard/state")
        self.declare_parameter("front_distance_topic", "/front_min_distance")
        self.declare_parameter("blocked_topic", "/obstacle_stop")

        self.input_cmd_topic = str(self.get_parameter("input_cmd_topic").value)
        self.output_cmd_topic = str(self.get_parameter("output_cmd_topic").value)
        self.scan_topic = str(self.get_parameter("scan_topic").value)

        self.front_min_angle_deg = float(self.get_parameter("front_min_angle_deg").value)
        self.front_max_angle_deg = float(self.get_parameter("front_max_angle_deg").value)

        self.stop_distance = float(self.get_parameter("stop_distance").value)
        self.clear_distance = float(self.get_parameter("clear_distance").value)

        self.cmd_timeout_sec = float(self.get_parameter("cmd_timeout_sec").value)
        self.scan_timeout_sec = float(self.get_parameter("scan_timeout_sec").value)

        self.last_cmd = Twist()
        self.last_cmd_time = 0.0

        self.front_min_distance = float("inf")
        self.last_scan_time = 0.0
        self.blocked = False

        self.cmd_pub = self.create_publisher(Twist, self.output_cmd_topic, 10)
        self.state_pub = self.create_publisher(
            String,
            str(self.get_parameter("state_topic").value),
            10,
        )
        self.front_pub = self.create_publisher(
            Float32,
            str(self.get_parameter("front_distance_topic").value),
            10,
        )
        self.blocked_pub = self.create_publisher(
            Bool,
            str(self.get_parameter("blocked_topic").value),
            10,
        )

        scan_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
        )

        self.create_subscription(Twist, self.input_cmd_topic, self.on_cmd, 10)
        self.create_subscription(LaserScan, self.scan_topic, self.on_scan, scan_qos)

        self.timer = self.create_timer(0.1, self.on_timer)

        self.get_logger().info(
            "voice stop guard started: "
            f"{self.input_cmd_topic} -> {self.output_cmd_topic}, "
            f"scan={self.scan_topic}, stop={self.stop_distance}, clear={self.clear_distance}"
        )

    def on_cmd(self, msg: Twist):
        self.last_cmd = msg
        self.last_cmd_time = time.time()

    def on_scan(self, msg: LaserScan):
        min_a = math.radians(self.front_min_angle_deg)
        max_a = math.radians(self.front_max_angle_deg)

        best = float("inf")

        angle = msg.angle_min
        for r in msg.ranges:
            if min_a <= angle <= max_a:
                if math.isfinite(r) and msg.range_min <= r <= msg.range_max:
                    if r < best:
                        best = r
            angle += msg.angle_increment

        self.front_min_distance = best
        self.last_scan_time = time.time()

        if best < self.stop_distance:
            self.blocked = True
        elif best > self.clear_distance:
            self.blocked = False

    def publish_state(self, text: str):
        msg = String()
        msg.data = text
        self.state_pub.publish(msg)

    def on_timer(self):
        now = time.time()

        front_msg = Float32()
        front_msg.data = float(self.front_min_distance)
        self.front_pub.publish(front_msg)

        blocked_msg = Bool()
        blocked_msg.data = bool(self.blocked)
        self.blocked_pub.publish(blocked_msg)

        cmd_age = now - self.last_cmd_time
        scan_age = now - self.last_scan_time

        out = Twist()

        if cmd_age > self.cmd_timeout_sec:
            self.cmd_pub.publish(out)
            self.publish_state("NO_CMD stop")
            return

        if scan_age > self.scan_timeout_sec:
            # scan이 죽으면 안전상 전진은 막고 회전만 허용
            out.angular.z = self.last_cmd.angular.z
            self.cmd_pub.publish(out)
            self.publish_state("NO_SCAN turn_only")
            return

        wants_forward = self.last_cmd.linear.x > 0.01

        if wants_forward and self.blocked:
            # 핵심: 횡이동 대신 정지.
            # 사람/장애물이 가까우면 q/e가 아니라 x.
            self.cmd_pub.publish(Twist())
            self.publish_state(
                f"FRONT_BLOCKED stop front={self.front_min_distance:.2f}"
            )
            return

        # 회전/후진/전진 모두 통과하되 lateral은 제거
        out.linear.x = self.last_cmd.linear.x
        out.linear.y = 0.0
        out.angular.z = self.last_cmd.angular.z

        self.cmd_pub.publish(out)

        self.publish_state(
            f"PASS vx={out.linear.x:.3f} wz={out.angular.z:.3f} "
            f"front={self.front_min_distance:.2f} blocked={int(self.blocked)}"
        )


def main(args=None):
    rclpy.init(args=args)
    node = VoiceStopGuardNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        try:
            node.cmd_pub.publish(Twist())
        except Exception:
            pass
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
