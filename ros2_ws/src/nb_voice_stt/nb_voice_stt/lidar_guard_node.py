#!/usr/bin/env python3

import math
from typing import Optional

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from sensor_msgs.msg import LaserScan
from std_msgs.msg import String


class LidarGuardNode(Node):
    def __init__(self):
        super().__init__("lidar_guard_node")

        self.declare_parameter("scan_topic", "/scan")
        self.declare_parameter("input_cmd_topic", "/mode_cmd")
        self.declare_parameter("output_cmd_topic", "/safe_cmd")

        self.declare_parameter("stop_distance", 0.60)
        self.declare_parameter("side_clear_distance", 0.80)

        self.declare_parameter("front_half_angle_deg", 18.0)
        self.declare_parameter("left_min_angle_deg", 20.0)
        self.declare_parameter("left_max_angle_deg", 85.0)
        self.declare_parameter("right_min_angle_deg", -85.0)
        self.declare_parameter("right_max_angle_deg", -20.0)

        # 차폭 기반 회피 거리 설정
        self.declare_parameter("robot_width_m", 0.50)
        self.declare_parameter("avoid_margin_m", 0.10)
        self.declare_parameter("assumed_strafe_speed_mps", 0.20)

        self.declare_parameter("allow_strafe", True)
        self.declare_parameter("turn_duration_sec", 0.80)
        self.declare_parameter("settle_duration_sec", 0.20)
        self.declare_parameter("max_avoid_attempts", 5)
        self.declare_parameter("front_clear_cycles_required", 2)

        self.scan_topic = str(self.get_parameter("scan_topic").value)
        self.input_cmd_topic = str(self.get_parameter("input_cmd_topic").value)
        self.output_cmd_topic = str(self.get_parameter("output_cmd_topic").value)

        self.stop_distance = float(self.get_parameter("stop_distance").value)
        self.side_clear_distance = float(self.get_parameter("side_clear_distance").value)

        self.front_half_angle_deg = float(self.get_parameter("front_half_angle_deg").value)
        self.left_min_angle_deg = float(self.get_parameter("left_min_angle_deg").value)
        self.left_max_angle_deg = float(self.get_parameter("left_max_angle_deg").value)
        self.right_min_angle_deg = float(self.get_parameter("right_min_angle_deg").value)
        self.right_max_angle_deg = float(self.get_parameter("right_max_angle_deg").value)

        self.robot_width_m = float(self.get_parameter("robot_width_m").value)
        self.avoid_margin_m = float(self.get_parameter("avoid_margin_m").value)
        self.assumed_strafe_speed_mps = float(self.get_parameter("assumed_strafe_speed_mps").value)

        self.allow_strafe = bool(self.get_parameter("allow_strafe").value)
        self.turn_duration_sec = float(self.get_parameter("turn_duration_sec").value)
        self.settle_duration_sec = float(self.get_parameter("settle_duration_sec").value)
        self.max_avoid_attempts = int(self.get_parameter("max_avoid_attempts").value)
        self.front_clear_cycles_required = int(self.get_parameter("front_clear_cycles_required").value)

        self.cmd_pub = self.create_publisher(String, self.output_cmd_topic, 10)

        scan_qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
            reliability=ReliabilityPolicy.BEST_EFFORT,
        )

        self.scan_sub = self.create_subscription(
            LaserScan,
            self.scan_topic,
            self.scan_callback,
            scan_qos,
        )

        self.cmd_sub = self.create_subscription(
            String,
            self.input_cmd_topic,
            self.cmd_callback,
            10,
        )

        self.latest_front_min = float("inf")
        self.latest_left_min = float("inf")
        self.latest_right_min = float("inf")
        self.scan_ready = False

        self.current_desired_cmd = "X"
        self.last_safe_cmd = None

        self.avoid_state = "idle"   # idle / avoid_move / settle
        self.avoid_cmd = "X"
        self.avoid_attempts = 0

        # 와리가리 방지용: 한 번 고른 회피 방향 유지
        self.sticky_side = None  # "LEFT" / "RIGHT" / None

        self.front_clear_cycles = 0

        self.phase_timer = None
        self.settle_timer = None

        # 차폭 + 여유만큼 옆으로 피할 시간을 계산
        sidestep_distance = self.robot_width_m + self.avoid_margin_m
        speed = max(0.05, self.assumed_strafe_speed_mps)
        self.computed_strafe_duration_sec = sidestep_distance / speed

        self.get_logger().info("LidarGuardNode started")
        self.get_logger().info(f"scan_topic={self.scan_topic}")
        self.get_logger().info(f"input_cmd_topic={self.input_cmd_topic}")
        self.get_logger().info(f"output_cmd_topic={self.output_cmd_topic}")
        self.get_logger().info("LaserScan subscriber uses BEST_EFFORT QoS")
        self.get_logger().info(
            f"robot_width_m={self.robot_width_m:.2f}, "
            f"avoid_margin_m={self.avoid_margin_m:.2f}, "
            f"assumed_strafe_speed_mps={self.assumed_strafe_speed_mps:.2f}, "
            f"computed_strafe_duration_sec={self.computed_strafe_duration_sec:.2f}"
        )

    def publish_cmd(self, cmd: str):
        msg = String()
        msg.data = cmd
        self.cmd_pub.publish(msg)

        if cmd != self.last_safe_cmd:
            self.last_safe_cmd = cmd
            self.get_logger().info(
                f"safe_cmd={cmd} "
                f"(front={self.latest_front_min:.3f}, "
                f"left={self.latest_left_min:.3f}, "
                f"right={self.latest_right_min:.3f}, "
                f"state={self.avoid_state}, "
                f"sticky_side={self.sticky_side}, "
                f"clear_cycles={self.front_clear_cycles})"
            )

    def normalize_command(self, text: str) -> Optional[str]:
        s = text.strip().upper()
        if not s:
            return None

        valid = {"W", "A", "S", "D", "X", "Q", "E", "R", "T", "F", "G", "L", "M"}
        if s in valid:
            return s

        mapping = {
            "FORWARD": "W",
            "BACKWARD": "S",
            "LEFT": "A",
            "RIGHT": "D",
            "STOP": "X",
        }
        return mapping.get(s)

    def front_is_clear(self) -> bool:
        return self.latest_front_min > self.stop_distance

    def cancel_timers(self):
        if self.phase_timer is not None:
            self.phase_timer.cancel()
            self.destroy_timer(self.phase_timer)
            self.phase_timer = None

        if self.settle_timer is not None:
            self.settle_timer.cancel()
            self.destroy_timer(self.settle_timer)
            self.settle_timer = None

    def reset_avoidance(self, clear_sticky_side: bool = True):
        self.cancel_timers()
        self.avoid_state = "idle"
        self.avoid_cmd = "X"
        self.avoid_attempts = 0
        if clear_sticky_side:
            self.sticky_side = None

    def scan_callback(self, msg: LaserScan):
        front_min = float("inf")
        left_min = float("inf")
        right_min = float("inf")

        angle = msg.angle_min
        for r in msg.ranges:
            if math.isfinite(r) and (msg.range_min <= r <= msg.range_max):
                deg = math.degrees(angle)

                if abs(deg) <= self.front_half_angle_deg:
                    if r < front_min:
                        front_min = r

                if self.left_min_angle_deg <= deg <= self.left_max_angle_deg:
                    if r < left_min:
                        left_min = r

                if self.right_min_angle_deg <= deg <= self.right_max_angle_deg:
                    if r < right_min:
                        right_min = r

            angle += msg.angle_increment

        self.latest_front_min = front_min
        self.latest_left_min = left_min
        self.latest_right_min = right_min
        self.scan_ready = True

        if self.front_is_clear():
            self.front_clear_cycles += 1
        else:
            self.front_clear_cycles = 0

        # forward 중이고 아직 avoidance가 아닐 때만 새 회피 시작
        if self.current_desired_cmd == "W" and self.avoid_state == "idle":
            if not self.front_is_clear():
                self.get_logger().warn(
                    f"front blocked while moving forward: "
                    f"front={self.latest_front_min:.3f}, "
                    f"left={self.latest_left_min:.3f}, "
                    f"right={self.latest_right_min:.3f}"
                )
                self.start_avoidance()

    def choose_sticky_side(self):
        if self.sticky_side is not None:
            return self.sticky_side

        if self.latest_left_min >= self.latest_right_min:
            self.sticky_side = "LEFT"
        else:
            self.sticky_side = "RIGHT"

        return self.sticky_side

    def choose_avoid_cmd(self):
        side = self.choose_sticky_side()

        if side == "LEFT":
            side_clearance = self.latest_left_min
            if self.allow_strafe and side_clearance > self.side_clear_distance:
                return "Q", self.computed_strafe_duration_sec
            return "A", self.turn_duration_sec

        side_clearance = self.latest_right_min
        if self.allow_strafe and side_clearance > self.side_clear_distance:
            return "E", self.computed_strafe_duration_sec
        return "D", self.turn_duration_sec

    def cmd_callback(self, msg: String):
        cmd = self.normalize_command(msg.data)
        if cmd is None:
            self.get_logger().warn(f"unknown input cmd: {msg.data}")
            return

        self.current_desired_cmd = cmd

        # stop은 항상 최우선
        if cmd == "X":
            self.reset_avoidance(clear_sticky_side=True)
            self.publish_cmd("X")
            return

        # forward가 아닌 수동 명령은 그대로 통과
        if cmd in {"A", "S", "D", "Q", "E", "R", "T", "F", "G", "L", "M"}:
            self.reset_avoidance(clear_sticky_side=True)
            self.publish_cmd(cmd)
            return

        # 여기서부터 forward 처리
        if cmd == "W":
            # 회피 중에는 mode manager가 W를 계속 보내더라도
            # 회피 시퀀스를 끝낼 때까지 무시
            if self.avoid_state != "idle":
                return

            if not self.scan_ready:
                self.get_logger().warn("scan not ready yet, allowing forward temporarily")
                self.publish_cmd("W")
                return

            if self.front_is_clear():
                self.publish_cmd("W")
            else:
                self.start_avoidance()

    def start_avoidance(self):
        if self.avoid_state != "idle":
            return

        if self.avoid_attempts >= self.max_avoid_attempts:
            self.get_logger().warn("max avoid attempts reached -> force stop")
            self.publish_cmd("X")
            return

        self.avoid_attempts += 1
        self.avoid_state = "avoid_move"

        self.avoid_cmd, duration = self.choose_avoid_cmd()

        self.get_logger().warn(
            f"start avoidance #{self.avoid_attempts}: "
            f"sticky_side={self.sticky_side}, "
            f"avoid_cmd={self.avoid_cmd}, "
            f"front={self.latest_front_min:.3f}, "
            f"left={self.latest_left_min:.3f}, "
            f"right={self.latest_right_min:.3f}, "
            f"duration={duration:.2f}s"
        )

        # 먼저 정지 후, 충분히 길게 회피
        self.publish_cmd("X")
        self.publish_cmd(self.avoid_cmd)

        self.cancel_timers()
        self.phase_timer = self.create_timer(duration, self.finish_avoid_phase_once)

    def finish_avoid_phase_once(self):
        if self.phase_timer is not None:
            self.phase_timer.cancel()
            self.destroy_timer(self.phase_timer)
            self.phase_timer = None

        self.publish_cmd("X")
        self.avoid_state = "settle"
        self.settle_timer = self.create_timer(self.settle_duration_sec, self.finish_settle_once)

    def finish_settle_once(self):
        if self.settle_timer is not None:
            self.settle_timer.cancel()
            self.destroy_timer(self.settle_timer)
            self.settle_timer = None

        self.avoid_state = "idle"

        if self.current_desired_cmd != "W":
            self.get_logger().info("desired cmd is no longer forward -> stop avoidance flow")
            self.reset_avoidance(clear_sticky_side=True)
            return

        if not self.scan_ready:
            self.publish_cmd("W")
            self.reset_avoidance(clear_sticky_side=True)
            return

        # 앞이 연속으로 충분히 clear일 때만 전진 재개
        if self.front_clear_cycles >= self.front_clear_cycles_required:
            self.get_logger().info(
                f"front clear enough -> resume forward "
                f"(clear_cycles={self.front_clear_cycles})"
            )
            self.publish_cmd("W")
            self.reset_avoidance(clear_sticky_side=True)
        else:
            self.get_logger().warn(
                f"front still blocked or unstable after avoidance: "
                f"front={self.latest_front_min:.3f}, "
                f"left={self.latest_left_min:.3f}, "
                f"right={self.latest_right_min:.3f}, "
                f"clear_cycles={self.front_clear_cycles}, "
                f"retry with same side={self.sticky_side}"
            )
            self.start_avoidance()

    def destroy_node(self):
        try:
            self.publish_cmd("X")
        except Exception:
            pass
        self.cancel_timers()
        return super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = LidarGuardNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
