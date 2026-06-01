#!/usr/bin/env python3

import math
import time
from typing import Optional, Tuple

import numpy as np

import rclpy
from rclpy.node import Node
from rclpy.qos import (
    QoSProfile,
    ReliabilityPolicy,
    HistoryPolicy,
    DurabilityPolicy,
)

from geometry_msgs.msg import Twist
from sensor_msgs.msg import LaserScan, Image
from std_msgs.msg import Bool, Float32, Float32MultiArray, String


class LidarDepthGuardNode(Node):
    """
    /cmd_vel_raw + /scan_front + depth image -> /cmd_vel

    목적:
      - 정면 장애물 감지
      - 좌/우 중 더 널널한 방향으로 횡이동 회피
      - LiDAR와 depth를 같이 사용

    출력:
      /cmd_vel
      /avoid_state
      /front_min_distance
      /depth_avoid_scores
      /fused_avoid_scores
      /obstacle_stop
    """

    def __init__(self) -> None:
        super().__init__("lidar_depth_guard_node")

        # Topics
        self.declare_parameter("input_cmd_topic", "/cmd_vel_raw")
        self.declare_parameter("output_cmd_topic", "/cmd_vel")
        self.declare_parameter("scan_topic", "/scan_front")
        self.declare_parameter("depth_topic", "/zeri/vlm/input_depth")

        # Timings
        self.declare_parameter("publish_rate_hz", 20.0)
        self.declare_parameter("command_timeout_sec", 0.5)
        self.declare_parameter("scan_timeout_sec", 0.5)
        self.declare_parameter("depth_timeout_sec", 0.7)

        # Distances
        self.declare_parameter("stop_distance", 0.55)
        self.declare_parameter("clear_distance", 0.85)
        self.declare_parameter("side_min_clearance", 0.45)
        self.declare_parameter("max_valid_range", 6.0)

        # Laser sectors [deg]
        self.declare_parameter("front_min_angle_deg", -25.0)
        self.declare_parameter("front_max_angle_deg", 25.0)
        self.declare_parameter("left_min_angle_deg", 25.0)
        self.declare_parameter("left_max_angle_deg", 90.0)
        self.declare_parameter("right_min_angle_deg", -90.0)
        self.declare_parameter("right_max_angle_deg", -25.0)

        # Depth ROI ratios
        # top-view depth 화면에서 좌/정면/우 영역을 나누는 비율
        self.declare_parameter("depth_roi_y_min", 0.35)
        self.declare_parameter("depth_roi_y_max", 0.95)

        self.declare_parameter("depth_left_x_min", 0.05)
        self.declare_parameter("depth_left_x_max", 0.35)
        self.declare_parameter("depth_front_x_min", 0.35)
        self.declare_parameter("depth_front_x_max", 0.65)
        self.declare_parameter("depth_right_x_min", 0.65)
        self.declare_parameter("depth_right_x_max", 0.95)

        self.declare_parameter("depth_min_m", 0.15)
        self.declare_parameter("depth_max_m", 4.00)
        self.declare_parameter("depth_near_m", 0.80)
        self.declare_parameter("depth_sample_stride", 4)

        # Fusion
        self.declare_parameter("lidar_weight", 0.70)
        self.declare_parameter("depth_weight", 0.30)
        self.declare_parameter("use_depth", True)

        # Avoid behavior
        self.declare_parameter("avoid_lateral_speed", 0.20)
        self.declare_parameter("avoid_min_time_sec", 0.7)
        self.declare_parameter("avoid_max_time_sec", 3.0)
        self.declare_parameter("left_is_positive_y", True)

        self.declare_parameter("allow_backward_when_blocked", True)
        self.declare_parameter("allow_rotation_when_blocked", True)
        self.declare_parameter("stop_on_scan_timeout", True)

        # Logs
        self.declare_parameter("log_state_change", True)
        self.declare_parameter("log_scores", False)

        self.input_cmd_topic = str(self.get_parameter("input_cmd_topic").value)
        self.output_cmd_topic = str(self.get_parameter("output_cmd_topic").value)
        self.scan_topic = str(self.get_parameter("scan_topic").value)
        self.depth_topic = str(self.get_parameter("depth_topic").value)

        self.publish_rate_hz = float(self.get_parameter("publish_rate_hz").value)
        self.command_timeout_sec = float(self.get_parameter("command_timeout_sec").value)
        self.scan_timeout_sec = float(self.get_parameter("scan_timeout_sec").value)
        self.depth_timeout_sec = float(self.get_parameter("depth_timeout_sec").value)

        self.stop_distance = float(self.get_parameter("stop_distance").value)
        self.clear_distance = float(self.get_parameter("clear_distance").value)
        self.side_min_clearance = float(self.get_parameter("side_min_clearance").value)
        self.max_valid_range = float(self.get_parameter("max_valid_range").value)

        self.front_min_angle_deg = float(self.get_parameter("front_min_angle_deg").value)
        self.front_max_angle_deg = float(self.get_parameter("front_max_angle_deg").value)
        self.left_min_angle_deg = float(self.get_parameter("left_min_angle_deg").value)
        self.left_max_angle_deg = float(self.get_parameter("left_max_angle_deg").value)
        self.right_min_angle_deg = float(self.get_parameter("right_min_angle_deg").value)
        self.right_max_angle_deg = float(self.get_parameter("right_max_angle_deg").value)

        self.depth_roi_y_min = float(self.get_parameter("depth_roi_y_min").value)
        self.depth_roi_y_max = float(self.get_parameter("depth_roi_y_max").value)

        self.depth_left_x_min = float(self.get_parameter("depth_left_x_min").value)
        self.depth_left_x_max = float(self.get_parameter("depth_left_x_max").value)
        self.depth_front_x_min = float(self.get_parameter("depth_front_x_min").value)
        self.depth_front_x_max = float(self.get_parameter("depth_front_x_max").value)
        self.depth_right_x_min = float(self.get_parameter("depth_right_x_min").value)
        self.depth_right_x_max = float(self.get_parameter("depth_right_x_max").value)

        self.depth_min_m = float(self.get_parameter("depth_min_m").value)
        self.depth_max_m = float(self.get_parameter("depth_max_m").value)
        self.depth_near_m = float(self.get_parameter("depth_near_m").value)
        self.depth_sample_stride = int(self.get_parameter("depth_sample_stride").value)

        self.lidar_weight = float(self.get_parameter("lidar_weight").value)
        self.depth_weight = float(self.get_parameter("depth_weight").value)
        self.use_depth = bool(self.get_parameter("use_depth").value)

        self.avoid_lateral_speed = float(self.get_parameter("avoid_lateral_speed").value)
        self.avoid_min_time_sec = float(self.get_parameter("avoid_min_time_sec").value)
        self.avoid_max_time_sec = float(self.get_parameter("avoid_max_time_sec").value)
        self.left_is_positive_y = bool(self.get_parameter("left_is_positive_y").value)

        self.allow_backward_when_blocked = bool(
            self.get_parameter("allow_backward_when_blocked").value
        )
        self.allow_rotation_when_blocked = bool(
            self.get_parameter("allow_rotation_when_blocked").value
        )
        self.stop_on_scan_timeout = bool(
            self.get_parameter("stop_on_scan_timeout").value
        )

        self.log_state_change = bool(self.get_parameter("log_state_change").value)
        self.log_scores = bool(self.get_parameter("log_scores").value)

        self.latest_cmd = Twist()
        self.last_cmd_time = 0.0

        self.last_scan_time = 0.0
        self.front_min = math.inf
        self.lidar_left_score = 0.0
        self.lidar_right_score = 0.0
        self.lidar_front_score = 0.0

        self.last_depth_time = 0.0
        self.depth_left_score = 0.0
        self.depth_front_score = 0.0
        self.depth_right_score = 0.0
        self.depth_confidence = 0.0

        self.fused_left_score = 0.0
        self.fused_right_score = 0.0
        self.fused_front_score = 0.0

        self.avoid_state = "NORMAL"
        self.avoid_start_time = 0.0
        self.last_logged_state = None

        sensor_qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
        )

        self.cmd_sub = self.create_subscription(
            Twist,
            self.input_cmd_topic,
            self.on_cmd,
            10,
        )

        self.scan_sub = self.create_subscription(
            LaserScan,
            self.scan_topic,
            self.on_scan,
            sensor_qos,
        )

        self.depth_sub = self.create_subscription(
            Image,
            self.depth_topic,
            self.on_depth,
            sensor_qos,
        )

        self.cmd_pub = self.create_publisher(Twist, self.output_cmd_topic, 10)
        self.obstacle_pub = self.create_publisher(Bool, "/obstacle_stop", 10)
        self.front_min_pub = self.create_publisher(Float32, "/front_min_distance", 10)
        self.avoid_state_pub = self.create_publisher(String, "/avoid_state", 10)
        self.depth_score_pub = self.create_publisher(
            Float32MultiArray,
            "/depth_avoid_scores",
            10,
        )
        self.fused_score_pub = self.create_publisher(
            Float32MultiArray,
            "/fused_avoid_scores",
            10,
        )

        period = 1.0 / max(self.publish_rate_hz, 1.0)
        self.timer = self.create_timer(period, self.on_timer)

        self.get_logger().info("lidar_depth_guard_node started")
        self.get_logger().info(f"  input_cmd_topic={self.input_cmd_topic}")
        self.get_logger().info(f"  output_cmd_topic={self.output_cmd_topic}")
        self.get_logger().info(f"  scan_topic={self.scan_topic}")
        self.get_logger().info(f"  depth_topic={self.depth_topic}")
        self.get_logger().info(f"  use_depth={self.use_depth}")
        self.get_logger().info(f"  lidar_weight={self.lidar_weight}")
        self.get_logger().info(f"  depth_weight={self.depth_weight}")

    def on_cmd(self, msg: Twist) -> None:
        self.latest_cmd = msg
        self.last_cmd_time = time.monotonic()

    def on_scan(self, msg: LaserScan) -> None:
        self.last_scan_time = time.monotonic()

        front_values = self._sector_values(
            msg,
            self.front_min_angle_deg,
            self.front_max_angle_deg,
        )
        left_values = self._sector_values(
            msg,
            self.left_min_angle_deg,
            self.left_max_angle_deg,
        )
        right_values = self._sector_values(
            msg,
            self.right_min_angle_deg,
            self.right_max_angle_deg,
        )

        self.front_min = self._safe_min(front_values)
        self.lidar_front_score = self._range_score(front_values)
        self.lidar_left_score = self._range_score(left_values)
        self.lidar_right_score = self._range_score(right_values)

    def on_depth(self, msg: Image) -> None:
        if not self.use_depth:
            return

        depth = self._image_to_depth_m(msg)
        if depth is None:
            return

        self.last_depth_time = time.monotonic()

        h, w = depth.shape

        y0 = int(h * self.depth_roi_y_min)
        y1 = int(h * self.depth_roi_y_max)

        left = self._depth_roi(
            depth,
            int(w * self.depth_left_x_min),
            int(w * self.depth_left_x_max),
            y0,
            y1,
        )

        front = self._depth_roi(
            depth,
            int(w * self.depth_front_x_min),
            int(w * self.depth_front_x_max),
            y0,
            y1,
        )

        right = self._depth_roi(
            depth,
            int(w * self.depth_right_x_min),
            int(w * self.depth_right_x_max),
            y0,
            y1,
        )

        self.depth_left_score, left_conf = self._depth_score(left)
        self.depth_front_score, front_conf = self._depth_score(front)
        self.depth_right_score, right_conf = self._depth_score(right)

        self.depth_confidence = max(0.0, min(1.0, (left_conf + front_conf + right_conf) / 3.0))

        msg_out = Float32MultiArray()
        msg_out.data = [
            float(self.depth_left_score),
            float(self.depth_front_score),
            float(self.depth_right_score),
            float(self.depth_confidence),
        ]
        self.depth_score_pub.publish(msg_out)

    def on_timer(self) -> None:
        now = time.monotonic()

        cmd_alive = (now - self.last_cmd_time) <= self.command_timeout_sec
        scan_alive = (now - self.last_scan_time) <= self.scan_timeout_sec
        depth_alive = (now - self.last_depth_time) <= self.depth_timeout_sec

        if not cmd_alive:
            self._publish_cmd(self._stop_cmd(), obstacle=False, state="NO_CMD")
            return

        if not scan_alive:
            if self.stop_on_scan_timeout:
                self._publish_cmd(self._stop_cmd(), obstacle=True, state="NO_SCAN")
            else:
                self._publish_cmd(self.latest_cmd, obstacle=False, state="NO_SCAN_PASS")
            return

        self._update_fused_scores(depth_alive)

        raw = self.latest_cmd

        forward_requested = raw.linear.x > 0.03
        backward_requested = raw.linear.x < -0.03
        rotation_requested = abs(raw.angular.z) > 0.10

        if backward_requested and self.allow_backward_when_blocked:
            self._reset_avoid()
            self._publish_cmd(raw, obstacle=False, state="BACKWARD_PASS")
            return

        if rotation_requested and not forward_requested and self.allow_rotation_when_blocked:
            self._reset_avoid()
            self._publish_cmd(raw, obstacle=False, state="ROTATION_PASS")
            return

        if not forward_requested:
            self._reset_avoid()
            self._publish_cmd(raw, obstacle=False, state="PASS")
            return

        front_blocked = self.front_min <= self.stop_distance
        front_clear = self.front_min >= self.clear_distance

        if self.avoid_state in ("AVOID_LEFT", "AVOID_RIGHT"):
            elapsed = now - self.avoid_start_time

            if front_clear and elapsed >= self.avoid_min_time_sec:
                self._reset_avoid()
                self._publish_cmd(raw, obstacle=False, state="CLEAR_FORWARD")
                return

            if elapsed >= self.avoid_max_time_sec:
                self._reset_avoid()
            else:
                cmd = self._avoid_cmd_from_state()
                self._publish_cmd(cmd, obstacle=True, state=self.avoid_state)
                return

        if front_blocked:
            direction = self._choose_avoid_direction()

            if direction is None:
                self._publish_cmd(self._stop_cmd(), obstacle=True, state="FULL_BLOCK")
                return

            self._start_avoid(direction, now)
            self._publish_cmd(
                self._avoid_cmd_from_state(),
                obstacle=True,
                state=self.avoid_state,
            )
            return

        self._reset_avoid()
        self._publish_cmd(raw, obstacle=False, state="FORWARD_PASS")

    def _update_fused_scores(self, depth_alive: bool) -> None:
        use_depth_now = self.use_depth and depth_alive and self.depth_confidence > 0.10

        if not use_depth_now:
            self.fused_left_score = self.lidar_left_score
            self.fused_front_score = self.lidar_front_score
            self.fused_right_score = self.lidar_right_score
        else:
            dw = self.depth_weight * self.depth_confidence
            lw = self.lidar_weight

            denom = max(lw + dw, 1e-6)

            self.fused_left_score = (
                lw * self.lidar_left_score + dw * self.depth_left_score
            ) / denom

            self.fused_front_score = (
                lw * self.lidar_front_score + dw * self.depth_front_score
            ) / denom

            self.fused_right_score = (
                lw * self.lidar_right_score + dw * self.depth_right_score
            ) / denom

        out = Float32MultiArray()
        out.data = [
            float(self.fused_left_score),
            float(self.fused_front_score),
            float(self.fused_right_score),
            float(self.depth_confidence),
        ]
        self.fused_score_pub.publish(out)

        if self.log_scores:
            self.get_logger().info(
                "scores "
                f"lidar[L={self.lidar_left_score:.2f},F={self.lidar_front_score:.2f},R={self.lidar_right_score:.2f}] "
                f"depth[L={self.depth_left_score:.2f},F={self.depth_front_score:.2f},R={self.depth_right_score:.2f},C={self.depth_confidence:.2f}] "
                f"fused[L={self.fused_left_score:.2f},F={self.fused_front_score:.2f},R={self.fused_right_score:.2f}]"
            )

    def _choose_avoid_direction(self) -> Optional[str]:
        left_ok = self.fused_left_score > 0.25
        right_ok = self.fused_right_score > 0.25

        if left_ok and right_ok:
            if self.fused_left_score >= self.fused_right_score:
                return "AVOID_LEFT"
            return "AVOID_RIGHT"

        if left_ok:
            return "AVOID_LEFT"

        if right_ok:
            return "AVOID_RIGHT"

        return None

    def _start_avoid(self, state: str, now: float) -> None:
        self.avoid_state = state
        self.avoid_start_time = now

    def _reset_avoid(self) -> None:
        self.avoid_state = "NORMAL"
        self.avoid_start_time = 0.0

    def _avoid_cmd_from_state(self) -> Twist:
        cmd = Twist()

        lateral = abs(self.avoid_lateral_speed)

        if self.avoid_state == "AVOID_LEFT":
            cmd.linear.y = lateral if self.left_is_positive_y else -lateral
        elif self.avoid_state == "AVOID_RIGHT":
            cmd.linear.y = -lateral if self.left_is_positive_y else lateral

        cmd.linear.x = 0.0
        cmd.angular.z = 0.0

        return cmd

    @staticmethod
    def _stop_cmd() -> Twist:
        return Twist()

    def _publish_cmd(self, cmd: Twist, obstacle: bool, state: str) -> None:
        self.cmd_pub.publish(cmd)

        obstacle_msg = Bool()
        obstacle_msg.data = obstacle
        self.obstacle_pub.publish(obstacle_msg)

        front_msg = Float32()
        if math.isfinite(self.front_min):
            front_msg.data = float(self.front_min)
        else:
            front_msg.data = float("inf")
        self.front_min_pub.publish(front_msg)

        state_msg = String()
        state_msg.data = state
        self.avoid_state_pub.publish(state_msg)

        if self.log_state_change and state != self.last_logged_state:
            self.get_logger().info(
                f"state={state}, "
                f"front_min={self.front_min:.2f}, "
                f"fused_left={self.fused_left_score:.2f}, "
                f"fused_right={self.fused_right_score:.2f}, "
                f"depth_conf={self.depth_confidence:.2f}"
            )
            self.last_logged_state = state

    def _sector_values(
        self,
        msg: LaserScan,
        min_angle_deg: float,
        max_angle_deg: float,
    ) -> np.ndarray:
        min_angle_rad = math.radians(min_angle_deg)
        max_angle_rad = math.radians(max_angle_deg)

        if min_angle_rad > max_angle_rad:
            min_angle_rad, max_angle_rad = max_angle_rad, min_angle_rad

        values = []

        angle = msg.angle_min
        for r in msg.ranges:
            if min_angle_rad <= angle <= max_angle_rad:
                if self._valid_range(r, msg):
                    values.append(float(r))
            angle += msg.angle_increment

        if not values:
            return np.array([], dtype=np.float32)

        return np.array(values, dtype=np.float32)

    def _valid_range(self, r: float, msg: LaserScan) -> bool:
        if not math.isfinite(r):
            return False
        if r <= 0.0:
            return False

        range_min = msg.range_min
        range_max = msg.range_max

        if not math.isfinite(range_min) or range_min <= 0.0:
            range_min = 0.03

        if not math.isfinite(range_max) or range_max <= 0.0:
            range_max = self.max_valid_range

        range_max = min(range_max, self.max_valid_range)

        return range_min <= r <= range_max

    def _safe_min(self, values: np.ndarray) -> float:
        if values.size == 0:
            return self.max_valid_range
        return float(np.min(values))

    def _range_score(self, values: np.ndarray) -> float:
        """
        0~1 점수.
        높을수록 공간이 널널함.
        """
        if values.size == 0:
            return 0.0

        clipped = np.clip(values, 0.0, self.max_valid_range)

        clear_ratio = float(np.mean(clipped >= self.side_min_clearance))
        mean_norm = float(np.mean(clipped) / self.max_valid_range)
        p30_norm = float(np.percentile(clipped, 30) / self.max_valid_range)

        score = 0.45 * p30_norm + 0.35 * mean_norm + 0.20 * clear_ratio
        return max(0.0, min(1.0, score))

    def _image_to_depth_m(self, msg: Image) -> Optional[np.ndarray]:
        h = int(msg.height)
        w = int(msg.width)

        if h <= 0 or w <= 0:
            return None

        enc = msg.encoding.lower()

        try:
            if enc in ("16uc1", "mono16"):
                arr = np.frombuffer(msg.data, dtype=np.uint16).reshape(h, w)
                depth_m = arr.astype(np.float32) * 0.001
                return depth_m

            if enc in ("32fc1",):
                arr = np.frombuffer(msg.data, dtype=np.float32).reshape(h, w)
                return arr.astype(np.float32)

            self.get_logger().warn(f"unsupported depth encoding: {msg.encoding}")
            return None

        except Exception as exc:
            self.get_logger().warn(f"failed to decode depth image: {exc}")
            return None

    def _depth_roi(
        self,
        depth: np.ndarray,
        x0: int,
        x1: int,
        y0: int,
        y1: int,
    ) -> np.ndarray:
        h, w = depth.shape

        x0 = max(0, min(w - 1, x0))
        x1 = max(0, min(w, x1))
        y0 = max(0, min(h - 1, y0))
        y1 = max(0, min(h, y1))

        if x1 <= x0 or y1 <= y0:
            return np.array([], dtype=np.float32)

        stride = max(1, self.depth_sample_stride)
        roi = depth[y0:y1:stride, x0:x1:stride].reshape(-1)
        return roi.astype(np.float32)

    def _depth_score(self, values: np.ndarray) -> Tuple[float, float]:
        """
        return:
          score      0~1, 높을수록 공간이 널널함
          confidence 0~1, 유효 depth 비율
        """
        if values.size == 0:
            return 0.0, 0.0

        valid = values[np.isfinite(values)]
        valid = valid[(valid >= self.depth_min_m) & (valid <= self.depth_max_m)]

        confidence = float(valid.size / max(values.size, 1))

        if valid.size == 0:
            return 0.0, 0.0

        median_depth = float(np.median(valid))
        p30_depth = float(np.percentile(valid, 30))
        near_ratio = float(np.mean(valid <= self.depth_near_m))

        median_norm = max(0.0, min(1.0, median_depth / self.depth_max_m))
        p30_norm = max(0.0, min(1.0, p30_depth / self.depth_max_m))

        score = 0.45 * p30_norm + 0.35 * median_norm + 0.20 * (1.0 - near_ratio)
        score *= max(0.0, min(1.0, confidence * 1.5))

        return max(0.0, min(1.0, score)), max(0.0, min(1.0, confidence))


def main(args=None) -> None:
    rclpy.init(args=args)

    node = LidarDepthGuardNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()